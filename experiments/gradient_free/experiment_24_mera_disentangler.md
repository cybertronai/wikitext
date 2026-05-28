# Experiment 20: MERA-2 (Tree TN + Disentangler Layer)

## Hypothesis
Inserting one layer of *disentanglers* between leaves and the bottom-tree level of a binary TTN (the defining feature of MERA, Vidal 2007) breaks the 1D area-law constraint on long-range correlations and yields polynomial — rather than exponential — correlation decay. For byte-level WikiText with T=64 window, this should give 5–10% absolute char-acc lift over experiment_23's pure TTN at the cost of one extra contraction layer. **Expected val char-acc 0.58–0.68**, with the upside scenario being 0.70+ if the disentangler layer absorbs intra-word byte correlations that the tree alone cannot.

## Motivation
TTN has strictly better correlation decay than MPS but still cannot exactly represent certain critical (long-range correlated) states. MERA adds horizontal "disentangler" unitaries between layers that explicitly transform pairs of adjacent leaves before tree-coarsening, capturing pairwise correlations that the tree's hierarchical coarse-graining alone misses. For language: characters within a word have strong pairwise dependencies (consonant-vowel, common bigrams) that should be handled by the disentangler layer, leaving the tree free to model word-level and longer structure. Sequence-modeling MERA literature is thin (Evenbly & Vidal 2009 is the theory; arXiv:1710.10248 Pestun-Vlassopoulos sketches an LM application but with no experiments), so any benchmark number here is a research first.

## Method
**Architecture**: T=64-byte window, 6-layer tree as in experiment_23, with **one disentangler layer between leaves and the first tree level**.

- **Leaf embeddings**: V=256 → D=64 dense linear (one-hot → embedding); 32 leaves at the base of the tree have V_local=256, projected to D=64.
- **Disentangler layer**: 32 disentangler tensors U_k ∈ R^(D × D × D × D) acting on each pair (2k, 2k+1) of leaves. These are 4-leg tensors (2 input legs, 2 output legs), constrained to be **isometric** (U U^† = I on input subspace).
- **Tree layer 1**: 16 tree tensors W_k ∈ R^(D × D × D), each combining two disentangler-outputs into one bond-D output.
- **Higher tree layers**: same as experiment_23, depth 5 above the disentangler layer.

Total params: disentanglers 32 · D⁴ = 32 · 64⁴ = 537 M (too many) — **reduce D=32**: 32 · 32⁴ = 33.5 M, tractable. Tree: 31 · D³ = 31 · 32³ = 1 M. Total ~35 M params.

**Training**: alternating sweeps over disentanglers and tree:
1. Fix all tree tensors. For each disentangler U_k, contract the rest of the network into an effective 4-leg tensor; SVD U_k onto its isometric constraint. (Evenbly 2009 "Algorithms for entanglement renormalization" — the canonical optimization procedure.)
2. Fix all disentanglers. Sweep tree as in experiment_23.
3. Repeat 2–3 times.

**Sliding-window AR inference**: as in experiment_23, mark the rightmost leaf as unmarginalized and contract the rest.

## Memory-Movement Analysis
- **Disentangler tensors at D=32**: 32 · 32⁴ · 4 B = 134 MB. Fits.
- **Per-step contraction (one window)**: O(T · D⁴) for disentangler layer + O(T · D³) for tree = T · D⁴ + T · D³ ≈ T · D⁴ ≈ 64 · 32⁴ = 67 M FLOPs/window. B=256 batched: 17 GFLOPs/batch. Eval over 60K chars: 60K · 67 M = 4 TFLOPs ≈ 0.05 s on A100. **Free.**
- **Training sweep over disentanglers**: per disentangler, the effective environment is (D × D × D × D × D × D × D × D) cube reduced over batch. With B=256, T=64, this is 256 · 64 · 32⁸ ≈ huge — cannot materialize. Mitigation: use Evenbly's iterative scheme — compute the SVD of the gradient direction, not the full effective tensor. Per-disentangler iteration: O(B · T · D⁵) ≈ 256 · 64 · 32⁵ = 550 GFLOPs ≈ 3 s per disentangler. 32 disentanglers · 3 alternation rounds = 300 s. **Tight — need to reduce.**
- **Practical knob**: D=24 instead of 32 → D⁴ drops 3.2×, D⁵ drops 4× → disentangler sweep cost ~80 s. **Use D=24** for v1.
- **Tree sweep**: same as experiment_23, ~5 s.
- **Total**: 80 s disentangler + 15 s tree (3 sweeps) + 30 s slack = 130 s training. **Under 300 s with margin.**
- **Arithmetic intensity**: D⁴ FLOPs / D³ bytes = D ≈ 24 FLOPs/byte — **bandwidth-bound**. Mitigation: use bf16 (cuts bytes in half) for forward pass, fp32 accumulator. Effective intensity 48 FLOPs/byte — still under A100 ridge. Accept bandwidth-bound; the cost is wall-clock, not energy.

## Setup
- T = 64-byte window, V = 256, **D = 24** (compromise for training cost).
- Architecture: 32 disentanglers (depth-0) + 6-level binary tree on top.
- Training: 3 alternation rounds, each round = 1 full disentangler sweep + 1 tree sweep.
- Data: 2 M training bytes broken into B=256 windows of T=64.
- Init: disentanglers = identity + small noise (so the model degenerates to TTN at init); tree tensors random isometric.
- Compare against: experiment_23 (pure TTN at D=128); experiment_11 (uMPS Born).

## Procedure
1. `mkdir submissions/mera2_d24_t64`.
2. Implement disentangler layer: 32 4-leg tensors stored as `(D, D, D, D)`. Forward applies pairwise: `disent_out_pair = einsum('ijkl,bi,bj->bkl', U_k, leaf[2k], leaf[2k+1])`.
3. Implement Evenbly-style isometric update on each disentangler:
   ```
   eff = compute_environment(U_k, other_tensors, training_windows)  # (D, D, D, D)
   U, _, V = torch.linalg.svd(eff.reshape(D*D, D*D))
   U_k_new = (U @ V).reshape(D, D, D, D)  # closest isometric
   ```
4. Tree sweep reused from experiment_23.
5. `CharModel`: same sliding-window protocol as experiment_23.
6. `python submit.py submissions/mera2_d24_t64 --yes`.

## Success Criteria
- **Primary**: val char-acc strictly above experiment_23 (TTN-only). Validates disentangler layer as a meaningful upgrade.
- **Strong**: val char-acc ≥ 0.65, energy ≤ 40 kJ.
- **Surprise**: val char-acc ≥ 0.70 — clears floor. First MERA-class submission.
- **Refutation**: val char-acc ≤ experiment_23 — disentangler layer at D=24 too small to add expressivity; the depth budget is better spent on more tree depth.

## Failure Modes & Diagnostics
- **Isometric constraint slowly drifts** (numerical SVD precision): re-isometrize every alternation round by SVD-projection.
- **Disentangler update non-monotonic in log-likelihood**: Evenbly's local update is not globally guaranteed to increase log-likelihood; if observed, fall back to one Riemannian gradient step on the isometric manifold.
- **D=24 too small to express byte bigram structure**: log per-pair MI(c_{2k}, c_{2k+1}) on training data; this should be carriable by D=24 in principle. If model achieves <0.45 acc, D is binding.
- **Environment-tensor computation is O(D^7) or worse** if implemented naively: use a careful contraction order (Evenbly 2009 algorithm 1).
- **Sliding-window evaluation breaks AR contract** if disentangler at the right boundary mixes future bytes: explicitly verify by checking P(c_t | c_<t) does not depend on c_t' for t' > t.

## Estimated Cost
1 Modal A100-80GB run × ~6 min wall ≈ $0.12 (most expensive in the portfolio). Variants D=32, T=128 would each add ~$0.15.

## References
- Vidal 2007, "Entanglement Renormalization", Phys. Rev. Lett. 99, 220405 / arXiv:cond-mat/0512165 — original MERA.
- Evenbly & Vidal 2009, "Algorithms for entanglement renormalization", Phys. Rev. B 79, 144108 / arXiv:0707.1454 — the optimization procedure used here.
- Pestun, Vlassopoulos 2017, "Tensor Network Language Model", arXiv:1710.10248 — sketches a MERA LM.
- Cheng, Wang, Xiang, Zhang 2019 — TTN baseline; MERA is the strict generalization.
- Companion: experiment_23 (TTN baseline), experiment_11 (uMPS Born).
