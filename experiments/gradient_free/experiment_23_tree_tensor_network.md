# Experiment 19: Binary Tree Tensor Network (TTN) Born Machine LM

## Hypothesis
A balanced binary Tree Tensor Network (TTN) over a length-T=64 sliding window of bytes, bond dimension D=128, trained by two-site sweeps along the tree, captures longer-range byte correlations than an MPS of equal parameter count (Cheng, Wang, Xiang, Zhang 2019: TTN NLL 94.25 vs MPS 101.45 on binarized MNIST at D_max=100). For byte-level WikiText the long-range argument is even stronger — word boundaries and syntactic structure span dozens of bytes. **Expected val char-acc 0.55–0.68**, plausibly above the uMPS Born result (experiment_11) because the tree better matches the hierarchical structure of language. Used as a sliding-window AR model: P(c_t | c_{t-T+1}…c_{t-1}) computed by leaving the rightmost leaf marginalized.

## Motivation
MPS has a strict 1D area law — correlation between two positions decays as O(exp(-|i-j|/ξ)) with ξ ≤ log D. For a byte sequence in WikiText, word lengths are 3–7 bytes, sentence lengths 50–200 bytes. An MPS with D=384 can carry correlations across ~D ≈ a few hundred bytes barely — but TTN, with its log-depth tree, has *polynomially* decaying correlations between any two leaves, and is the structurally correct ansatz for hierarchical data. Cheng 2019 (PRB 99, 155131) demonstrates TTN > MPS at equal parameter count for natural images; the same argument applies to byte-level text where hierarchical structure (chars → words → sentences) dominates.

## Method
**Architecture**: balanced binary tree over T=64 leaves (each leaf = one byte position, V=256). Depth = log2(64) = 6. At each internal node, a 3-leg tensor contracts two children of bond dim D_below into one parent of bond dim D_above. Bond dimensions: D=128 throughout interior, V=256 at leaves (categorical one-hot input).

```
Leaf l: one-hot byte vector x_l ∈ R^256
Internal node n (with children L, R): tensor T_n ∈ R^(D_L × D_R × D_parent)
Root: tensor T_root ∈ R^(D_L × D_R × 1) — produces a scalar amplitude
```

The (Born-machine) probability over the 64-byte window:
```
psi(c_1, ..., c_64) = contract tree with leaves set to e_{c_i}
P(c_1, ..., c_64) = psi^2 / Z
```

**Sliding-window AR inference**: for position t > 64, predict c_t given c_{t-63}…c_{t-1}: set the first 63 leaves to observed bytes, leave leaf 64 as a free variable. The marginal P(c_64 = v | c_1..63) requires contracting the partial tree, which costs O(log T · D³ · V) ≈ 6·128³·256 ≈ 3 GFLOPs per inference step. At 60K val chars → 180 TFLOPs total ≈ 1.8 s on A100. Fast.

**Training (DMRG-style sweep on tree)**:
1. Pick a node, identify its environment: two child subtrees + parent subtree.
2. Form the local effective tensor by contracting child environments × parent environment against the empirical (D_L × D_R × D_parent) statistics.
3. Solve the local normal equations, SVD-truncate to bond D.
4. Sweep the tree top-down then bottom-up.

Cost per node update: O(D⁴) local solve = 128⁴ ≈ 268 M FLOPs. Tree has 2T-1 = 127 nodes. One full sweep ≈ 34 GFLOPs ≈ 0.5 s on A100. **5 sweeps = 2.5 s of GPU work**; the wall-clock will be dominated by environment management.

## Memory-Movement Analysis
- **Tree storage**: 127 internal-node tensors, each D³ = 128³·4 = 8 MB. Total ~1 GB. Fits.
- **Per-window environment caches**: for each window in the training batch, the partial contractions at each tree node are cached. (B, num_nodes, D, D, D) = B·127·D³ · 4 B. For B=256: 256 · 1 GB = 256 GB → **does not fit**. Mitigation: process one window at a time during sweep, recompute environments on-the-fly. Cost: O(B · log T) extra contractions per sweep step; manageable.
- **Per-step training FLOPs (batched)**: 5 sweeps × 127 nodes × O(B · D³ · V) ≈ 5 · 127 · 256 · 128³ · 256 ≈ 700 TFLOPs total. At A100 70% peak (220 TFLOPs bf16): 5 s. **Massive headroom.**
- **Sliding-window evaluation**: O(B · T · log T · D³ · V) for batched inference. For eval over 60K chars: 60K · 6 · D³ · V = 60K · 6 · 128³ · 256 = 200 GFLOPs ≈ 2 s. **Free.**
- **Arithmetic intensity**: D³ FLOPs against D² bytes = D ≈ 128 FLOPs/byte. Right at A100 ridge — **borderline compute-bound**. Increase to D=192 if first run is bandwidth-limited.

## Setup
- T = 64-byte window, V = 256, balanced binary tree (depth 6).
- Bond dim D = 128 internal, leaf input dim V = 256.
- Training: 5 two-site sweeps over the tree on 5 M bytes broken into B=256 windows of T=64 (≈80K windows total).
- Initialization: random isometric tensors (QR of Gaussian), boundary scalar at root.
- All fp32 throughout (cheap given parameter count).
- Compare against: experiment_11 (uMPS Born D=384) — *expected to be the closest comparator*; modded_nanogpt baseline.

## Procedure
1. `mkdir submissions/ttn_d128_t64`.
2. Implement tree topology as a list of `Node(parent, left_child, right_child, tensor)` objects. Tensors stored as `(D, D, D)` for internal, `(V, D)` for leaves' embedding (one-hot fixed, learned projection to D).
3. Implement `tree_contract(byte_window, tree)` returning `psi` (scalar amplitude).
4. Implement `sweep(tree, training_windows)`:
   - For each node n in top-down then bottom-up order:
     - Compute left-child environment, right-child environment, parent environment.
     - Form local effective tensor M ∈ R^(D × D × D × D × D × D) (3 input legs, 3 output legs after vectorization), reduced over the batch.
     - Solve the local least-squares for the new T_n; SVD-truncate.
5. `CharModel`: maintain a rolling buffer of last T-1 bytes; `predict()` does sliding-window evaluation with leaf 64 marginalized; `observe(c)` rolls the buffer.
6. `python submit.py submissions/ttn_d128_t64 --yes`.

## Success Criteria
- **Primary**: val char-acc ≥ experiment_11's uMPS Born result. Validates "TTN > MPS for byte LM."
- **Strong**: val char-acc ≥ 0.65, energy ≤ 30 kJ.
- **Surprise**: val char-acc ≥ 0.70 — clears floor with the first TTN LM in literature.
- **Refutation**: val char-acc ≤ experiment_11 — hierarchical TN parameterization does not pay off at T=64, D=128 on bytes (despite Cheng 2019's MNIST result). The tree's window structure may be wrong for streaming text.

## Failure Modes & Diagnostics
- **Sliding window violates AR property**: when computing P(c_t | c_{t-T+1}…c_{t-1}), only the rightmost leaf should be marginalized; verify the implementation by checking that for T=4, the tree result matches a hand-computed 4-byte joint. Test on toy distribution first.
- **Local DMRG solve numerical conditioning**: at deep tree nodes the effective tensor may be rank-deficient; add relative ridge `λ · trace(M)/D³ · I`.
- **Environment caches inconsistent during sweep**: when one node updates, all environments through it become stale. Recompute environments per-window per sweep direction — O(log T) extra work per step.
- **Tree memory footprint blows** if B kept in cache: process windows in mini-batches of 32 during sweep; recompute environments on-the-fly.
- **Born-rule normalization Z is hard for full tree**: compute Z by contracting the tree with all leaves set to identity (trace), then renormalize psi^2 by Z. For T=64, Z is a scalar; for the marginalized eval, Z_marg = Σ_v psi(c_<64, v)^2 is a per-window quantity.

## Estimated Cost
1 Modal A100-80GB run × ~5 min wall ≈ $0.10. Variants D=64 (cheap baseline), D=192 (richer), T=128 (longer window) — each ~$0.10.

## References
- Cheng, Wang, Xiang, Zhang 2019, "Tree tensor networks for generative modeling", PRB 99, 155131 / arXiv:1901.02217 — TTN > MPS on MNIST at equal D.
- Shi, Duan, Vidal 2006, "Classical simulation of quantum many-body systems with a tree tensor network", Phys. Rev. A 74, 022320 — TTN contraction algorithms.
- Liu, Zhang, Yang, Hou, Liu 2019, "Machine Learning by Two-Dimensional Hierarchical Tensor Networks" — relevant for image but methodology transfers.
- Companion: experiment_11 (uMPS Born), experiment_24 (MERA — more expressive cousin).
