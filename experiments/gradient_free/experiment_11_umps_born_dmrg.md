# Experiment 11: uMPS Born-Machine LM with DMRG Sweeps (D=384, byte alphabet)

## Hypothesis
A translation-invariant uniform Matrix Product State (uMPS) Born-machine with bond dimension D=384, trained by 4–5 left-to-right two-site DMRG sweeps over ~50 MB of WikiText-103 bytes, clears val char-acc 0.55 within the 300 s budget and is structurally compute-bound on A100 (arithmetic intensity ~D ≈ 384 FLOPs/byte). It probably does not clear the 0.70 floor on the first attempt; this is a **capability demo** — the first byte-level WikiText uMPS number in the repo. The point is to establish whether the mechanism produces non-trivial char-acc at all.

## Motivation
No tensor-network LM has been submitted in this repo. The kernel-ridge family already failed at four distinct angles (rff_ridge, rff_linear_head, krr_ngram, nystrom_krr_hybrid, poly_tensorsketch_ridge — all 0.30–0.59 acc). The closed-form / linear-model story is exhausted. **A non-linear, compute-bound, backprop-free probabilistic model is the open gap.** uMPS is the canonical such model: exact normalization via the Born rule, multilinear (no chain-rule backprop), sweeps map to dense BLAS-3, with proven sequence-modeling capability on synthetic CFLs (Miller, Rabusseau, Terilla, AISTATS 2021).

## Method
Single core tensor `A: (D, V=256, D)` shared across positions (translation-invariant). Boundary vectors `L, R: (D,)` learned. Joint over a length-T window:
```
psi(c_1, ..., c_T) = L^T · A[:, c_1, :] · A[:, c_2, :] · ... · A[:, c_T, :] · R
P(c_1, ..., c_T) = psi^2 / Z,   Z = trace((sum_v A[:,v,:] kron A[:,v,:])^T)
```
Trained by two-site DMRG: at each adjacent pair of (conceptual) positions, contract `A · A → B: (D, V, V, D)`, form the local effective normal equations against the empirical (V,V) bigram tensor weighted by the left/right environments, solve a small linear system, SVD back to two cores with bond truncation to D. Sweep left-to-right then right-to-left. Numerically stabilized by carrying a `log_norm` scalar alongside every environment vector (arXiv:2510.00382 prescription) — environments are re-normalized every contraction step.

**Inference (CharModel)**: maintain running left-environment vector `L_t: (D,)` in fp32. `predict()` returns the length-256 distribution `|L_t · A[:, v, :] · R|^2 / sum_v(...)`. `observe(c)` does `L_t ← L_t · A[:, c, :]` followed by renormalization (and a log-norm carry, unused here). `reset()` sets `L_t = L`.

## Memory-Movement Analysis
- **Per-DMRG-update FLOPs**: forming the local effective matrix is one `(D·V) × (B·D)` GEMM and one `(D·V·V·D)` reshape; solving the local generalized least-squares is one `(D²V²) × (D²V²)` Cholesky (≈ 4 · D⁴ · V² FLOPs ≈ for D=384, V=256: 256 · 384⁴ · 1 ≈ 5.6 · 10¹³ FLOPs per update — way too expensive).
- **Truncation**: use a *batched* form where each update operates on a per-position mini-batch of B=4096 windows and the (V,V) bigram tensor is **chunked** by V to keep the working set in HBM. Effective cost per update: O(B · D² · V) for environment accumulation + O(D³ · V²) for the local solve, ≈ 4096 · 384² · 256 ≈ 1.5 · 10¹¹ FLOPs ≈ 0.5 s on A100 at 50 % peak. ~5 s per sweep over a 50 MB corpus broken into windows of length T=256. **Budget: 5 sweeps × 5 s = 25 s for sweeps + 30 s data prep + 5 s setup → ~60 s total. Massive headroom.**
- **Core tensor memory**: A is D·V·D · 4 B (fp32) = 384·256·384·4 = 150 MB. Fits in HBM with margin. L2 cache misses on the V-axis dominate; chunk V into 32-slice tiles (each tile 18.8 MB) so the working core fits in L2 (40 MB on A100).
- **Inference**: `predict()` is one (D × V × D) · (D,) contraction = D·V·D · 2 FLOPs = 75 MFLOPs per char, fully in L2 since L_t is 1.5 KB. ~30 µs / char on A100; 60K val chars → 1.8 s eval. Free.
- **Arithmetic intensity**: D ≈ 384 FLOPs/byte, just over A100's 156 ridge — **compute-bound**.

## Setup
- Dataset: WikiText-103 train split as raw bytes, V=256 alphabet, no embedding.
- Architecture: single core `A ∈ R^(D × 256 × D)` with D=384, fp32. Boundary vectors `L, R ∈ R^D`, fp32. Translation-invariant (one core for all positions).
- Training: 5 left-to-right two-site DMRG sweeps over the first 50 MB of training text broken into windows of length T=256 (≈ 200K windows). Each sweep does one local update per (conceptual) position pair in {0,…,T-2}, batched across all windows. Bond dim held fixed at D=384 (no adaptive growth in v1).
- Initialization: random Gaussian scaled to ensure E[‖A · v‖] ≈ ‖v‖ (identity-on-average init per Wall 2025); `L = R = ones / √D`.
- Baseline (already on leaderboard): `hopfield_layer` 40.2 kJ / 0.7293; `modded_nanogpt` 51.7 kJ / 0.7374.
- Reference for "expected ceiling": no precedent; literature on Born-MPS for natural-language is empty at byte level.

## Procedure
1. `mkdir submissions/umps_born_d384 && touch submissions/umps_born_d384/submission.py`.
2. Implement the core `train()` with three sub-functions: `init_core`, `build_environments(text_windows, A, L, R)` returning all left and right environments per window, and `dmrg_sweep(A, env_L, env_R, text_windows)` which updates A in place.
3. Implement `CharModel` with running left-env, prefetched right-env (set to R since uMPS is translation-invariant — there is no fixed end position in streaming inference).
4. Local test on 5 MB train slice / 1 K val chars: confirm sweep converges (log-likelihood monotonically increases) and `predict()` produces a valid distribution.
5. `python submit.py submissions/umps_born_d384 --yes`.
6. Record val char-acc, energy, wall-clock. Compare against `hopfield_layer` and `modded_nanogpt`.

## Success Criteria
- **Capability**: val char-acc ≥ 0.50 — first non-trivial byte-level uMPS LM number in the repo.
- **Surprise**: val char-acc ≥ 0.70 — clears the floor; first non-NN, fully-multilinear submission to pass.
- **Refutation**: val char-acc ≤ 0.35 (worse than bigram) — area-law on bytes is too tight at D=384; mechanism is structurally incapable of carrying English n-gram statistics without much larger D. Reconsider before scaling.

## Failure Modes & Diagnostics
- **Born-rule overflow on long chains** (arXiv:2510.00382: overflow at T=100 in two iterations): mitigated by log-norm carry. Diagnostic: assert `|log_norm|` increment per position < 5 during the first sweep.
- **Local solve becomes ill-conditioned** when the (V,V) bigram tensor has rank < D²: add a relative ridge `λ · trace(M)/D² · I` to the local normal equations.
- **Sweep direction asymmetry** (L-to-R favored over R-to-L because L_t accumulates first): alternate sweep direction every pass.
- **DMRG stalls at a local minimum after sweep 1** (Tang et al. 2025): retry with two-site updates allowed to *grow* D to D=512 if local condition number is good. Hold for v2 if v1 falls short.
- **fp32 core too large for L2**: tile the V axis (32 slices of 8 each) so working core is 18.8 MB.
- **Wall-clock blown by data movement, not compute**: profile sweep 1; if `<30 %` of time in matmul kernels, switch from naive Python sweep loop to a single `torch.einsum`-driven batched update.

## Estimated Cost
1 Modal A100-80GB run × ~5 min wall ≈ $0.10. Re-running with D ∈ {256, 512} would add 2 runs ≈ $0.30 total.

## References
- Miller, Rabusseau, Terilla 2021, "Tensor Networks for Probabilistic Sequence Modeling", AISTATS — uMPS, O(log T) sampling, synthetic CFL.
- Han, Wang, Fan, Wang, Zhang 2018, "Unsupervised Generative Modeling Using Matrix Product States", PRX 8, 031012 — DMRG-trained Born MPS, MNIST.
- Stoudenmire & Schwab 2016, "Supervised Learning with Tensor Networks", NeurIPS.
- Wall, Bevilacqua, Carleo 2025, "Initialization and training of matrix product state probabilistic models", arXiv:2505.06419 — identity-on-average init, log-norm carry, training-stability prescription.
- Glasser et al. 2019, "Expressive Power of Tensor-Network Factorizations for Probabilistic Modeling", NeurIPS — Born-MPS strictly more expressive than HMM at equal hidden dim.
- `research/non_nn_methods/spec_01_uniform_mps_born_machine.md` — the research-tier spec this experiment operationalizes.
