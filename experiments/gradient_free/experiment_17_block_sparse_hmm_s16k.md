# Experiment 17: Block-Diagonal Sparse HMM with S=16384 (Mixture-of-Experts HMM)

## Hypothesis
A block-diagonal HMM with K=4 blocks of S_k=4096 states each, plus a low-rank rank-r=64 cross-block coupling, behaves as an effective S=16384-state HMM at O(S²/K) ≈ K× cheaper compute than a dense S=16384 HMM. This **structured-sparsity** approach trades exact full-rank S²=268M params for K·S_k² + 2·S·r = 67M + 2M = 69M params with the same effective hidden capacity. Trained by GPU Baum-Welch in 3–5 EM iterations within 300 s. **Expected val char-acc ≥ experiment_16's (S=4096 dense) result** — block-diagonal-plus-low-rank is a strict generalization, with the low-rank coupling carrying inter-block correlations that bigram statistics demand.

## Motivation
Experiment_16 caps at S=4096 because dense S²=16.7 M transition entries × 5 EM iterations × M-step accumulation runs into memory bandwidth on the M-step. To reach S=16384 with a dense matrix requires 268 M entries — bigger but tractable. The *interesting* question is whether **block sparsity beats density at equal compute**: K=4 blocks of S_k=4096 each means within-block transitions are dense (capturing local context families) and across-block transitions are low-rank (capturing global syntactic categories). This mirrors the mixture-of-experts HMM in Tran et al. ("Unsupervised Neural Hidden Markov Models", 2016, arXiv:1609.09007) and the related "modular HMM" literature.

## Method
**Architecture**: Decompose transition matrix A ∈ R^(S × S), S=16384, K=4 blocks of S_k=S/K=4096:
```
A = block_diag(A_1, A_2, A_3, A_4) + U · V^T
```
where each A_k ∈ R^(S_k × S_k) is a dense sub-transition matrix and U, V ∈ R^(S × r=64) provide the low-rank cross-block coupling. Total params: K·S_k² + 2·S·r = 67 M + 2 M = 69 M (vs 268 M dense, 268 M / 4 = 67 M strict block-diag). Emission B ∈ R^(S × V=256) is dense, 4 M params.

**Constraint maintenance**: A must have row-sums = 1. After each M-step, project: rows of (block_diag + UV^T) → renormalize each row by its sum. The low-rank UV^T can introduce small negative entries (no constraint enforced during the rank update); clip max(0, ·) before renormalizing.

**Training**: Baum-Welch with the structured A. The forward step `(B, S) × (S, S) → (B, S)` is decomposed:
- Block-diag part: K = 4 independent (B, S_k) × (S_k, S_k) GEMMs, 4 · B · S_k² · 2 FLOPs = 4·B·4096²·2 = 134 · B GFLOPs.
- Low-rank part: ((B, S) × (S, r)) × (r, S) = 2 GEMMs of B·S·r·2 each = 2·B·16384·64·2 ≈ 4 · B GFLOPs.
- Total: 138 · B GFLOPs/step ≈ 35 GFLOPs at B=256 ≈ 200 ms/step on A100. T=512, EM=3 iter → ~ 100 s training.

**Inference**: same as experiment_16, with structured A. `predict()` returns Σ_s p(s | c_<=t) · B[s, v]. `observe(c)` does one forward step in 35 ms wall — feasible for 60K val chars: 60K · 35 ms / B_eval = 2100 s if B_eval=1. **Inference does NOT fit in 5 min wall.** Mitigation: precompute (B[s, c]) for the running c and process all 256 candidates per char by a single B-matmul of α; the forward update for `observe(c)` is one structured matvec of size S = 16K, taking 8.3 GFLOPs ≈ 0.5 ms on A100. Total eval: 30 s. **Fits.**

## Memory-Movement Analysis
- **A storage**: 4 · 4096² · 4 B = 268 MB for block-diag; 2 · 16384 · 64 · 4 = 8 MB for U, V. Total transition state ~280 MB. Fits trivially.
- **α buffer**: (B=256, T=512, S=16384) = 8 GB fp32. Marginal — use `T=256` if peak memory complains.
- **M-step ξ accumulation**: full ξ is S² = 268 M entries which is too big to materialize; instead accumulate block-by-block: 4 · S_k² ξ_k accumulators (67 M each) updated via einsum, and U/V are updated by SVD on the residual `(A_emp - block_diag(A_k_new))` projected to rank r=64. SVD on a 16K×16K matrix is 16K³ ≈ 4 TFLOPs ≈ 20 s — **the dominant cost per EM iter**. Mitigation: use truncated randomized SVD with target rank r=64 → O(S²·r) = 16K²·64·2 ≈ 33 GFLOPs ≈ 0.2 s.
- **Total per EM iter**: forward+backward 200 s for 10M bytes (above), M-step 0.5 s. Wait — that's too slow. Recompute: 10 M bytes / (B·T) = 10M / 131K = 76 batches × 200 ms forward + 200 ms backward = 30 s per iter. **3 EM iters: ~90 s. Plus ~5 s M-step. Plus 20 s slack.** Under 200 s training.
- **Arithmetic intensity**: dense block GEMM is fully compute-bound (B·S_k² FLOPs / S_k²·4 + B·S_k·4 ≈ B FLOPs/byte). Low-rank part: similar. **Strongly compute-bound at B=256.**

## Setup
- S = 16384, K=4 blocks of S_k=4096, low-rank coupling r=64.
- V = 256 emissions, dense.
- 3 Baum-Welch iterations over 10 M bytes, B=256, T=512.
- Init: π ~ Dirichlet(1); A_k ~ row-Dirichlet(0.1); U, V ~ Gaussian(0, 0.01); B ~ row-Dirichlet(0.5).
- All fp32 with log-space forward-backward.
- Compare against: experiment_16 (dense HMM S=4096, pending); experiment_18 (tensor-train HMM, pending).

## Procedure
1. `mkdir submissions/blockdiag_hmm_s16k`.
2. Implement `structured_matvec(alpha, A_blocks, U, V)`: 4 parallel (B, S_k) × (S_k, S_k) GEMMs via `bmm`, then add `(alpha @ V) @ U^T`.
3. Implement `forward_pass`, `backward_pass`, `m_step` reusing the structure.
4. M-step: for each block, accumulate block_diag ξ_k. The residual A_full_empirical - block_diag(A_k_new) is computed implicitly via low-rank decomposition by randomized SVD on the gradient direction.
5. EM loop: 3 iterations; log train log-likelihood per iter.
6. `CharModel`: `predict()` returns `softmax(B^T @ softmax(log_alpha))`; `observe(c)` runs one structured forward step.
7. `python submit.py submissions/blockdiag_hmm_s16k --yes`.

## Success Criteria
- **Primary**: val char-acc ≥ experiment_16's result. Validates "block-diagonal at S=16k ≥ dense at S=4k."
- **Strong**: val char-acc ≥ 0.65, energy ≤ 35 kJ.
- **Surprise**: val char-acc ≥ 0.70 — clears floor.
- **Refutation**: val char-acc ≤ experiment_16's — block-diagonal structure costs more than it gives, K=4 partition is too coarse, or 16K states is past HMM's expressivity ceiling on bytes.

## Failure Modes & Diagnostics
- **Low-rank UV^T violates row-stochastic constraint after M-step**: clip rows to non-negative, renormalize. Log fraction of rows where clipping changed >1% of mass.
- **Block partition is degenerate** (all 4 blocks learn identical local statistics): force initial block specialization by seeding each block A_k with a different sub-distribution over emissions. Diagnostic: emission entropy per block should differ post-EM.
- **α buffer memory** at (B=256, T=512, S=16384) = 8 GB: drop B to 128 if peak exceeds 30 GB.
- **M-step randomized SVD is unstable**: fall back to truncated power iteration (cheaper, more stable for r=64).
- **EM diverges due to mis-implementation of forward log-space update**: validate against experiment_16's forward by checking that S=4096 block-diag (K=1) gives identical log-likelihood to dense S=4096 HMM.

## Estimated Cost
1 Modal A100-80GB run × ~5 min wall ≈ $0.10. Variant: K=8 / S_k=2048 (more blocks, finer specialization): +$0.10. Variant: r=128 (richer cross-block coupling): +$0.10.

## References
- Tran, Bisk, Vaswani, Marcu, Knight 2016, "Unsupervised Neural Hidden Markov Models", arXiv:1609.09007 — neural reparameterization of HMM transition; we keep it classical.
- Buntine 2002, "Variational Extensions to EM and Multinomial PCA" — sparse multinomial M-step tricks.
- Halko, Martinsson, Tropp 2011, "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions" — randomized SVD foundations.
- Glasser et al. 2019, NeurIPS — HMM ↔ non-negative MPS equivalence; informs comparison with experiment_18.
- Companion: experiment_16 (dense S=4096 HMM); experiment_18 (tensor-train HMM = non-negative MPS).
