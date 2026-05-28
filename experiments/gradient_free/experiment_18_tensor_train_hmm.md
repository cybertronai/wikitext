# Experiment 18: Tensor-Train HMM (Non-Negative MPS via Baum-Welch on Cores)

## Hypothesis
A tensor-train (TT) parameterization of an HMM transition matrix — equivalently, a non-negative MPS with bond dimension D — captures an effective S = D^L state space at O(L·D³·V) compute, where L is the TT-depth. With D=64, L=3, this gives S_eff = 262 K hidden states with the same compute cost as the dense S=4096 HMM in experiment_16. Trained by **Baum-Welch on the TT cores** (a structured M-step rather than dense matrix updates), it should clear the dense HMM's val char-acc and approach the floor at 0.70. This experiment is the **direct bridge between the HMM family (experiments 16, 17) and the MPS family (experiments 11, 12, 13)** — non-negative MPS *is* an HMM with bond as hidden dim (Glasser 2019).

## Motivation
Glasser et al. 2019 proves: non-negative MPS with bond dimension D ≡ HMM with S=D hidden states. The strict expressivity hierarchy is HMM ⊂ non-negative MPS ⊂ Born MPS ⊂ LPS. **Experiments 16 and 11 should give similar results when properly tuned, modulo expressivity differences from the squared-vs-non-negative parameterization.** The TT-HMM is the "fair" middle ground: non-negative parameterization (matches HMM semantics, no Born-rule normalization issues), structured cores (matches MPS compute pattern), and Baum-Welch training (matches HMM training). It produces a structurally clean comparison: if TT-HMM beats dense HMM at equal compute, the structured parameterization wins for natural-language statistics.

## Method
**Architecture**: Hidden state z_t lives in {0,…,D-1}^L (L=3 dims, D=64 each), effectively S_eff = D^L = 262144 states. Transition factorized as TT:
```
A[(i_1, i_2, i_3), (j_1, j_2, j_3)] = T_1[i_1, j_1, k_1] · T_2[k_1, i_2, j_2, k_2] · T_3[k_2, i_3, j_3]
```
where T_l ∈ R_+^(...) are non-negative TT cores with bond dimension R=32. Emission B[(i_1, i_2, i_3), v] factorized similarly: `B = E_1[i_1, m_1] · E_2[m_1, i_2, m_2] · E_3[m_2, i_3, v]` with emission TT bond M=16.

**Total parameters**: T cores: D²·R + D²·R² + D²·R ≈ 64²·32 + 64²·32² + 64²·32 = 131K + 4.2M + 131K = 4.5 M. E cores: D·M + D·M·M + D·M·V = 64·16 + 64·16² + 64·16·256 = 1K + 16K + 262K = 280 K. **Total ~5 M params** for an effective 262K-state HMM (vs 67 G params for dense S=262K).

**Training**: Standard Baum-Welch but the forward step `α @ A` becomes a structured TT contraction:
```
α_t ∈ R^D^L,    α_{t+1} = (α_t @ A) ⊙ B[:, c_{t+1}]   # ⊙ = elementwise
```
Implementing α @ A: contract α (viewed as a (D, D, D) tensor) against T_1, T_2, T_3 in three sequential matmuls, then materialize the (D, D, D) result. Cost per step: 3 · D²·R²·D = 3·D³·R² = 3·64³·32² = 800 MFLOPs/step. For B=256 batched: 200 GFLOPs/step ≈ 2 s per batched step at A100 70% peak. Wait — that's slow. Reconcile: the per-step cost is independent of S_eff because the TT contracts efficiently. 200 GFLOPs · T=256 = 50 TFLOPs per batch ≈ 0.5 s on A100. 76 batches × 5 EM iter = 200 s. **Tight but feasible.**

**M-step on TT cores**: This is the non-standard part. Maximize expected complete-data log-likelihood over the TT cores subject to non-negativity. Two options:
- (a) **Joint EM via factorized posteriors**: marginalize γ_t over each TT-bond axis separately; each core is updated by a separate moment-matching step. Standard for TT-HMM (Cui, Hong, Kachen-Kalchet, et al. 2016, "Tensor-train factorization of Hidden Markov Models").
- (b) **Alternating projection**: at each EM iteration, do a single dense ξ accumulation in the (D, D, D, D, D, D) space (too big — 262K² = 68 G entries) — *not feasible*. Use option (a).

Approximate non-negativity via clipping after each M-step update; renormalize cores so the implied A is row-stochastic (verified by sampling, not by exact projection).

## Memory-Movement Analysis
- **α buffer**: (B, T, D, D, D) = 256 · 512 · 64³ fp32 = 137 GB → does NOT fit. Mitigation: factorize α itself as a TT (low-rank approximation), or reduce L to 2 with D=128. Going with **L=2, D=128**: α buffer (B, T, D, D) = 256 · 512 · 128² · 4 = 8.6 GB, fits. S_eff = D² = 16384, comparable to experiment_17.
- **TT contraction step (L=2)**: per-step cost is D³·R² · 2 = 128³·32²·2 = 4.3 GFLOPs/step. B=256: 1.1 TFLOPs/batched step ≈ 5 ms. 76 batches · 5 EM iter · 2 (fwd+bwd) = 6 s. **Massive headroom for higher D or longer T.**
- **Parameters at L=2, D=128, R=32**: T cores: D²·R + D²·R = 2 · 16384·32 = 1 M. E cores: D·V·M + D·M = 128·256·16 + 128·16 = 524K + 2K = 526K. **Total 1.5 M params for S_eff=16384 HMM.** Dense S=16384 HMM has 268M params — TT is 180× smaller.
- **Arithmetic intensity**: TT contraction is sequence of small GEMMs; intensity ≈ R² ≈ 1000 FLOPs/byte — **strongly compute-bound**.

**Revised plan: L=2, D=128, R=32**.

## Setup
- L = 2 TT layers, D = 128 per axis, TT bond R = 32. S_eff = D² = 16384.
- V = 256 emissions, emission TT bond M = 16.
- 5 Baum-Welch iterations over 10 M bytes, B = 256, T = 512.
- Init: TT cores sampled non-negative (abs of Gaussian), normalized so the implied A is row-stochastic on a random sample of states.
- All fp32 with log-space forward-backward.
- Compare against: experiment_16 (dense S=4096 HMM); experiment_17 (block-diag S=16384); experiment_11 (uMPS Born D=384).

## Procedure
1. `mkdir submissions/tt_hmm_l2_d128`.
2. Implement TT contraction `tt_matvec(alpha, T_cores)`: alpha (B, D, D) → reshape (B, D, D) → contract along axis-2 with T_1[i_1, j_1, k_1] → (B, D, k_1, D) → contract with T_2[k_1, j_2, ·] → output (B, D, D).
3. Implement `forward_pass`, `backward_pass`, `m_step` on TT cores. M-step option (a): factorized-posterior moment matching.
4. EM loop: 5 iterations. Log train log-likelihood per iter.
5. `CharModel`: maintain alpha ∈ R^(D×D) log-space; `predict()` does TT contraction with emission cores to get (V,) distribution; `observe(c)` runs one TT-structured forward step.
6. `python submit.py submissions/tt_hmm_l2_d128 --yes`.

## Success Criteria
- **Capability**: val char-acc ≥ experiment_16's dense S=4096 result, despite using 180× fewer params per S_eff.
- **Strong**: val char-acc ≥ experiment_17's block-diag S=16384 result.
- **Surprise**: val char-acc ≥ 0.70 — clears floor. First TT-HMM submission anywhere on byte-level WikiText.
- **Refutation**: val char-acc ≤ experiment_16 — TT parameterization is too restrictive at R=32; the bond dim is the binding constraint and 16384 states cannot be carried.

## Failure Modes & Diagnostics
- **Non-negativity violated by M-step**: clip-and-renormalize; log fraction of clipped entries per iter. Should be <5% post-clip in steady-state EM.
- **TT contraction implementation bug**: verify on a tiny dense reference: D=4, L=2, write out the dense (16, 16) A explicitly, check forward-pass log-likelihood matches dense HMM log-likelihood up to fp32.
- **Approximate M-step does not maximize expected log-likelihood**: log train log-likelihood per EM iter; must be monotonically non-decreasing. If it drops, the factorized-posterior approximation is too coarse; switch to gradient-ascent on TT cores (still no chain rule through nonlinearities — TT contractions are multilinear).
- **EM converges in 1 iteration to a degenerate fixed point**: typical for low-rank parameterizations. Mitigate with random restarts (3 seeds, keep best).
- **Bond R=32 too small**: try R=64 (cost 4× per step, still 25 s total → fine).
- **Emission TT bond M=16 too small for 256-byte alphabet**: try M=32 (E_3 grows 2× → still tiny).

## Estimated Cost
1 Modal A100-80GB run × ~4 min wall ≈ $0.08. Variants R=64, L=3 with smaller D: each ~$0.08.

## References
- Glasser, Sweke, Pancotti, Eisert, Cirac 2019, "Expressive Power of Tensor-Network Factorizations for Probabilistic Modeling", NeurIPS — HMM ↔ non-negative MPS equivalence.
- Cui, Tao, Imamovic, Bittencourt-Silva, Strauss, Lyne 2016, "Tensor-Train Factorization of Hidden Markov Models", IEEE Trans. Signal Processing 64(16) — TT-HMM formulation and Baum-Welch on cores.
- Han, Wang, Fan, Wang, Zhang 2018, "Unsupervised Generative Modeling Using Matrix Product States", PRX — sampling and density estimation perspective.
- Rabiner 1989 — canonical Baum-Welch reference.
- Companion experiments: 11 (uMPS Born), 16 (dense HMM), 17 (block-diag HMM).
