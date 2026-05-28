# Experiment 16: Dense GPU HMM with S=4096 Hidden States, Baum-Welch EM

## Hypothesis
A standard left-to-right discrete HMM with S=4096 hidden states, V=256 emission alphabet, trained by 3–5 iterations of GPU-batched Baum-Welch (EM) over 5–10 M bytes of WikiText-103, clears val char-acc 0.55 within the 300 s budget. The forward-backward recursion is dominated by a single (S × S) × (B × S) GEMM per step → fully Tensor-Core-bound at S=4096 (B·S²·2 ≈ 33 M·B FLOPs per step). HMMs predate RNNs as the canonical AR text model; the question is *what acc you reach if you scale S past what 1990s hardware allowed*.

## Motivation
Modern HMM implementations on GPU support >10⁴ states (wirelessinnovation.org HMM parallelization paper, 2011) but **no submission in this repo uses an HMM** — the closest is CTW (`ctw_d24`, DQ at 0.475). A dense GPU HMM is the *cleanest possible* gradient-free, non-NN AR baseline; it's also the **degenerate case of a non-negative MPS with bond D = S** (Glasser et al. 2019). If a vanilla S=4096 HMM hits 0.55, it sets the floor for what the MPS family (experiments 11, 12, 13, 18) must clear to justify their additional complexity. If it hits 0.70, it's a stronger result: a fully classical model on the leaderboard.

## Method
**Model**: P(c_1..c_T) = Σ_z π[z_1] · ∏_t A[z_{t-1}, z_t] · B[z_t, c_t], where π ∈ R^S, A ∈ R^(S×S), B ∈ R^(S×V). All in fp32; log-space for stability.

**Training (Baum-Welch)**: standard forward-backward over B parallel sequences of length T:
- Forward: α_t[s] = (Σ_s' α_{t-1}[s'] · A[s', s]) · B[s, c_t], with running log-norm c_t.
- Backward: β_t[s] = Σ_s' A[s, s'] · B[s', c_{t+1}] · β_{t+1}[s'].
- Posteriors γ_t[s] = α_t[s] · β_t[s] / Σ_s α_t[s] · β_t[s]; pairwise ξ_t[s, s'] = α_t[s] · A[s,s'] · B[s', c_{t+1}] · β_{t+1}[s'] / Z.
- M-step: A_new[s, s'] = Σ_(t,b) ξ_t[s, s'] / Σ_(t,b) γ_t[s]; B_new[s, v] = Σ_(t,b)[c_t=v] γ_t[s] / Σ γ_t[s]; π_new = γ_1 mean.

**Inference (CharModel)**: maintain running α_t ∈ R^S in log-space. `predict()` returns P(c_{t+1} = v | c_<=t) = Σ_s p(s | c_<=t) · B[s, v] where p(s | c_<=t) = α_t[s] / Σ α_t. `observe(c)` does α_{t+1}[s'] = (Σ_s α_t[s] · A[s, s']) · B[s', c], renormalized.

## Memory-Movement Analysis
- **Forward step FLOPs**: (B, S) × (S, S) GEMM = B·S²·2 FLOPs; for B=256, S=4096: 8.6 GFLOPs/step. Plus B·S elementwise × B[s, c_t]. For T=1024 sequence length: 8.8 TFLOPs/forward-pass per batch. At A100 70% peak (220 TFLOPs bf16, 100 TFLOPs fp32) ≈ 40 ms/batch forward.
- **Backward symmetric**: same cost.
- **M-step**: ξ accumulation across (T, B) is the dominant cost — but ξ is (S, S) reduced over (T, B), implemented as `einsum('tbs,tbsj->sj', alpha, A * B_obs_beta)`. Reduced output is (S, S) = 67 MB.
- **One EM iteration over 10 M bytes**: 10M / (B·T) = 10M / 262144 = 38 batches × ~120 ms (forward + backward + M-step) ≈ 5 s. **5 EM iterations: ~25 s. Plenty of headroom.**
- **Parameters**: A is 4096² · 4 B = 67 MB; B is 4096·256·4 = 4 MB; π is 16 KB. Tiny.
- **Memory peak**: α and β buffers are (B, T, S) = 256·1024·4096·4 = 4 GB each — fits comfortably on 80 GB. Use `torch.cuda.empty_cache()` between forward and backward to keep peak below 12 GB.
- **Arithmetic intensity**: S²·B·2 FLOPs / (S²·4 + B·S·4) bytes = 2B / (1 + 4/S) ≈ 2B ≈ 512 FLOPs/byte at B=256. **Strongly compute-bound on A100.**
- **Log-space arithmetic**: use `torch.logsumexp` for the forward `α_{t-1} · A` sum; cost is the same matmul but in log-space (logsumexp = max + log(sum(exp(x - max))). One log + one max + one matmul per step. Same flop count, slight memory bandwidth penalty.

## Setup
- S=4096 hidden states, V=256 emission alphabet.
- Init: π ~ Dirichlet(1), A ~ row-Dirichlet(0.1) (sparser prior), B ~ row-Dirichlet(0.5).
- Training: 5 Baum-Welch iterations over 10 M bytes of train, batch B=256, sequence length T=1024.
- All computation in fp32 with log-space stabilization.
- Compare against: `ctw_d24` (0.475 / 0.7 kJ); experiment_11 uMPS Born (pending) — *expected to outperform HMM if Glasser hierarchy holds in practice*.
- Reference points: classical 5-gram char model on PTB ≈ 0.60 acc; 1990s small-S HMM LMs ≈ 0.55 acc.

## Procedure
1. `mkdir submissions/dense_hmm_s4k && touch submissions/dense_hmm_s4k/submission.py`.
2. Implement `forward_pass(observations, log_A, log_B, log_pi)` returning log_alpha and log-likelihood. Use `torch.logsumexp(log_alpha[:, :, None] + log_A[None, :, :], dim=1)` for the matrix-vector log-sum.
3. Implement `backward_pass(observations, log_A, log_B)` symmetric.
4. Implement `m_step(log_alpha, log_beta, observations, log_A, log_B, log_pi)` → new (log_A, log_B, log_pi). Use einsum for ξ accumulation. Normalize rows.
5. EM loop: 5 iterations. Log train log-likelihood per iteration — should monotonically increase.
6. `CharModel`: maintain log_alpha ∈ R^S; `predict()` returns `softmax(log_B[s, :].T @ softmax(log_alpha))` (S → V projection); `observe(c)` updates log_alpha via one forward step.
7. `python submit.py submissions/dense_hmm_s4k --yes`.

## Success Criteria
- **Capability**: val char-acc ≥ 0.55 — strictly above CTW's 0.475, validating "vanilla HMM with modern S is competitive."
- **Strong**: val char-acc ≥ 0.65, energy ≤ 25 kJ.
- **Surprise**: val char-acc ≥ 0.70 — clears floor. **First HMM result above the floor in this repo would be a strong capability demo for the entire pre-NN era of LMs.**
- **Refutation**: val char-acc ≤ 0.50 — vanilla HMM mechanism is capped on bytes. *Expected to also bound MPS upside*: per Glasser 2019, non-negative MPS = HMM with S = D; if S=4096 fails, MPS at D=4096 should also fail.

## Failure Modes & Diagnostics
- **EM converges to a degenerate local optimum** (all mass on one state): log emission entropy per state. If <50% of states are active by iteration 3, raise emission Dirichlet prior to 1.0 and re-init.
- **log-space underflow at long T**: `torch.logsumexp` handles this. Sanity: log-likelihoods should be in [-T·log_V, 0] = [-T·5.5, 0]. If much more negative, log-space accounting has a bug.
- **fp32 vs fp64**: fp32 forward-backward is standard but may diverge at S=4096. Diagnostic: compare per-batch log-likelihood between fp32 and fp64 on a 100-byte slice; difference should be <1e-3. If not, switch to fp64 (acceptable since FLOPs are small).
- **M-step underflow** in ξ when γ approaches 0: clip γ ≥ 1e-30 before division.
- **Sequence length T=1024 too short** to see distant correlations: try T=2048 (memory doubles to 8 GB α buffer, still fits).
- **State permutation symmetry** across EM restarts: this is benign (model is invariant to state relabeling); just note that comparing A matrices across runs requires permutation alignment.

## Estimated Cost
1 Modal A100-80GB run × ~3 min wall ≈ $0.06. An S=8192 variant (matrix A becomes 268 MB, GEMM cost 4×) at the upper limit of 300 s would be ~$0.10. An EM=10-iteration variant if acc grows monotonically: ~$0.06.

## References
- Rabiner 1989, "A tutorial on hidden Markov models and selected applications in speech recognition", Proc. IEEE 77(2) — canonical Baum-Welch reference.
- Glasser, Sweke, Pancotti, Eisert, Cirac 2019, "Expressive Power of Tensor-Network Factorizations for Probabilistic Modeling", NeurIPS — proves non-negative MPS with bond D = HMM with S hidden states.
- Hymel & DePardo 2011, "Parallel Implementation of Hidden Markov Models for Wireless Applications", Wireless Innovation Forum — early GPU HMM, S up to 10⁴.
- NCBI PMC2430973, "Implementing EM and Viterbi algorithms for HMM in linear memory" — linear-memory forward-backward, useful if alpha/beta buffers blow.
- `submissions/ctw_d24/result.json` — nearest-neighbor classical AR submission (0.475 acc).
