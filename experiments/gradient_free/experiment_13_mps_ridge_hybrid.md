# Experiment 13: Small-D uMPS + Closed-Form Ridge Head Hybrid

## Hypothesis
A small uMPS with D=128 trained by a single DMRG sweep produces a sufficient bond-vector feature representation that a closed-form ridge-regression categorical head over the running left-environment can clear 0.55 char-acc in <60 s of training. The MPS provides the *nonlinear feature map* that previous closed-form ridge attempts (rff_ridge, krr_ngram — all DQ'd at 0.30–0.45) lacked. By keeping D small, the entire training fits in <100 s and energy stays well under 20 kJ.

## Motivation
The kernel-ridge family failed four times because **bilinear features over byte n-grams are insufficient** for byte-level English. RFF, Nyström, TensorSketch — all give linear or low-degree polynomial features of n-gram contexts. uMPS gives a *multilinear* feature: the running left-environment L_t ∈ R^D summarizes the entire prefix via a chain of matrix multiplications, with the expressivity hierarchy proven strictly above HMMs (Glasser 2019). Combining MPS feature extraction with closed-form ridge — instead of DMRG sweeps over the head — is a hybrid that **leverages the previously-failed ridge infrastructure** as a low-risk addition to the MPS direction. If experiment_11 lands at 0.50 with D=384 in 60 s, this should give comparable or better acc in less time.

## Method
1. **Random uMPS feature extractor**: sample core A ∈ R^(D × V × D), D=128, V=256 with the Wall 2025 identity-on-average init. **Do not DMRG-train it** — leave A and boundary L_0 random. The point is the multilinear *random projection*, analogous to ESN's fixed random reservoir, but with multilinear expressivity instead of leaky tanh.
2. **Stream training corpus through MPS**: for each position t in the corpus, compute the running left-environment L_t = (L_{t-1} · A[:, c_{t-1}, :]) / norm. After ~2 M streamed bytes, this yields a (2M, D=128) feature matrix Φ with normalized rows.
3. **Closed-form ridge head**: solve `W = (Φ^T Φ + λI)^{-1} Φ^T Y` where Y is the one-hot (2M, 256) next-byte matrix and λ = 1e-3 · trace(Φ^T Φ)/D. Cholesky in fp64 then cast back to fp32.
4. **Optional one-shot DMRG refinement** (held for v2 if v1 underperforms): after ridge, do one left-to-right DMRG sweep on A using the ridge head as a fixed observation operator.

## Memory-Movement Analysis
- **Feature streaming**: each step is one (D × V × D) gather + (1, D) × (D, D) matvec = D² FLOPs = 16K FLOPs/byte. For 2 M bytes: 3.3 · 10¹⁰ FLOPs ≈ 0.1 s on A100 if batched. The bottleneck is the Python streaming loop (cf. ESN pass 2 failure). Batched-streaming approach: split corpus into B=128 parallel chunks of 16K bytes each; per-step kernel updates all B left-envs in parallel via a single (D × V × D) · (B, D) → (B, D) contraction routed through a per-position byte gather. ~200 µs/step × 16K steps = 3.2 s. **Vastly under budget.**
- **Φ matrix**: (2M, 128) fp32 = 1 GB. Trivial on 80 GB.
- **Normal equations**: Φ^T Φ is (128, 128); negligible. Cholesky in 128 dims is ~1 ms. The dominant cost is one Φ^T · Y_onehot matmul = 2M · 128 · 256 · 2 FLOPs = 1.3 · 10¹¹ FLOPs ≈ 0.4 s.
- **Total training**: <10 s. **Plenty of slack to widen to N_train = 16M or D = 256.**
- **Arithmetic intensity** of feature streaming: D = 128 FLOPs/byte, right at the bandwidth/compute boundary on A100 — likely slightly memory-bound at this D. Mitigation: use bf16 cores with fp32 accumulator.

## Setup
- D = 128, V = 256, fp32 throughout.
- N_train = 4 M bytes (configurable up to 16 M if wall-clock allows).
- Streaming batched across B = 128 chunks.
- Ridge λ = 1e-3 (relative to trace).
- Init: Wall-2025 identity-on-average (sample A with column-norms ≈ 1 in the bond direction).
- Baseline comparisons: rff_ridge_v1 (0.364 / 2.6 kJ), experiment_11 (uMPS Born; pending), experiment_12 (AMPS; pending).

## Procedure
1. `cp -r submissions/rff_ridge_v1 submissions/mps_ridge_d128`. Replace the RFF projection with a streaming uMPS-feature extractor.
2. Implement `stream_features(text_bytes, A, L_0, B=128)` returning Φ ∈ R^(N, D) and labels y ∈ {0,…,255}^N. Use a Python loop only over T positions per chunk; vectorize across B.
3. Build XtX = Φ^T Φ, XtY = Φ^T · one_hot(y) in fp64 chunked accumulation as in ESN pass 1.
4. Solve W = cholesky_solve(XtY, cholesky(XtX + λ·trace/D · I)) in fp64; cast to fp32.
5. `CharModel`: maintain running L_t (fp32); `predict()` returns softmax of W · L_t over 256 bytes; `observe(c)` does L_t ← L_t · A[:, ord(c)%256, :] / norm; `reset()` sets L_t to L_0.
6. `python submit.py submissions/mps_ridge_d128 --yes`.

## Success Criteria
- **Primary**: val char-acc ≥ 0.55, **strictly above** the best ridge-family result (rff_linear_head, 0.588). This validates "MPS gives a richer feature than RFF/poly/Nyström."
- **Strong**: val char-acc ≥ 0.65, energy ≤ 10 kJ. Beats the 51.7 kJ baseline on energy by 5×.
- **Refutation**: val char-acc ≤ 0.45 — random uMPS features are no better than RFF; the bond-vector chain does not capture useful byte structure without DMRG training. Falls back to experiment_11/12.

## Failure Modes & Diagnostics
- **Bond vector L_t collapses to zero / explodes during streaming**: log `‖L_t‖` distribution at t = 100, 1000, 10000. With identity-on-average init and renormalization at every step, ‖L_t‖ = 1 by construction. If not, renorm step is buggy.
- **Φ^T Φ rank-deficient**: D=128 vs. effective rank of bond-vector trajectory may be <128 (collapsing to a low-dim attractor). Diagnostic: `torch.linalg.matrix_rank(XtX, rtol=1e-6)`. If rank < 64, increase A's stochasticity at init.
- **Per-byte python loop too slow**: same failure as ESN pass 2 (300 s wall). Mitigation: batch B=128 chunks, batch-stream B chunks in a *single* CUDA-graph-able loop. Profile sweep 1 of streaming; if step time > 1 ms, drop N_train to 1 M.
- **Char-acc plateau at 0.55** (close to rff_linear_head's 0.588): the random uMPS features are dominated by the unigram statistics carried in the bond axis. Add a one-shot DMRG sweep refining A → run as v2.

## Estimated Cost
1 Modal A100-80GB run × ~3 min wall ≈ $0.06. Variants D=64 (cheaper) and D=256 (richer) would each add ~$0.06.

## References
- Spec_01_uniform_mps_born_machine.md — operational template.
- Han 2018, Miller 2021, Wall 2025 (cited in spec_01).
- Lukoševičius & Jaeger 2009, "Reservoir computing approaches to recurrent neural network training", Comput. Sci. Rev. — the **methodological inspiration**: random nonlinear feature extractor + closed-form linear readout.
- rff_ridge_v1, rff_linear_head submissions — the failed ridge family this experiment supersedes.
