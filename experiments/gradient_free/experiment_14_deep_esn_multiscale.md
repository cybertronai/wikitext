# Experiment 14: Deep ESN with Multi-Scale Timescales (3-layer stack, ridge readout)

## Hypothesis
The two prior ESN failures (pass 1: N=8192 single-layer, 0.347 acc; pass 2: N=16384 with K=4 input, blew the 300 s budget) both used a *single* reservoir with one timescale. A **3-layer Deep-ESN** (Gallicchio & Micheli 2017) with explicit multi-scale leak rates (a₁=0.5, a₂=0.1, a₃=0.02) provides a hierarchical decomposition — fast layer for char-bigram features, mid layer for short-range syntactic structure, slow layer for word-boundary / sentence-level context. The concatenated state ψ = [x₁; x₂; x₃] gives the ridge readout a much richer feature space at *equal total reservoir size* (N_total = 3·N_each), which is the dimension that actually controlled ESN pass 1's accuracy. **Expected acc 0.50–0.62**, likely still below 0.70, but the first ESN result in repo that respects what reservoir computing literature has been doing since 2017.

## Motivation
DeepESN is the standard upgrade path for vanilla ESN when single-layer underfits. The Gallicchio & Micheli (Frontiers 2020, "Exploiting Multiple Timescales in Hierarchical Echo State Networks") result shows multi-timescale stacking captures dynamics that single-layer ESN cannot. ESN pass 1's 0.347 acc is *below bigram* (~0.50) on bytes — a clear sign the readout was starved of useful features. The fix is not "bigger N" (pass 2 showed throughput dies) but "structured N with hierarchical timescales."

## Method
Three stacked sparse reservoirs:
- **Layer 1**: N₁=4096, leak a₁=0.5, density 0.05, ρ=0.9. Input = K=2 most-recent bytes one-hot (D_in=512).
- **Layer 2**: N₂=4096, leak a₂=0.1, density 0.05, ρ=0.9. Input = x₁ (4096 → 4096 dense W_12).
- **Layer 3**: N₃=4096, leak a₃=0.02, density 0.05, ρ=0.85. Input = x₂ (4096 → 4096 dense W_23).

All W_in, W_res, W_inter layer-to-layer matrices are *fixed random*, sampled once. State update at each layer is the standard leaky-tanh:
```
x_l[t+1] = (1 - a_l) x_l[t] + a_l tanh(W_res_l · x_l[t] + W_in_l · u_l[t])
```
where u_l[t] is layer l's input. Readout features ψ_t = concat(x₁, x₂, x₃, u_in_t) of dimension 4096·3 + 512 = 12800. Closed-form ridge: `W_out = (Ψ^T Ψ + λI)^{-1} Ψ^T Y_onehot`.

The **multi-timescale design** is critical: a₁=0.5 means layer 1 forgets ~63% of its state in 1 step (fast bigram-level dynamics); a₃=0.02 means layer 3 retains 98% per step → effective memory of ~50 chars (~word-boundary timescale).

## Memory-Movement Analysis
- **State update FLOPs/step**: each layer needs one sparse N×N matvec at density 0.05 (~N²·0.05·2 FLOPs) + one dense N×N inter-layer matvec for layers 2,3 (~N²·2 FLOPs). For N=4096: sparse matvec ~1.7 MFLOPs (fast on A100 sparse path, ~10 µs), dense matvec ~33 MFLOPs (~50 µs). Three layers → ~180 µs/step Python overhead-dominated.
- **Streaming batched (B chunks in parallel)**: B=128 chunks, the dense layer-to-layer matvec becomes a (4096, 4096) × (4096, 128) GEMM = 4 GFLOPs ≈ 200 µs at A100 throughput. Sparse matvec batched is harder but doable via `torch.sparse_csr_tensor` × dense (4096, 128). Step time ~1 ms.
- **N_train budget**: 1 ms/step × 200 K steps = 200 s. Reserve 50 s for Cholesky. → **N_train = 200K · B = 25.6 M bytes** if we run at B=128, T=200K per chunk.
- **Ψ matrix**: (25.6 M, 12800) fp32 = 1.3 TB → does not fit. Need chunked accumulation directly into Ψ^T Ψ (12800, 12800) fp64 = 1.3 GB (fits) and Ψ^T Y (12800, 256) fp64 = 26 MB.
- **Cholesky** (12800, 12800) fp64: ~15 s on A100. Solve for 256 RHS: ~1 s.
- **Total budget**: 200 s stream + 15 s Cholesky + 5 s init + 30 s slack = 250 s. **Under 300 s with 50 s margin.**
- **Arithmetic intensity** of layer-to-layer dense matvec batched: 8 GFLOPs / (N²·4 + B·N·4) bytes ≈ 100 FLOPs/byte → bandwidth-bound but not catastrophically. Sparse matvec is heavily memory-bound regardless.

## Setup
- Train slice: first 25.6 M bytes (~25 MB) of WikiText-103 train.
- Reservoirs: 3 layers of N=4096 each; ρ=[0.9, 0.9, 0.85]; leak a=[0.5, 0.1, 0.02]; density 0.05.
- Input: K=2 byte one-hot, D_in=512.
- Ψ_t = concat(x₁, x₂, x₃, u_in) ∈ R^12800.
- Ridge λ = 1e-3 (relative to trace(Ψ^T Ψ)/12800).
- B=128 chunks streamed in parallel; washout = 200 bytes per chunk.
- Compare against: ESN pass 1 (0.347 / 29.8 kJ); ESN pass 2 (DQ time exceeded); hopfield_layer (0.7293 / 40.2 kJ).
- All matrices fp32; Cholesky in fp64.

## Procedure
1. `cp -r submissions/rff_ridge_v1 submissions/deep_esn_multiscale`.
2. Implement `build_reservoirs(seed)`: sample three sparse matrices, three dense input/inter matrices, rescale spectral radii via 30-step power iteration on dense probes.
3. Implement `stream_features(text, reservoirs, B)`: B-chunked streaming, per-step three sequential layer updates, accumulate Ψ^T Ψ and Ψ^T Y every 2000 inner steps.
4. Cholesky-solve W_out in fp64.
5. `CharModel`: maintain three reservoir states + K-byte history ring; `predict()` returns `softmax(W_out · ψ)` as decodable dict.
6. `python submit.py submissions/deep_esn_multiscale --yes`.

## Success Criteria
- **Capability**: val char-acc ≥ 0.50 — strictly above ESN pass 1 (0.347), establishing that multi-scale stacking *is* the lever.
- **Strong**: val char-acc ≥ 0.62 (the honest target pass 1 spec set), energy ≤ 30 kJ.
- **Surprise**: val char-acc ≥ 0.70 — clears floor. First reservoir-computing submission to pass.
- **Refutation**: val char-acc ≤ pass 1's 0.347 — multi-scale leak diversification does not unlock anything; ESN family is mechanically incapable at byte level. End line for the ESN direction.

## Failure Modes & Diagnostics
- **Wall-clock dies in Python streaming loop** (pass 2 failure): the per-step inner loop must be a single CUDA-graph-able sequence of three fused kernels. Pre-compile with `torch.compile(fullgraph=True)` on the inner step. If profiling shows >2 ms/step, drop N_train to 12 M (B=64).
- **Spectral radius computation slow** (30-step power iteration on three matrices): ~3 s total, fine.
- **Layer 3 saturation**: a=0.02 means x₃ may collapse to a single attractor over long sequences. Diagnostic: log var(x₃) at t=1000, 10000, 100000. If var monotonically decreasing → reservoir is too damped; increase a₃ to 0.05.
- **Inter-layer dense W_12, W_23 amplify or shrink signal**: rescale W_12, W_23 so output activation variance ≈ input activation variance (standard reservoir hygiene). Use scaling `‖W‖_2 ≈ 1/√N`.
- **Cholesky of (12800, 12800) OOM or numerically singular**: float64 should be fine (~1.3 GB), but if singular use `torch.linalg.lstsq(XtX_chol, XtY)`. Singular value diagnostic: log smallest singular value.
- **Concat-input redundancy in ψ**: u_in is already in W_in_1, so adding it to ψ may be redundant. If acc < 0.45, try ψ = concat(x₁, x₂, x₃) only.

## Estimated Cost
1 Modal A100-80GB run × ~5 min wall ≈ $0.10. A simpler 2-layer variant with N=8192 each as fallback would add another $0.10.

## References
- Gallicchio & Micheli 2017, "Deep Echo State Network (DeepESN): A Brief Survey", arXiv:1712.04323.
- Gallicchio, Micheli, Pedrelli 2020, "Exploiting Multiple Timescales in Hierarchical Echo State Networks", Frontiers in Applied Mathematics & Statistics.
- Lukoševičius & Jaeger 2009, "Reservoir computing approaches", Comput. Sci. Rev. — canonical ESN reference.
- `research/gradfree-survey/designs/method_esn-ridge-readout_pass_2.md` — analysis of the prior batched-streaming failure that informs the throughput design here.
- `research/gradfree-survey/results/method_esn-ridge-readout_pass_1.json` — 0.347 acc baseline to exceed.
