# ESN with Ridge-Regression Readout — Pass 2

**Direction chosen: A — Vectorised batched streaming + wider reservoir + K=4 short-history input.**

Pass 1 burned the full 300 s budget on a Python-loop streaming pass over 2 M bytes (~7100 steps/s) and finished at 0.347 char-acc — below even a naive bigram. Two findings drive pass 2: (i) throughput is the bottleneck — a batched-streaming GPU loop should free 50-100x compute headroom; (ii) one-hot byte input is information-starved at the readout, so we fold the last K bytes into the input to give the ridge solve immediate access to recent context. We keep the gradient-free, single-Cholesky-solve premise intact.

## 1. Hypothesis

Batched streaming lifts reservoir throughput from ~7 k to ~500 k effective steps/s, letting us (a) widen to N = 16384, (b) stream 16 M training bytes instead of 2 M (8x more rows for ridge), and (c) feed a richer K=4-byte short-history input. Predicted **val_char_acc 0.50-0.58** vs pass 1's 0.347 — a lift of +0.15-0.23. Still below the 0.62 honest-target of pass 1, but a defensible gradient-free baseline. We do not expect to reach 0.70.

## 2. Model

- **Encoding**: K = 4 most-recent UTF-8 bytes, each one-hot, concatenated -> input dim D_in = 1024.
- **Reservoir size N = 16384**, single reservoir.
- **W_res**: sparse (N, N), density 0.02 (~5.4 M nz), entries Uniform(-1, 1) at nz, rescaled to spectral radius rho = 0.9 via 30-step power iteration on a dense probe. Stored as `torch.sparse_csr_tensor` in float32.
- **W_in**: dense (N, 1024), entries Uniform(-0.4, +0.4).
- **Leak a = 0.3**, nonlinearity = tanh. State clamp [-5, 5] safety net.
- **Update**: `x_{t+1} = (1-a) x_t + a tanh(W_res @ x_t + W_in @ u_t)` where `u_t` is the K=4-byte one-hot concat.
- **Readout**: `logits = W_out @ phi_t`, where phi_t = concat(x_t, u_t) -> dim N + D_in = 17408; W_out shape (256, 17408). Including u_t in phi gives the readout a direct bigram path even if reservoir mixing is weak. Solved closed-form ridge.

## 3. Training procedure

Streaming is **batched across B=64 parallel chunks of the training corpus** — each chunk advances its own reservoir state independently, sharing W_res / W_in.

1. Encode train_text to uint8 on GPU. Take N_train = 16_000_000 bytes; split into B=64 contiguous chunks of 250_000 bytes each.
2. Initialise X_state (B, N) = 0, U_state (B, K, 256) = 0. Each chunk does its own washout = 1000 bytes.
3. Per-step inner loop (250 k iterations): gather `byte_b,t` for all B chunks (shape B), build u_t (B, 1024) by rolling/scattering the last K bytes one-hot, then **one batched sparse matmul** `W_res @ X_state.T` (sparse-dense matmul, B columns) + dense `W_in @ u_t.T`, apply tanh + leak. After warmup, append phi (concat x, u) (B, 17408) and label `byte_{t+1}` (B,) to per-chunk row buffers.
4. **Chunked normal-equations accumulation**: every 2000 inner steps we have a (B * 2000, 17408) = (128000, 17408) float32 block. Compute its `Phi.T @ Phi` (17408 x 17408 float64) and `Phi.T @ Onehot(y)` (17408 x 256 float64) and accumulate into running XtX, XtY. Drop the row block. Total rows accumulated: ~16 M.
5. Solve: `W_out.T = cholesky_solve(XtY, cholesky(XtX + lambda * trace(XtX)/N_phi * I))` in float64, cast back to float32.
6. `CharModel`: `reset()` zeros x and the K-byte history ring buffer; `observe(c)` runs one update per UTF-8 byte (eval is single-stream — see `wikitext.evaluate`, one `reset()` then per-char `observe`); `predict()` returns `softmax(W_out @ phi)` as decodable-byte dict.

## 4. Hyperparameters

- N = 16384, D_in = 1024, K = 4, B = 64, N_train = 16_000_000.
- density = 0.02, rho = 0.9, leak = 0.3, input_scale = 0.4.
- ridge lambda = 1e-3 (relative, scaled by trace(XtX)/N_phi).
- washout = 1000, accumulation block = 2000 steps.
- dtype: states float32, XtX/XtY float64.
- seed = `int(os.environ.get("SEED", 0))`.

## 5. Expected wall time (A100-80GB)

- Init + spectral radius: ~3 s.
- Streaming: 250 k inner steps, each does one (N x N, density 0.02) sparse matmul against B=64 dense columns (~7 GFLOP, ~0.4 ms on A100 sparse path) + dense (16384 x 1024) @ (1024 x 64) (~2 GFLOP, ~0.1 ms) + pointwise (~0.05 ms). ~0.6 ms / step * 250 k = **~150 s**. Effective throughput 16 M / 150 s = 107 k bytes/s (vs pass 1's ~7 k).
- XtX accumulation: 125 blocks of (128 k, 17408) -> Phi.T @ Phi ~ 80 GFLOP each ~ 80 ms on A100; total ~10 s.
- Cholesky (17408, 17408) float64: ~25 s. Solve for 256 RHS: ~2 s.
- **Total: ~190 s**, ~100 s headroom. If streaming slips, drop N_train to 12 M.

## 6. Success criterion

- **Target val_char_acc >= 0.52** (honest; pass 1 hit 0.347, bigram-table baseline ~0.50).
- **Stretch: 0.58.**
- **Energy: < 25 kJ** (lower than pass 1's 29.8 kJ because the GPU is doing useful work instead of waiting on Python).
- Single-shot. No retune to chase the bar.

## 7. Failure modes anticipated

- **Sparse-batched matmul slower than projected** (execution-bug): if step time > 1.0 ms, drop B to 32 and N_train to 8 M.
- **XtX conditioning at N_phi=17408** (design-failure): float64 + relative ridge; if Cholesky fails, fall back to `torch.linalg.lstsq` on the same normal equations.
- **OOM on the (17408, 17408) Cholesky** (~2.4 GB float64) and the per-block Phi (~9 GB float32) (execution-bug): both fit on 80 GB with margin.
- **Inter-chunk correlation inflating XtX rank-deficiency** (design-failure): B=64 chunks from one corpus are loosely correlated; relative ridge absorbs this. Not expected to break.
- **Acc still < 0.50** (design-failure): would mean the readout-from-reservoir story is fundamentally weak on byte-level English; we report and move on.

## 8. What we will NOT do

- NOT stack reservoirs (kept as separate future direction).
- NOT change the readout (still closed-form ridge, one solve).
- NOT learn W_in or W_res (gradient-free premise).
- NOT swap nonlinearity (still tanh).
- NOT do a hyperparameter sweep — single config, honest measurement.
- NOT batch evaluation (eval is a single stream per `wikitext.evaluate`; batching there would change semantics).

---

**Key lines**: direction = batched streaming (B=64) + N=16384 + K=4-byte input + phi includes input; success criterion **val_char_acc >= 0.52** (stretch 0.58), energy **< 25 kJ**.
