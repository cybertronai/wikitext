# ESN with Ridge-Regression Readout — Pass 1

## 1. Hypothesis

A fixed random sparse recurrent reservoir of leaky-tanh units, with only a linear softmax readout fit by closed-form ridge regression, can match small-transformer next-byte accuracy on WikiText-103 within the 300 s budget. The reservoir provides a high-dimensional nonlinear projection of byte history; the readout is a single (8192 x 256) matrix solved via Cholesky on the normal equations. We expect this to be very fast (single streaming pass + one solve), but to land in the 0.55-0.68 char-acc band — likely below the 0.70 bar, but worth measuring honestly as a strong gradient-free baseline.

## 2. Model

- **Encoding**: raw UTF-8 bytes (vocab = 256). No learned embedding; W_in is fixed random.
- **Reservoir size N = 8192** (single reservoir, no stacking).
- **Recurrent matrix W_res**: shape (N, N), density = 0.05 (sparse, stored as `torch.sparse_csr_tensor`), entries iid Uniform(-1, 1) at the nonzero positions, then rescaled so spectral radius rho(W_res) = 0.95. Spectral radius estimated by 30 iterations of power method on a dense float32 random probe (no full eig decomposition).
- **Input matrix W_in**: shape (N, 256), dense, entries iid Uniform(-input_scale, +input_scale), input_scale = 0.5.
- **Leak rate a = 0.3**; nonlinearity = tanh.
- **State update**: `x[t+1] = (1 - a) * x[t] + a * tanh(W_res @ x[t] + W_in[:, byte_t])` with x[0] = 0.
- **Readout**: `logits = W_out @ x[t]`, where W_out has shape (256, N). Softmax over 256 bytes for `predict()`.
- All state and matrices held on GPU in float32.

## 3. Training procedure

`train(train_text, valid_text=None)` does:

1. Encode `train_text` to a `torch.uint8` tensor on GPU.
2. Sample W_in, W_res with fixed seed (`os.environ["SEED"]` if set, else 0). Build W_res as sparse_csr. Rescale to rho = 0.95 via power iteration.
3. Streaming forward pass over N_train = 2_000_000 bytes:
   - Warmup (washout): first 1000 bytes update x but are NOT recorded.
   - For each subsequent byte t: compute x[t+1], then append x[t+1] (as the state used to predict byte at t+2) to row buffer X (shape M x N), and append `byte[t+2]` label to y (shape M,). Total M = N_train - 1001.
   - Keep X in float32, accumulate `XtX = X.T @ X` and `XtY_onehot = X.T @ onehot(y)` in float64 in **chunks of 50_000 rows** to bound peak memory: each chunk is 50_000 * 8192 * 4 B = 1.6 GB, well under 80 GB.
4. Solve ridge readout: `W_out.T = cholesky_solve(XtY, cholesky(XtX + lambda * I))`. Use `torch.linalg.cholesky` then `torch.linalg.cholesky_solve`, in float64, then cast W_out back to float32.
5. Wrap into a `CharModel`: `reset()` zeros x; `observe(c)` runs one state update per UTF-8 byte of c; `predict()` returns `softmax(W_out @ x)` as `{chr(b): p for b in 0..255 if decodable}`.

## 4. Hyperparameters

- N = 8192
- density = 0.05
- leak a = 0.3
- spectral radius rho = 0.95
- input_scale = 0.5
- ridge lambda = 1e-2 (scaled by trace(XtX)/N for numerical sanity)
- N_train = 2_000_000 bytes
- washout = 1000 bytes
- dtype: states float32, normal equations float64
- seed = `int(os.environ.get("SEED", 0))`

## 5. Expected wall time (A100-80GB)

- Reservoir init + spectral radius power iter: ~2 s.
- Streaming 2M states: each step is one sparse matvec (W_res @ x: ~0.05 ms at 5% density, N=8192) + dense W_in column lookup + tanh. At ~0.1 ms/step the Python loop dominates; batched chunked approach (process 4096 consecutive bytes per kernel launch via a Python loop with CUDA graphs is overkill — straight loop is ~200 s worst case). Budget: **180 s** for streaming.
- XtX accumulation: 40 chunks of (50_000, 8192) float32; each chunk's `X.T @ X` is ~5 GFLOP, ~10 ms on A100. Total: ~1 s.
- Cholesky of (8192, 8192) float64: ~3 s. Cholesky solve for 256 RHS: ~0.5 s.
- **Total: ~190 s**, leaving ~100 s headroom for tuning slack or longer N_train.

## 6. Success criterion

- **Target val_char_acc >= 0.62** on first 60K val chars (honest call: 0.70 is unlikely without architectural additions).
- **Stretch**: 0.68.
- **Energy**: <40 kJ (well under transformer baseline, since no backprop).
- We report whatever we get; we do NOT retune to chase the 0.70 bar.

## 7. Failure modes anticipated

- **Numerical conditioning of XtX** (design-failure): N=8192 with correlated states may have cond > 1e10; mitigation = float64 normal equations + relative ridge.
- **Exploding states** if rho mis-estimated by power iteration (execution-bug): clamp x to [-5, 5] after update as a safety net.
- **OOM during XtX accumulation** (execution-bug): chunked accumulation already addresses this; max live tensor is ~1.6 GB.
- **Slow Python streaming loop** blowing the 300 s budget (execution-bug): if profiling shows >250 s in the loop, fall back to N_train = 1_000_000.
- **Char-acc plateau in 0.55-0.62 band** (design-failure, expected): inherent to vanilla ESN on byte-level English; acknowledged.

## 8. What we will NOT do

- NOT iterate the readout (closed-form ridge, ONE solve, no SGD refinement).
- NOT stack reservoirs (single reservoir only; stacking would double cost and is not the bottleneck).
- NOT use a learned embedding or learned W_in (defeats the gradient-free premise).
- NOT use bigram/trigram input augmentation in pass 1 (kept as a future lever).
- NOT do an in-budget hyperparameter sweep (single config; honest single-shot evaluation).
- NOT switch to codepoint vocab (bytes are simpler and bounded at 256).

---

**Key lines**: reservoir N = 8192; success criterion val_char_acc >= 0.62 (stretch 0.68), energy <40 kJ.
