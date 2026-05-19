"""ESN with Ridge-Regression Readout — Pass 1.

A fixed random sparse recurrent reservoir of leaky-tanh units, with only a
linear softmax readout fit by closed-form ridge regression (Cholesky on the
normal equations). Byte-level vocab (256). Single reservoir, no stacking,
no learned embedding, no SGD.

Spec: .survey/designs/method_esn-ridge-readout_pass_1.md
"""
from __future__ import annotations

__author__ = "@survey-esn"

import os
import time

import torch
import torch.nn.functional as F

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Hyperparameters (from spec, do not vary)
# ---------------------------------------------------------------------------
N_RESERVOIR = 8192
DENSITY = 0.05
LEAK_A = 0.3
SPECTRAL_RADIUS = 0.95
INPUT_SCALE = 0.5
RIDGE_LAMBDA = 1e-2  # scaled by trace(XtX)/N
N_TRAIN = 2_000_000
WASHOUT = 1000
CHUNK_ROWS = 50_000
VOCAB = 256


# ---------------------------------------------------------------------------
# Reservoir construction
# ---------------------------------------------------------------------------

def _build_sparse_W_res(
    n: int, density: float, rho_target: float, generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Build sparse CSR W_res with given density and spectral radius."""
    nnz = int(n * n * density)
    # Sample indices via random permutation; for N=8192, n*n = 6.7e7 entries
    # and nnz ~ 3.35e6 — fine on GPU.
    flat_idx = torch.randperm(n * n, generator=generator, device=device)[:nnz]
    rows = flat_idx // n
    cols = flat_idx % n
    vals = (torch.rand(nnz, generator=generator, device=device) * 2.0 - 1.0).float()

    # Build dense matrix first to estimate spectral radius via power iteration,
    # then rescale and convert to sparse.
    W_dense = torch.zeros(n, n, dtype=torch.float32, device=device)
    W_dense[rows, cols] = vals

    # Power iteration on a dense float32 random probe.
    v = torch.randn(n, dtype=torch.float32, device=device, generator=generator)
    v = v / (v.norm() + 1e-12)
    for _ in range(30):
        v = W_dense @ v
        nv = v.norm()
        if nv < 1e-12:
            break
        v = v / nv
    # Rayleigh quotient: estimate of dominant eigenvalue magnitude.
    rho_est = (W_dense @ v).norm().item()
    if rho_est < 1e-12:
        rho_est = 1.0
    scale = rho_target / rho_est
    W_dense.mul_(scale)
    # Convert to sparse CSR for fast matvec.
    W_sparse = W_dense.to_sparse_csr()
    del W_dense
    return W_sparse


def _build_W_in(
    n: int, vocab: int, input_scale: float,
    generator: torch.Generator, device: torch.device,
) -> torch.Tensor:
    return (
        torch.rand(n, vocab, generator=generator, device=device).float() * 2.0 - 1.0
    ) * input_scale


# ---------------------------------------------------------------------------
# CharModel wrapper
# ---------------------------------------------------------------------------

class ESNCharModel(CharModel):
    def __init__(
        self,
        W_res_sparse: torch.Tensor,  # (N, N) sparse CSR
        W_in: torch.Tensor,           # (N, 256) dense
        W_out: torch.Tensor,          # (256, N) dense
        leak: float,
        device: torch.device,
    ):
        self.W_res = W_res_sparse
        self.W_in = W_in
        self.W_out = W_out
        self.leak = leak
        self.device = device
        self.N = W_in.shape[0]
        self.x = torch.zeros(self.N, dtype=torch.float32, device=device)
        # Precompute string keys for predict()'s output dict (only valid
        # single-byte UTF-8 chars are bytes 0..127; multi-byte chars need
        # >=2 bytes so single-byte 128..255 are not valid UTF-8 alone).
        self._byte_keys: list[str | None] = []
        for b in range(256):
            try:
                self._byte_keys.append(bytes([b]).decode("utf-8"))
            except UnicodeDecodeError:
                self._byte_keys.append(None)

    @torch.no_grad()
    def reset(self) -> None:
        self.x.zero_()

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        logits = self.W_out @ self.x  # (256,)
        probs = F.softmax(logits.float(), dim=-1).tolist()
        out: dict[str, float] = {}
        for b, key in enumerate(self._byte_keys):
            if key is not None:
                out[key] = probs[b]
        return out

    @torch.no_grad()
    def _step(self, byte: int) -> None:
        # x[t+1] = (1 - a) * x[t] + a * tanh(W_res @ x[t] + W_in[:, byte])
        pre = torch.sparse.mm(self.W_res, self.x.unsqueeze(1)).squeeze(1)
        pre.add_(self.W_in[:, byte])
        torch.tanh_(pre)
        self.x.mul_(1.0 - self.leak).add_(pre, alpha=self.leak)
        # Safety clamp against rare blow-ups (spec failure mode 7.2).
        self.x.clamp_(-5.0, 5.0)

    @torch.no_grad()
    def observe(self, char: str) -> None:
        for b in char.encode("utf-8"):
            self._step(int(b))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def _streaming_states(
    bytes_t: torch.Tensor,  # (T,) uint8 on device
    W_res: torch.Tensor,    # sparse CSR
    W_in: torch.Tensor,     # (N, 256)
    leak: float,
    washout: int,
    chunk_rows: int,
    device: torch.device,
):
    """Stream the reservoir over `bytes_t`, accumulating XtX and XtY.

    State buffer for each row r corresponds to predicting label bytes_t[r+2]
    from state x produced after consuming bytes_t[r+1]. The first `washout`
    state updates are not recorded.

    Returns (XtX, XtY) both float64 on device.
    """
    N = W_in.shape[0]
    T = bytes_t.numel()

    # State on GPU.
    x = torch.zeros(N, dtype=torch.float32, device=device)

    # Normal-equation accumulators in float64.
    XtX = torch.zeros(N, N, dtype=torch.float64, device=device)
    XtY = torch.zeros(N, VOCAB, dtype=torch.float64, device=device)

    # Row buffer for the current chunk.
    X_buf = torch.empty(chunk_rows, N, dtype=torch.float32, device=device)
    y_buf = torch.empty(chunk_rows, dtype=torch.long, device=device)
    buf_filled = 0

    t0 = time.monotonic()
    last_log = t0

    # Drive states for t = 0 .. T-2 (we always need a "next byte" label).
    # After update for input bytes_t[t], state predicts bytes_t[t+1].
    # We record (state_after_byte_t, label=bytes_t[t+1]) for t >= washout-1
    # so the first recorded sample corresponds to state after WASHOUT inputs.
    n_recorded = 0
    n_steps = T - 1  # we have labels for indices 1..T-1

    bytes_list = bytes_t  # keep as tensor; index as int conversion below
    # Pull host-side ints in chunks to minimize Python/GPU sync overhead.
    HOST_CHUNK = 4096
    t = 0
    while t < n_steps:
        end = min(t + HOST_CHUNK, n_steps)
        # Move this slice's input bytes (t..end-1) and labels (t+1..end) to host once.
        in_chunk = bytes_t[t:end].tolist()
        # We will record after each step.
        for j, b in enumerate(in_chunk):
            # Step: consume input byte `b`, advance state.
            pre = torch.sparse.mm(W_res, x.unsqueeze(1)).squeeze(1)
            pre.add_(W_in[:, b])
            torch.tanh_(pre)
            x.mul_(1.0 - leak).add_(pre, alpha=leak)
            x.clamp_(-5.0, 5.0)

            abs_t = t + j  # input index just consumed
            if abs_t >= washout - 1:
                # State x now predicts bytes_t[abs_t + 1].
                # Avoid per-step .item() sync; copy the row and label via tensor index.
                X_buf[buf_filled].copy_(x)
                y_buf[buf_filled] = bytes_t[abs_t + 1].long()
                buf_filled += 1
                n_recorded += 1
                if buf_filled == chunk_rows:
                    # Flush: accumulate XtX += X^T X and XtY += X^T onehot(y)
                    Xc = X_buf  # (chunk_rows, N)
                    Yc = F.one_hot(y_buf, num_classes=VOCAB).float()  # (chunk_rows, 256)
                    XtX.add_((Xc.t().double()) @ (Xc.double()))
                    XtY.add_((Xc.t().double()) @ (Yc.double()))
                    buf_filled = 0

                    now = time.monotonic()
                    if now - last_log > 10.0:
                        elapsed = now - t0
                        rate = n_recorded / max(1e-9, elapsed)
                        print(
                            f"[esn] streamed {n_recorded:,} states "
                            f"({rate:,.0f} st/s, elapsed {elapsed:.1f}s)",
                            flush=True,
                        )
                        last_log = now
        t = end

    # Flush remainder.
    if buf_filled > 0:
        Xc = X_buf[:buf_filled]
        Yc = F.one_hot(y_buf[:buf_filled], num_classes=VOCAB).float()
        XtX.add_((Xc.t().double()) @ (Xc.double()))
        XtY.add_((Xc.t().double()) @ (Yc.double()))

    del X_buf, y_buf
    torch.cuda.empty_cache()
    return XtX, XtY, n_recorded


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed = int(os.environ.get("SEED", "0"))
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("ESN submission requires CUDA.")
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    print(f"[esn] seed={seed} N={N_RESERVOIR} density={DENSITY} "
          f"rho={SPECTRAL_RADIUS} leak={LEAK_A} input_scale={INPUT_SCALE} "
          f"N_train={N_TRAIN} washout={WASHOUT} lambda={RIDGE_LAMBDA}",
          flush=True)

    t_build0 = time.monotonic()
    W_res = _build_sparse_W_res(N_RESERVOIR, DENSITY, SPECTRAL_RADIUS, gen, device)
    W_in = _build_W_in(N_RESERVOIR, VOCAB, INPUT_SCALE, gen, device)
    print(f"[esn] reservoir built in {time.monotonic() - t_build0:.2f}s",
          flush=True)

    # Encode training text as uint8 bytes and trim to N_TRAIN.
    raw = train_text.encode("utf-8")
    n_avail = len(raw)
    n_use = min(N_TRAIN, n_avail)
    bytes_cpu = torch.frombuffer(bytearray(raw[:n_use]), dtype=torch.uint8)
    bytes_t = bytes_cpu.to(device)
    print(f"[esn] training on {n_use:,} bytes (avail {n_avail:,})", flush=True)

    # Streaming pass + normal-equation accumulation.
    t_stream0 = time.monotonic()
    XtX, XtY, n_recorded = _streaming_states(
        bytes_t, W_res, W_in, LEAK_A, WASHOUT, CHUNK_ROWS, device,
    )
    print(f"[esn] streamed {n_recorded:,} states in "
          f"{time.monotonic() - t_stream0:.1f}s", flush=True)

    # Free input bytes; we no longer need them.
    del bytes_t, bytes_cpu

    # Ridge regularizer scaled by trace(XtX)/N for numerical sanity.
    trace_per_n = (torch.diagonal(XtX).sum() / N_RESERVOIR).item()
    ridge = RIDGE_LAMBDA * trace_per_n
    print(f"[esn] trace(XtX)/N = {trace_per_n:.4g}  ridge = {ridge:.4g}",
          flush=True)
    XtX.diagonal().add_(ridge)

    t_solve0 = time.monotonic()
    L = torch.linalg.cholesky(XtX)
    # Solve XtX @ W_out_T = XtY  =>  W_out_T shape (N, 256)
    W_out_T = torch.cholesky_solve(XtY, L)
    W_out = W_out_T.t().contiguous().float()  # (256, N) float32
    print(f"[esn] ridge solve in {time.monotonic() - t_solve0:.2f}s",
          flush=True)

    # Free large training tensors.
    del XtX, XtY, L, W_out_T
    torch.cuda.empty_cache()

    return ESNCharModel(W_res, W_in, W_out, LEAK_A, device)
