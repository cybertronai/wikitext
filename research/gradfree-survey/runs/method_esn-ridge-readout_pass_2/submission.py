"""ESN with Ridge-Regression Readout — Pass 2.

Batched-streaming Echo State Network: B=64 parallel reservoir streams
sharing a single sparse W_res and dense W_in, K=4-byte short-history
input, ridge readout over phi = concat(x_state, u_input). Closed-form
Cholesky solve on float64 normal equations.

Spec: .survey/designs/method_esn-ridge-readout_pass_2.md
"""
from __future__ import annotations

__author__ = "@survey-esn-p2"

import os
import time

import torch
import torch.nn.functional as F

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Hyperparameters (from spec — DO NOT vary)
# ---------------------------------------------------------------------------
N_RESERVOIR = 16384
DENSITY = 0.02
LEAK_A = 0.3
SPECTRAL_RADIUS = 0.9
INPUT_SCALE = 0.4
RIDGE_LAMBDA = 1e-3  # relative, scaled by trace(XtX)/N_phi
K_HISTORY = 4
B_STREAMS = 64
N_TRAIN = 16_000_000
WASHOUT = 1000
ACCUM_BLOCK = 2000  # inner steps between Phi.T @ Phi flushes
VOCAB = 256
D_IN = K_HISTORY * VOCAB  # 1024
N_PHI = N_RESERVOIR + D_IN  # 17408


# ---------------------------------------------------------------------------
# Reservoir construction
# ---------------------------------------------------------------------------

def _build_sparse_W_res(
    n: int, density: float, rho_target: float,
    generator: torch.Generator, device: torch.device,
) -> torch.Tensor:
    """Build sparse CSR W_res scaled to target spectral radius.

    For N=16384 a dense N*N float32 matrix is 1.07 GB — fits on A100-80GB
    during the build/power-iteration phase, then we drop it and keep only
    the sparse CSR representation (~65 MB for 5.4M nz).
    """
    nnz = int(n * n * density)
    # Sample nnz unique flat indices.
    flat_idx = torch.randperm(n * n, generator=generator, device=device)[:nnz]
    rows = flat_idx // n
    cols = flat_idx % n
    vals = (torch.rand(nnz, generator=generator, device=device) * 2.0 - 1.0).float()

    # Dense scratch for spectral radius estimation via power iteration.
    W_dense = torch.zeros(n, n, dtype=torch.float32, device=device)
    W_dense[rows, cols] = vals

    v = torch.randn(n, dtype=torch.float32, device=device, generator=generator)
    v = v / (v.norm() + 1e-12)
    for _ in range(30):
        v = W_dense @ v
        nv = v.norm()
        if nv < 1e-12:
            break
        v = v / nv
    rho_est = (W_dense @ v).norm().item()
    if rho_est < 1e-12:
        rho_est = 1.0
    scale = rho_target / rho_est
    W_dense.mul_(scale)
    W_sparse = W_dense.to_sparse_csr()
    del W_dense, flat_idx, rows, cols, vals
    torch.cuda.empty_cache()
    return W_sparse


def _build_W_in(
    n: int, d_in: int, input_scale: float,
    generator: torch.Generator, device: torch.device,
) -> torch.Tensor:
    """W_in : (N, D_in) uniform on [-input_scale, +input_scale]."""
    w = (
        torch.rand(n, d_in, generator=generator, device=device).float() * 2.0 - 1.0
    ) * input_scale
    return w


# ---------------------------------------------------------------------------
# CharModel wrapper (single-stream eval)
# ---------------------------------------------------------------------------

class ESNCharModel(CharModel):
    def __init__(
        self,
        W_res_sparse: torch.Tensor,  # (N, N) sparse CSR
        W_in: torch.Tensor,           # (N, D_in)
        W_out: torch.Tensor,          # (256, N_phi)
        leak: float,
        device: torch.device,
        k_history: int,
    ):
        self.W_res = W_res_sparse
        self.W_in = W_in
        self.W_out = W_out
        self.leak = leak
        self.device = device
        self.k = k_history
        self.N = W_in.shape[0]
        # Reservoir state, single stream.
        self.x = torch.zeros(self.N, dtype=torch.float32, device=device)
        # K-byte history one-hot ring buffer; we keep as a (K, 256) tensor,
        # newest at slot[k-1]; built each step into a flat D_in vector.
        self.u_hist = torch.zeros(self.k, VOCAB, dtype=torch.float32, device=device)
        # Pre-allocate the u_flat scratch.
        self.u_flat = torch.zeros(self.k * VOCAB, dtype=torch.float32, device=device)
        # Pre-allocate phi scratch.
        self.phi = torch.zeros(self.N + self.k * VOCAB, dtype=torch.float32, device=device)
        # Valid single-byte UTF-8 keys (00..7F) for predict() output.
        self._byte_keys: list[str | None] = []
        for b in range(256):
            try:
                self._byte_keys.append(bytes([b]).decode("utf-8"))
            except UnicodeDecodeError:
                self._byte_keys.append(None)

    @torch.no_grad()
    def reset(self) -> None:
        self.x.zero_()
        self.u_hist.zero_()

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        # phi = concat(x, flatten(u_hist))
        self.phi[: self.N].copy_(self.x)
        self.phi[self.N:].copy_(self.u_hist.reshape(-1))
        logits = self.W_out @ self.phi  # (256,)
        probs = F.softmax(logits.float(), dim=-1).tolist()
        out: dict[str, float] = {}
        for b, key in enumerate(self._byte_keys):
            if key is not None:
                out[key] = probs[b]
        return out

    @torch.no_grad()
    def _step(self, byte: int) -> None:
        # Roll u_hist: shift left, place new one-hot at slot k-1.
        # u_hist[0..k-2] := u_hist[1..k-1]; u_hist[k-1] := onehot(byte).
        if self.k > 1:
            self.u_hist[: self.k - 1].copy_(self.u_hist[1:].clone())
        self.u_hist[self.k - 1].zero_()
        self.u_hist[self.k - 1, byte] = 1.0
        # Flatten u for input projection.
        u_flat = self.u_hist.reshape(-1)  # (D_in,)
        # pre = W_res @ x + W_in @ u_flat
        pre = torch.sparse.mm(self.W_res, self.x.unsqueeze(1)).squeeze(1)
        pre.add_(self.W_in @ u_flat)
        torch.tanh_(pre)
        self.x.mul_(1.0 - self.leak).add_(pre, alpha=self.leak)
        self.x.clamp_(-5.0, 5.0)

    @torch.no_grad()
    def observe(self, char: str) -> None:
        for b in char.encode("utf-8"):
            self._step(int(b))


# ---------------------------------------------------------------------------
# Training: batched streaming + chunked normal-equation accumulation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _stream_and_accumulate(
    bytes_t: torch.Tensor,  # (T,) uint8 on device, T >= B * chunk_len
    W_res: torch.Tensor,    # sparse CSR (N, N)
    W_in: torch.Tensor,     # (N, D_in)
    leak: float,
    B: int,
    K: int,
    washout: int,
    accum_block: int,
    device: torch.device,
):
    """Run B parallel reservoir streams over disjoint contiguous slices of
    bytes_t and accumulate Phi.T @ Phi and Phi.T @ Onehot(y) into float64
    XtX, XtY where phi = concat(x_state, u_history_flat).

    Each stream b reads chunk_len = T // B bytes; row index t within a
    chunk produces feature phi_b,t computed from x_b after consuming
    bytes_b,t, paired with label bytes_b,t+1. Records start after the
    `washout` first per-stream steps.
    """
    N = W_in.shape[0]
    D_in = W_in.shape[1]
    N_phi_local = N + D_in
    T = bytes_t.numel()
    chunk_len = T // B
    # We need a label at t+1 for the last recorded step; cap inner loop at
    # chunk_len - 1 inputs consumed (so the label of the last is in-range).
    n_steps = chunk_len - 1
    if n_steps <= washout:
        raise RuntimeError(
            f"chunk_len={chunk_len} too small for washout={washout}"
        )

    # Reshape input as (B, chunk_len) contiguous slices.
    inputs = bytes_t[: B * chunk_len].view(B, chunk_len)  # uint8

    # State (B, N) and rolling history (B, K, 256) one-hot.
    X = torch.zeros(B, N, dtype=torch.float32, device=device)
    Uhist = torch.zeros(B, K, VOCAB, dtype=torch.float32, device=device)

    # Pre-allocate inner block buffers for phi and labels.
    # phi block shape (accum_block, B, N_phi) is huge (~9 GB for 2000*64*17408*4);
    # instead allocate (B * accum_block, N_phi) flat, filled per step.
    Phi_block = torch.empty(
        accum_block * B, N_phi_local, dtype=torch.float32, device=device,
    )
    y_block = torch.empty(accum_block * B, dtype=torch.long, device=device)
    block_filled = 0  # number of (B-wide) inner steps written into block

    # Accumulators.
    XtX = torch.zeros(N_phi_local, N_phi_local, dtype=torch.float64, device=device)
    XtY = torch.zeros(N_phi_local, VOCAB, dtype=torch.float64, device=device)

    n_recorded = 0
    t0 = time.monotonic()
    last_log = t0

    # Helper: flush current accumulated block into XtX/XtY.
    def _flush():
        nonlocal block_filled
        if block_filled == 0:
            return
        n_rows = block_filled * B
        Pc = Phi_block[:n_rows]  # (n_rows, N_phi)
        Yc = F.one_hot(y_block[:n_rows], num_classes=VOCAB).float()  # (n_rows, 256)
        # Float64 GEMM for normal equations stability.
        Pd = Pc.double()
        XtX.add_(Pd.t() @ Pd)
        XtY.add_(Pd.t() @ Yc.double())
        block_filled = 0

    # Pre-build per-stream batch-index helper for scatter.
    batch_idx = torch.arange(B, device=device)

    for t in range(n_steps):
        # Roll Uhist left along K dim and scatter new one-hot at slot K-1.
        if K > 1:
            Uhist[:, : K - 1, :] = Uhist[:, 1:, :].clone()
        Uhist[:, K - 1, :].zero_()
        in_bytes = inputs[:, t].long()  # (B,)
        Uhist[batch_idx, K - 1, in_bytes] = 1.0

        u_flat = Uhist.view(B, K * VOCAB)  # (B, D_in)

        # pre = W_res @ X.T  -> (N, B)  via sparse-dense matmul, then
        # add W_in @ u_flat.T -> (N, B); transpose back later.
        # torch.sparse.mm(sparse(N,N), dense(N,B)) -> (N, B).
        pre_NB = torch.sparse.mm(W_res, X.t())  # (N, B)
        pre_NB.add_(W_in @ u_flat.t())  # (N, B)
        torch.tanh_(pre_NB)
        # Update state: X = (1-a) X + a pre.T
        X.mul_(1.0 - leak).add_(pre_NB.t(), alpha=leak)
        X.clamp_(-5.0, 5.0)

        if t >= washout - 1:
            # Record: phi = concat(X, u_flat) along feature dim, label is
            # next byte in each chunk.
            base = block_filled * B
            Phi_block[base : base + B, :N].copy_(X)
            Phi_block[base : base + B, N:].copy_(u_flat)
            y_block[base : base + B] = inputs[:, t + 1].long()
            block_filled += 1
            n_recorded += B
            if block_filled == accum_block:
                _flush()

                now = time.monotonic()
                if now - last_log > 10.0:
                    elapsed = now - t0
                    rate = n_recorded / max(1e-9, elapsed)
                    inner_steps_done = t + 1
                    print(
                        f"[esn-p2] streamed {n_recorded:,} rows "
                        f"(inner step {inner_steps_done:,}/{n_steps:,}, "
                        f"{rate:,.0f} rows/s, elapsed {elapsed:.1f}s)",
                        flush=True,
                    )
                    last_log = now

    _flush()

    del Phi_block, y_block, X, Uhist
    torch.cuda.empty_cache()
    return XtX, XtY, n_recorded


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed = int(os.environ.get("SEED", "0"))
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("ESN pass-2 submission requires CUDA.")
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    print(
        f"[esn-p2] seed={seed} N={N_RESERVOIR} density={DENSITY} "
        f"rho={SPECTRAL_RADIUS} leak={LEAK_A} input_scale={INPUT_SCALE} "
        f"K={K_HISTORY} B={B_STREAMS} N_train={N_TRAIN} washout={WASHOUT} "
        f"accum_block={ACCUM_BLOCK} lambda={RIDGE_LAMBDA} N_phi={N_PHI}",
        flush=True,
    )

    t_build0 = time.monotonic()
    W_res = _build_sparse_W_res(N_RESERVOIR, DENSITY, SPECTRAL_RADIUS, gen, device)
    W_in = _build_W_in(N_RESERVOIR, D_IN, INPUT_SCALE, gen, device)
    print(
        f"[esn-p2] reservoir built in {time.monotonic() - t_build0:.2f}s",
        flush=True,
    )

    # Encode training text as uint8 bytes; trim to N_TRAIN.
    raw = train_text.encode("utf-8")
    n_avail = len(raw)
    n_use = min(N_TRAIN, n_avail)
    bytes_cpu = torch.frombuffer(bytearray(raw[:n_use]), dtype=torch.uint8)
    bytes_t = bytes_cpu.to(device)
    print(
        f"[esn-p2] training on {n_use:,} bytes (avail {n_avail:,}); "
        f"per-stream chunk_len = {n_use // B_STREAMS:,}",
        flush=True,
    )

    # Stream + accumulate.
    t_stream0 = time.monotonic()
    XtX, XtY, n_recorded = _stream_and_accumulate(
        bytes_t, W_res, W_in, LEAK_A,
        B_STREAMS, K_HISTORY, WASHOUT, ACCUM_BLOCK, device,
    )
    print(
        f"[esn-p2] streamed {n_recorded:,} rows in "
        f"{time.monotonic() - t_stream0:.1f}s",
        flush=True,
    )

    del bytes_t, bytes_cpu

    # Ridge regularizer scaled by trace(XtX)/N_phi.
    trace_per_n = (torch.diagonal(XtX).sum() / N_PHI).item()
    ridge = RIDGE_LAMBDA * trace_per_n
    print(
        f"[esn-p2] trace(XtX)/N_phi = {trace_per_n:.4g}  ridge = {ridge:.4g}",
        flush=True,
    )
    XtX.diagonal().add_(ridge)

    t_solve0 = time.monotonic()
    try:
        L = torch.linalg.cholesky(XtX)
        # XtX @ W_out_T = XtY  ->  W_out_T : (N_phi, 256)
        W_out_T = torch.cholesky_solve(XtY, L)
        del L
    except Exception as e:
        # Fall back to lstsq if Cholesky fails (anticipated failure mode).
        print(f"[esn-p2] cholesky failed ({e}); falling back to lstsq",
              flush=True)
        sol = torch.linalg.lstsq(XtX, XtY)
        W_out_T = sol.solution
    W_out = W_out_T.t().contiguous().float()  # (256, N_phi)
    print(
        f"[esn-p2] ridge solve in {time.monotonic() - t_solve0:.2f}s",
        flush=True,
    )

    del XtX, XtY, W_out_T
    torch.cuda.empty_cache()

    return ESNCharModel(W_res, W_in, W_out, LEAK_A, device, K_HISTORY)
