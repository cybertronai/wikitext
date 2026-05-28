"""Polynomial TensorSketch (Pham-Pagh 2013) + closed-form ridge LM.

Per research/non_nn_methods/spec_03_polynomial_tensorsketch_ridge.md.

Mechanism:
  * Context window K=16 bytes encoded as a positional one-hot of size
    d = K*256 = 4096 (sparse: exactly K active indices per position).
  * Polynomial kernel of degree p approximated by TensorSketch:
        phi(x) = IFFT( prod_{i=1..p} FFT( CountSketch_i(x) ) )
    Each CountSketch_i maps R^d -> R^m via independent random hashes
    h_i: [d] -> [m] and signs s_i: [d] -> {+1, -1}.
  * Linear ridge head W (m x 256) solved in closed form via Cholesky on
    G = Phi.T @ Phi + lambda I (m x m), and Phi.T @ Y (m x 256), where
    Y is the one-hot of the next byte. No SGD; one Cholesky.

This is paradigm-A in the "kernel-machine-replaces-model" sense: a fixed
random feature map + closed-form readout.
"""
from __future__ import annotations

__author__ = "@ab-10"

import os
import time

import torch
from torch import Tensor

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

K = 16            # context window in bytes
M = 8192          # TensorSketch dimension
P = 3             # polynomial degree
LAMBDA = 1.0      # ridge jitter
BATCH = 4096      # chunk of contexts per accumulator pass
N_TRAIN_POS = 5_000_000   # cap on training positions used (spec: 5e6)

D_INPUT = K * 256


# ---------------------------------------------------------------------------
# TensorSketch construction
# ---------------------------------------------------------------------------

def _draw_hashes(p: int, d: int, m: int, device, seed: int) -> tuple[Tensor, Tensor]:
    """Return:
      H : (p, d) int64 with values in [0, m).
      S : (p, d) float32 with values in {-1, +1}.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    H = torch.randint(0, m, (p, d), generator=g, device=device, dtype=torch.int64)
    sign_bits = torch.randint(0, 2, (p, d), generator=g, device=device, dtype=torch.int32)
    S = (sign_bits.to(torch.float32) * 2.0 - 1.0)
    return H, S


def _count_sketches_for_windows(
    windows: Tensor,    # (B, K) uint8, byte at each window position
    H: Tensor,          # (P, K*256) int64
    S: Tensor,          # (P, K*256) float32
    m: int,
) -> Tensor:
    """Compute the p CountSketches for B contexts of length K.

    For position k in [0,K) and byte b at that position, the active
    feature index in the d=K*256-dim one-hot is k*256 + b. We gather
    h_i[k*256 + b] and s_i[k*256 + b] and scatter-add into a (B, m) buffer.

    Returns CS : (P, B, m) float32.
    """
    B, k_ = windows.shape
    assert k_ == K
    device = windows.device

    pos_offset = torch.arange(K, device=device, dtype=torch.int64) * 256  # (K,)
    feature_idx = pos_offset.unsqueeze(0) + windows.to(torch.int64)        # (B, K)

    # Per hash i: gather h_i[feature_idx] -> (B, K), s_i[feature_idx] -> (B, K).
    # We do all P hashes at once: H[:, feature_idx] -> (P, B, K).
    gather_H = H[:, feature_idx]   # (P, B, K)
    gather_S = S[:, feature_idx]   # (P, B, K)

    cs = torch.zeros((P, B, m), dtype=torch.float32, device=device)
    cs.scatter_add_(2, gather_H, gather_S)
    return cs


def _tensor_sketch(
    cs: Tensor,   # (P, B, m) float32
) -> Tensor:
    """Apply FFT-based circular convolution chain to produce phi.

    phi[t, :] = IFFT( prod_{i=1..p} FFT( CS_i(x_t) ) )
    """
    spec = torch.fft.rfft(cs, dim=2)        # (P, B, m//2+1) complex
    prod = spec.prod(dim=0)                 # (B, m//2+1) complex
    phi = torch.fft.irfft(prod, n=cs.shape[2], dim=1)   # (B, m) float32
    return phi


# ---------------------------------------------------------------------------
# Eval-time per-char model wrapper
# ---------------------------------------------------------------------------

class PolyTSModel(CharModel):
    """Streaming CharModel: maintain ring buffer of last K bytes, recompute
    phi on each predict via fresh TensorSketch (one window at a time).

    Predict cost is dominated by 3 small FFTs of length M; ~100us per byte
    on A100, well inside the 50-min eval budget.
    """

    def __init__(
        self,
        H: Tensor,        # (P, d_input) int64 on CUDA
        S: Tensor,        # (P, d_input) float32 on CUDA
        W: Tensor,        # (m, 256) float32 on CUDA
        K: int,
        M: int,
        P: int,
    ):
        self.H = H
        self.S = S
        self.W = W
        self.K = K
        self.M = M
        self.P = P
        self.device = W.device
        self.history = bytearray()
        # Persistent single-window buffers to avoid allocs.
        self._win = torch.zeros((1, K), dtype=torch.uint8, device=self.device)

    def reset(self) -> None:
        self.history.clear()
        self._win.zero_()

    def predict(self) -> dict[str, float]:
        # Build the K-byte left-padded window.
        if len(self.history) >= self.K:
            tail = self.history[-self.K:]
            self._win[0].copy_(torch.frombuffer(bytes(tail), dtype=torch.uint8).to(self.device))
        else:
            # Left-pad with zero bytes.
            self._win.zero_()
            n = len(self.history)
            if n > 0:
                tail = self.history[-n:]
                self._win[0, -n:].copy_(torch.frombuffer(bytes(tail), dtype=torch.uint8).to(self.device))

        cs = _count_sketches_for_windows(self._win, self.H, self.S, self.M)  # (P,1,m)
        phi = _tensor_sketch(cs)  # (1, m)
        logits = phi @ self.W   # (1, 256)
        idx = int(logits.argmax(dim=1).item())
        return {chr(idx): 1.0}

    def observe(self, char: str) -> None:
        self.history.extend(char.encode("utf-8"))
        if len(self.history) > self.K + 8:
            del self.history[: len(self.history) - self.K]


# ---------------------------------------------------------------------------
# Train entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 0
    print(f"[poly_ts] seed={seed} K={K} M={M} P={P} lambda={LAMBDA} N={N_TRAIN_POS:,}",
          flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for poly_tensorsketch_ridge")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    # Enable TF32 for the fp32 matmuls in the Gram accumulator (~1.5x on A100).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t_total = time.monotonic()

    # ---- Encode train text as uint8 on GPU. ----
    raw = train_text.encode("utf-8")
    n_bytes_total = len(raw)
    # Use up to N_TRAIN_POS *positions*; positions are i = K..N-1.
    # Take a prefix of length min(n_bytes_total, K + N_TRAIN_POS).
    max_take = min(n_bytes_total, K + N_TRAIN_POS)
    train_bytes_u8 = torch.frombuffer(bytearray(raw[:max_take]), dtype=torch.uint8).to(device)
    n_bytes = train_bytes_u8.numel()
    print(f"[poly_ts] encoded train: {n_bytes:,} bytes  "
          f"{time.monotonic()-t_total:.1f}s", flush=True)

    # ---- Draw hashes ----
    H, S = _draw_hashes(P, D_INPUT, M, device, seed=seed * 7919 + 17)
    print(f"[poly_ts] drew hashes H={tuple(H.shape)} S={tuple(S.shape)}", flush=True)

    # ---- Stream contexts, accumulate Phi^T Phi and Phi^T Y. ----
    G = torch.zeros((M, M), dtype=torch.float32, device=device)        # (m, m)
    PhiTY = torch.zeros((M, 256), dtype=torch.float32, device=device)  # (m, 256)

    n_positions = n_bytes - K  # positions t = K, K+1, ..., n_bytes - 1
    if n_positions <= 0:
        raise RuntimeError(f"train too short: {n_bytes} bytes < K+1={K+1}")
    print(f"[poly_ts] streaming {n_positions:,} positions in batches of {BATCH:,}",
          flush=True)

    arange_K = torch.arange(K, device=device, dtype=torch.int64)  # (K,)

    t_acc = time.monotonic()
    n_done = 0
    for start in range(0, n_positions, BATCH):
        end = min(start + BATCH, n_positions)
        B = end - start
        # For position t in [K+start, K+end), the window is bytes[t-K..t-1]
        # and the target is bytes[t]. Equivalently, the b-th window in
        # this batch is bytes[start+b .. start+b+K-1] and target is
        # bytes[start+b+K].
        row_start = torch.arange(start, end, device=device, dtype=torch.int64)  # (B,)
        idx_mat = row_start.unsqueeze(1) + arange_K.unsqueeze(0)  # (B, K)
        windows = train_bytes_u8[idx_mat]  # (B, K) uint8
        targets = train_bytes_u8[K + row_start].to(torch.int64)  # (B,)

        cs = _count_sketches_for_windows(windows, H, S, M)  # (P, B, m)
        phi = _tensor_sketch(cs)  # (B, m) float32

        # Accumulate Phi^T Phi += phi.T @ phi  (m x m)
        G.addmm_(phi.t(), phi)
        # Accumulate Phi^T Y += phi.T @ Y (where Y is one-hot of target).
        # Equivalent: PhiTY[:, c] += sum_{t: target_t == c} phi[t]
        PhiTY.index_add_(1, targets, phi.t())   # phi.t() : (m, B)

        n_done += B
        if n_done % (BATCH * 16) == 0 or end == n_positions:
            torch.cuda.synchronize()
            dt = time.monotonic() - t_acc
            rate = n_done / max(1e-9, dt)
            print(f"[poly_ts]   pos {n_done:>10,}/{n_positions:,}  "
                  f"({100*n_done/n_positions:5.1f}%)  "
                  f"{rate:8.0f} pos/s  {dt:6.1f}s", flush=True)

    torch.cuda.synchronize()
    print(f"[poly_ts] accumulation done: {time.monotonic()-t_acc:.1f}s", flush=True)

    # ---- Cholesky solve: W = (G + lambda I)^{-1} Phi^T Y ----
    t_chol = time.monotonic()
    # Add jitter on the diagonal in-place.
    diag_idx = torch.arange(M, device=device)
    G[diag_idx, diag_idx] += LAMBDA
    print(f"[poly_ts] solving (m={M}) ...", flush=True)
    try:
        L = torch.linalg.cholesky(G)
        W = torch.cholesky_solve(PhiTY, L)
    except Exception as e:
        print(f"[poly_ts] Cholesky failed ({e}); falling back to torch.linalg.solve",
              flush=True)
        W = torch.linalg.solve(G, PhiTY)
    torch.cuda.synchronize()
    print(f"[poly_ts] solve done: {time.monotonic()-t_chol:.1f}s  W={tuple(W.shape)}",
          flush=True)

    print(f"[poly_ts] total train wall: {time.monotonic()-t_total:.1f}s", flush=True)

    # Move G off the GPU to free HBM. Keep W, H, S on device.
    del G, PhiTY
    torch.cuda.empty_cache()

    return PolyTSModel(H=H, S=S, W=W, K=K, M=M, P=P)
