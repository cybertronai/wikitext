"""Random Fourier Features + closed-form ridge readout for byte-level char LM.

Per research/non_nn_methods/spec_02_rff_closed_form_ridge.md.

Mechanism (gradient-free, no SGD on phi or W):
  1. Encode byte context window x_t = (c_{t-K+1}, ..., c_t) via a frozen
     random projection R : R^{256*K} -> R^d.  We materialize this as a
     learned-but-frozen byte embedding table E (256, d_byte) and sum across
     positions with a frozen positional code so the input lives in R^d.
  2. Apply Rahimi-Recht random Fourier features:
        phi(x) = sqrt(2/m) * cos(omega^T x + b)
        omega ~ N(0, gamma^2 I_d),   b ~ U[0, 2pi]
  3. Stream chunks through phi; accumulate two normal-equation buffers on
     GPU: G = Phi^T Phi (m x m) and C = Phi^T Y (m x 256).
  4. Closed-form: W = (G + lambda I)^{-1} C  (single Cholesky).
  5. Predict next byte via argmax_c (phi(x_t) @ W)_c.

Hits Tensor Cores: streaming Phi @ Phi^T in batched chunks is one
massive bf16 matmul per chunk plus an m x m fp32 accumulator. Phi
materialization itself is also bf16 matmul-bound. Only sequential pass
is the single Cholesky on (m, m) at the end — negligible.

Hyperparameters (chosen from spec § 9):
  K        = 16        # context window length, bytes
  d_byte   = 64        # per-byte embedding (frozen random projection)
  d_in     = K * d_byte = 1024  # RFF input dim
  m        = 8192      # RFF feature count
  gamma    = 0.3       # RBF bandwidth (1 / sqrt(2 sigma^2))
  lam      = 1e-2      # ridge jitter
"""
from __future__ import annotations

__author__ = "@ab-10"

import math
import os
import time

import torch
from torch import Tensor

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

K = 16              # context window length
D_BYTE = 64         # per-byte random embedding dim
D_IN = K * D_BYTE   # RFF input dim
M = 8192            # RFF feature count
GAMMA = 0.3         # RBF bandwidth
LAMBDA = 1e-2       # ridge regularization

# Streaming chunk size for Phi accumulation. Memory: chunk_size * m * 2B
# (bf16) ~= 64K * 8192 * 2B = 1 GiB. Comfortable on A100-80GB.
CHUNK_SIZE = 65_536

# Cap training positions used. Full WikiText-103 train = ~530 MB of utf8.
# We byte-stream so N is byte count. 5e6 is the spec recommendation; we
# scale up if budget allows.
MAX_TRAIN_POSITIONS = 8_000_000


# ---------------------------------------------------------------------------
# Streaming feature builder
# ---------------------------------------------------------------------------

def _build_byte_embedding(d_byte: int, generator: torch.Generator, device) -> Tensor:
    """Frozen random per-byte embedding (256, d_byte).

    Standard normal entries, scaled by 1/sqrt(d_byte) so the per-byte
    contribution to a length-K window has roughly unit norm.
    """
    E = torch.randn(256, d_byte, generator=generator, device="cpu", dtype=torch.float32)
    E = E / (d_byte ** 0.5)
    return E.to(device)


def _build_rff(d_in: int, m: int, gamma: float,
               generator: torch.Generator, device) -> tuple[Tensor, Tensor]:
    """Return (omega, b) for the RFF map phi(x) = sqrt(2/m) cos(omega^T x + b).

    omega ~ N(0, gamma^2 I_{d_in}) shape (d_in, m)
    b     ~ U[0, 2pi]              shape (m,)
    """
    omega = torch.randn(d_in, m, generator=generator, device="cpu", dtype=torch.float32) * gamma
    b = torch.rand(m, generator=generator, device="cpu", dtype=torch.float32) * (2 * math.pi)
    return omega.to(device), b.to(device)


def _windows_to_input(
    byte_ids: Tensor,    # (N,) int64, full byte stream on GPU
    starts: Tensor,      # (B,) int64, window start indices on GPU
    E: Tensor,           # (256, d_byte) float32
    K: int,
) -> Tensor:
    """Materialize a batch of B context vectors of dim K*d_byte.

    Each window is the concatenation of K byte embeddings:
        x_t = [E[c_{t-K+1}] || ... || E[c_t]]
    """
    B = starts.shape[0]
    d_byte = E.shape[1]
    # (B, K) byte ids
    idx = starts.unsqueeze(1) + torch.arange(K, device=starts.device).unsqueeze(0)
    ctx_ids = byte_ids[idx]                       # (B, K)
    emb = E[ctx_ids]                              # (B, K, d_byte)
    return emb.reshape(B, K * d_byte)             # (B, K*d_byte)


def _rff_map(
    x: Tensor,           # (B, d_in)  float32
    omega_bf: Tensor,    # (d_in, m)  bf16
    b: Tensor,           # (m,)       float32
    m: int,
) -> Tensor:
    """phi(x) = sqrt(2/m) cos(omega^T x + b), returns (B, m) in bf16."""
    # Run the big matmul in bf16 for Tensor Core throughput.
    z = (x.to(torch.bfloat16) @ omega_bf)                   # (B, m) bf16
    z = z.to(torch.float32) + b
    phi = torch.cos(z) * (math.sqrt(2.0 / m))
    return phi.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# CharModel
# ---------------------------------------------------------------------------

class RFFRidgeModel(CharModel):
    """Stream a byte ring-buffer of length K; predict via phi(x) @ W."""

    def __init__(
        self,
        W: Tensor,           # (m, 256) float32 on device
        E: Tensor,           # (256, d_byte) float32 on device
        omega: Tensor,       # (d_in, m) float32 on device
        b: Tensor,           # (m,) float32 on device
        K: int,
        m: int,
    ):
        self._W = W
        self._E = E
        self._omega_bf = omega.to(torch.bfloat16)
        self._b = b
        self._K = K
        self._m = m
        self._device = W.device
        # Ring buffer of last K byte ids as a Python bytearray. We rebuild
        # the input vector each predict() — phi recompute per char is the
        # documented behavior in spec § 5.
        self._history = bytearray()

    def reset(self) -> None:
        self._history.clear()

    def _phi_current(self) -> Tensor:
        """Compute phi(x_t) for the current K-byte context window.

        If history is shorter than K, left-pad with zero bytes (byte 0).
        """
        # Build context bytes of length K (pad with byte 0 if short).
        ctx = bytearray(self._K)
        if len(self._history) >= self._K:
            ctx = self._history[-self._K:]
        else:
            # Left-pad with 0s.
            tail = self._history
            ctx[self._K - len(tail):] = tail
        ctx_ids = torch.frombuffer(bytes(ctx), dtype=torch.uint8).to(
            self._device, dtype=torch.int64, non_blocking=True
        )  # (K,)
        x = self._E[ctx_ids].reshape(1, -1)  # (1, K*d_byte)
        z = (x.to(torch.bfloat16) @ self._omega_bf)  # (1, m)
        z = z.to(torch.float32) + self._b
        phi = torch.cos(z) * (math.sqrt(2.0 / self._m))  # (1, m) float32
        return phi

    def predict(self) -> dict[str, float]:
        with torch.inference_mode():
            phi = self._phi_current()                 # (1, m) float32
            logits = phi @ self._W                    # (1, 256) float32
            best = int(logits.argmax(dim=-1).item())
        return {chr(best): 1.0}

    def observe(self, char: str) -> None:
        self._history.extend(char.encode("utf-8"))
        if len(self._history) > self._K:
            del self._history[:-self._K]


# ---------------------------------------------------------------------------
# Training: build (Phi^T Phi), (Phi^T Y) in chunks; closed-form solve.
# ---------------------------------------------------------------------------

def _seed_generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 1337
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = _seed_generator(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[rff_ridge] WARNING: no CUDA device — will be slow.")

    t_total = time.monotonic()

    # ---- byte stream ------------------------------------------------------
    raw = train_text.encode("utf-8")
    n_bytes = len(raw)
    # Cap training positions used.
    n_use = min(n_bytes, MAX_TRAIN_POSITIONS + K)
    byte_ids = torch.frombuffer(bytearray(raw[:n_use]), dtype=torch.uint8).to(
        device, dtype=torch.int64
    )
    n_positions = byte_ids.numel() - K  # valid (window, next-byte) pairs
    print(f"[rff_ridge] N_bytes={n_bytes:,}  using {byte_ids.numel():,} "
          f"({n_positions:,} train positions)", flush=True)

    # ---- frozen random maps ----------------------------------------------
    E = _build_byte_embedding(D_BYTE, gen, device)
    omega, b = _build_rff(D_IN, M, GAMMA, gen, device)
    omega_bf = omega.to(torch.bfloat16)
    print(f"[rff_ridge] K={K} d_byte={D_BYTE} d_in={D_IN} m={M} "
          f"gamma={GAMMA} lambda={LAMBDA}", flush=True)

    # ---- accumulators -----------------------------------------------------
    # G accumulates Phi^T Phi (m, m) in fp32; C accumulates Phi^T Y (m, 256) in fp32.
    G = torch.zeros((M, M), dtype=torch.float32, device=device)
    C = torch.zeros((M, 256), dtype=torch.float32, device=device)

    # ---- stream training positions in chunks ------------------------------
    t_stream = time.monotonic()
    pos = 0
    n_done = 0
    while pos < n_positions:
        end = min(pos + CHUNK_SIZE, n_positions)
        starts = torch.arange(pos, end, device=device)
        # X: (B, d_in)
        x = _windows_to_input(byte_ids, starts, E, K)
        # phi: (B, m) bf16
        phi = _rff_map(x, omega_bf, b, M)
        # Y: one-hot of next byte → keep as int64 indices for scatter.
        next_ids = byte_ids[starts + K]            # (B,)

        # G += phi^T @ phi  (m, m) — main compute. bf16 inputs hit Tensor
        # Cores; cuBLAS accumulates in fp32 internally.  We materialise the
        # bf16 product and cast to fp32 for the running fp32 accumulator.
        gram = phi.t() @ phi  # (m, m) bf16, computed in TC, fp32 internal acc
        G.add_(gram.to(torch.float32))
        del gram

        # C += phi^T @ Y. Y is one-hot, so we can avoid building it: for each
        # row r, C[:, next_ids[r]] += phi[r, :].
        C.index_add_(1, next_ids, phi.t().to(torch.float32))

        n_done += end - pos
        pos = end

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_stream_done = time.monotonic()
    print(f"[rff_ridge] streamed {n_done:,} positions in "
          f"{t_stream_done - t_stream:.1f}s "
          f"({n_done / max(1e-9, t_stream_done - t_stream):,.0f} pos/s)", flush=True)

    # ---- closed-form ridge solve -----------------------------------------
    t_chol = time.monotonic()
    # Add jitter for numerical stability.
    G.add_(torch.eye(M, dtype=torch.float32, device=device), alpha=LAMBDA)
    try:
        L = torch.linalg.cholesky(G)
        W = torch.cholesky_solve(C, L)  # (m, 256)
    except Exception as e:
        print(f"[rff_ridge] Cholesky failed ({e}); retrying with larger jitter")
        # Fallback: bump lambda 100x.
        G.add_(torch.eye(M, dtype=torch.float32, device=device), alpha=LAMBDA * 99)
        L = torch.linalg.cholesky(G)
        W = torch.cholesky_solve(C, L)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[rff_ridge] Cholesky solve: {time.monotonic() - t_chol:.2f}s  "
          f"W.shape={tuple(W.shape)}", flush=True)

    # Free intermediates we don't need at inference.
    del G, C
    torch.cuda.empty_cache() if device.type == "cuda" else None

    print(f"[rff_ridge] total build: {time.monotonic() - t_total:.1f}s",
          flush=True)

    return RFFRidgeModel(W=W, E=E, omega=omega, b=b, K=K, m=M)
