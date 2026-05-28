"""Closed-form Kernel Ridge Regression over hashed byte n-gram features.

Per experiments/kernel_methods/experiment_01_krr_byte_ngram_baseline.md.

Trains in closed form (no SGD, no autograd):
  W = (Phi^T Phi + lambda I)^{-1} Phi^T Y
where Phi is an (N x F) sparse design matrix of hashed byte n-gram
features over a sliding window of W=16 bytes (n in 1..6), and Y is a
one-hot (N x 256) next-byte target matrix. F = 8192 to fit comfortably in
A100 HBM and keep the F x F Cholesky cheap.

The hashing function is splitmix64 over packed-byte n-gram integers, with
both the table index and a per-feature sign chosen from independent bits
of the hash to reduce systematic collision bias (a-la feature hashing).

Hyperparameter sweep over lambda in {1e-2, 1e0, 1e2} on a small held-out
slice of the training data picks the best ridge strength before returning.
"""
from __future__ import annotations

__author__ = "@ab-10"

import os
import time

import torch
import torch.nn.functional as F

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Hyperparameters (per spec)
# ---------------------------------------------------------------------------

F_FEATS = 8192            # hashing-trick feature dim (fits A100 HBM at 256 MB Cholesky)
N_SAMPLES = 200_000       # subsampled (context, next-byte) pairs
W_CONTEXT = 16            # byte context window
NGRAM_MAX = 6             # n in 1..6
LAMBDAS = (1e-2, 1e0, 1e2)  # ridge sweep
HELDOUT_FRAC = 0.05       # of N_SAMPLES used to pick lambda


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

# splitmix64 constants — re-interpreted as signed int64 two's-complement
# because PyTorch int64 is signed and Python literals for the raw uint64
# values overflow on tensor construction.
def _to_signed_i64(u: int) -> int:
    u &= (1 << 64) - 1
    return u - (1 << 64) if u >= (1 << 63) else u


_SPLITMIX_C1 = _to_signed_i64(0x9E3779B97F4A7C15)
_SPLITMIX_C2 = _to_signed_i64(0xBF58476D1CE4E5B9)
_SPLITMIX_C3 = _to_signed_i64(0x94D049BB133111EB)


def _lsr(x: torch.Tensor, n: int) -> torch.Tensor:
    """Logical (unsigned) right shift on int64.

    PyTorch's ``>>`` on int64 is arithmetic (sign-extends), but splitmix64
    wants logical shifts — mask off the sign-extended high bits.
    """
    mask = (1 << (64 - n)) - 1
    return (x >> n) & mask


@torch.no_grad()
def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    """splitmix64 finalizer over an int64 tensor. Vectorized on GPU.

    Standard avalanche-mixer used in many fast hashes (e.g. xoroshiro).
    We don't care about cryptographic strength — we want low collision
    rate of structured byte n-gram inputs into a small F=8192 table.

    All ops mod 2^64: int64 + and * wrap implicitly; we use logical right
    shift (``_lsr``) to avoid sign extension contaminating the mixed bits.
    """
    x = x + _SPLITMIX_C1
    x = x ^ _lsr(x, 30)
    x = x * _SPLITMIX_C2
    x = x ^ _lsr(x, 27)
    x = x * _SPLITMIX_C3
    x = x ^ _lsr(x, 31)
    return x


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Per-n weight: 1/n (per spec). Stored as a float tensor for the sparse build.
def _ngram_weight(n: int) -> float:
    return 1.0 / n


@torch.no_grad()
def _build_phi_indices(
    contexts: torch.Tensor,  # (N, W) uint8
    F_feats: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the sparse-feature (row, col, val) triples for a batch of
    N byte contexts.

    For each row n, for each n-gram (n=1..NGRAM_MAX) ending in the
    last n bytes of the W-byte window:
      - pack the n bytes into an int64 key with an n-prefix to avoid
        cross-n hash collisions
      - hash to a column in [0, F)
      - sign-flip with the topmost mixed bit, so weighted sums are
        unbiased under collisions (feature hashing trick).

    Returns:
        rows: (M,) long
        cols: (M,) long
        vals: (M,) float32

    where M = N * total_ngrams_per_row.
    """
    N, W = contexts.shape
    contexts_i64 = contexts.to(torch.int64).to(device)  # (N, W)

    all_rows = []
    all_cols = []
    all_vals = []

    # Row indices template — will be re-used per n.
    row_template = torch.arange(N, device=device, dtype=torch.int64)

    for n in range(1, NGRAM_MAX + 1):
        # Number of n-grams in a W-byte window = W - n + 1.
        n_pos = W - n + 1
        if n_pos <= 0:
            continue

        # Pack each n-byte n-gram into an int64. Highest 8 bits hold
        # `n` itself so 1-grams and 2-grams etc don't collide in the
        # hash input space.
        # keys: (N, n_pos)
        keys = torch.full((N, n_pos), n, dtype=torch.int64, device=device) << 56
        for j in range(n):
            # n-gram covers positions [start, start+n) for start in [0, n_pos)
            # so byte at offset j contributes contexts[:, start+j]
            byte_slice = contexts_i64[:, j : j + n_pos]  # (N, n_pos)
            keys = keys | (byte_slice << (j * 8))

        # Hash.
        h = _splitmix64(keys)  # (N, n_pos), int64 (sign-bits arbitrary)

        # Use lower bits for column, one upper bit for sign.
        cols = (h & (F_feats - 1)).to(torch.int64)
        # Sign bit: bit 33 of the hash (well-mixed, independent of cols).
        sign_bit = (h >> 33) & 1
        signs = (1 - 2 * sign_bit).to(torch.float32)
        weight = _ngram_weight(n)
        vals = signs * weight

        # Row indices: repeat row id n_pos times each.
        rows = row_template.unsqueeze(1).expand(N, n_pos).reshape(-1)
        cols = cols.reshape(-1)
        vals = vals.reshape(-1)

        all_rows.append(rows)
        all_cols.append(cols)
        all_vals.append(vals)

    rows = torch.cat(all_rows, dim=0)
    cols = torch.cat(all_cols, dim=0)
    vals = torch.cat(all_vals, dim=0)
    return rows, cols, vals


@torch.no_grad()
def _phi_features_dense_row(
    context: torch.Tensor,  # (W,) uint8
    F_feats: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a single dense feature row for a length-W byte context.

    Used at inference time (one row per predict() call). For F=8192 a
    dense vector is ~32 KB — trivially fast on the GPU.
    """
    rows, cols, vals = _build_phi_indices(context.unsqueeze(0), F_feats, device)
    phi = torch.zeros(F_feats, dtype=torch.float32, device=device)
    phi.scatter_add_(0, cols, vals)
    return phi


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sample_contexts(
    train_bytes: torch.Tensor,  # (T,) uint8 on device
    n_samples: int,
    W: int,
    device: torch.device,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample n_samples (context, target_byte) pairs.

    Returns (contexts, targets):
        contexts: (n_samples, W) uint8
        targets:  (n_samples,) int64
    """
    T = train_bytes.numel()
    if T < W + 1:
        raise ValueError(f"need at least W+1 = {W + 1} bytes, got {T}")
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.manual_seed(0)
    # Indices i in [0, T - W - 1], window = bytes[i : i+W], target = bytes[i+W]
    idx = torch.randint(0, T - W, (n_samples,), generator=g, device=device)
    # Gather (n_samples, W) windows.
    offsets = idx.unsqueeze(1) + torch.arange(W, device=device).unsqueeze(0)  # (N, W)
    contexts = train_bytes[offsets]
    targets = train_bytes[idx + W].to(torch.int64)
    return contexts, targets


@torch.no_grad()
def _build_sparse_phi(
    contexts: torch.Tensor,
    F_feats: int,
    device: torch.device,
) -> torch.Tensor:
    """Build (N, F) sparse_coo Phi from contexts."""
    N = contexts.shape[0]
    rows, cols, vals = _build_phi_indices(contexts, F_feats, device)
    indices = torch.stack([rows, cols], dim=0)
    # COO coalesce will sum repeat entries — desired (collisions of
    # different ngrams into the same col on the same row sum, like
    # standard feature hashing).
    phi = torch.sparse_coo_tensor(
        indices, vals, size=(N, F_feats), device=device, dtype=torch.float32
    ).coalesce()
    return phi


@torch.no_grad()
def _compute_gram_and_phity(
    phi: torch.Tensor,            # sparse (N, F)
    targets: torch.Tensor,        # (N,) int64
    F_feats: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (PhiT_Phi, PhiT_Y) — dense (F, F) and (F, 256).

    Implementation: instead of materializing a dense (N, F) matrix
    (which is 6.5 GB for N=200K, F=8192 in fp32), we use a sparse @
    sparse matmul for PhiT_Phi and sparse @ dense scatter for PhiT_Y.

    Sparse @ sparse on the GPU produces a sparse result; we densify it
    once at the end. For 8192x8192 fp32 that's only 256 MB.
    """
    # PhiT_Phi: ( (N x F).T ) @ (N x F)  =  (F x F)
    phi_t = phi.transpose(0, 1).coalesce()
    # Sparse-sparse matmul (CUDA backend) — returns sparse COO.
    # Densify with .to_dense().
    ptp_sparse = torch.sparse.mm(phi_t, phi)  # (F, F) sparse
    PtP = ptp_sparse.to_dense() if ptp_sparse.is_sparse else ptp_sparse
    PtP = PtP.contiguous()

    # PhiT_Y: build sparse Y as (N, 256), then PhiT @ Y; but it's
    # simpler/faster to scatter-add directly: for each non-zero (i, k)
    # in phi with value v, add v to PhiT_Y[k, targets[i]].
    PtY = torch.zeros(F_feats, 256, dtype=torch.float32, device=device)
    phi_c = phi.coalesce()
    nz_indices = phi_c.indices()  # (2, M)
    nz_values = phi_c.values()    # (M,)
    rows_idx = nz_indices[0]
    cols_idx = nz_indices[1]
    target_class = targets[rows_idx]  # (M,) int64
    # Linear index into the (F, 256) buffer.
    lin = cols_idx * 256 + target_class
    PtY.view(-1).scatter_add_(0, lin, nz_values)

    return PtP, PtY


@torch.no_grad()
def _eval_acc(
    W_mat: torch.Tensor,        # (F, 256)
    contexts: torch.Tensor,     # (n, W) uint8
    targets: torch.Tensor,      # (n,) int64
    F_feats: int,
    device: torch.device,
    batch: int = 1024,
) -> float:
    """Compute argmax-prediction accuracy on a held-out subset.

    Builds sparse Phi for the held-out rows, computes Phi @ W via
    sparse @ dense, argmaxes, compares to targets.
    """
    n = contexts.shape[0]
    correct = 0
    for i in range(0, n, batch):
        end = min(n, i + batch)
        ctx = contexts[i:end]
        tgt = targets[i:end]
        phi = _build_sparse_phi(ctx, F_feats, device)
        logits = torch.sparse.mm(phi, W_mat)  # (b, 256)
        pred = logits.argmax(dim=-1)
        correct += int((pred == tgt).sum().item())
    return correct / max(1, n)


# ---------------------------------------------------------------------------
# CharModel wrapper
# ---------------------------------------------------------------------------

class KRRNgramCharModel(CharModel):
    """Streaming CharModel that hashes the rolling W-byte context and
    multiplies by the pre-computed weight matrix W to get next-byte logits.
    """

    def __init__(self, W_mat: torch.Tensor, F_feats: int, W_ctx: int):
        self.W_mat = W_mat  # (F, 256) float32 on GPU
        self.F_feats = F_feats
        self.W_ctx = W_ctx
        self.device = W_mat.device
        # Rolling buffer of last W bytes. Zero-padded at start.
        self._buf = bytearray(W_ctx)

    def reset(self) -> None:
        self._buf = bytearray(self.W_ctx)

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        ctx = torch.frombuffer(bytes(self._buf), dtype=torch.uint8).to(self.device)
        phi = _phi_features_dense_row(ctx, self.F_feats, self.device)  # (F,)
        # Logits: phi @ W_mat -> (256,)
        logits = phi @ self.W_mat
        probs = torch.softmax(logits.float(), dim=-1)
        out: dict[str, float] = {}
        for byte_id, p in enumerate(probs.tolist()):
            try:
                ch = bytes([byte_id]).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out[ch] = p
        return out

    def observe(self, char: str) -> None:
        for byte in char.encode("utf-8"):
            self._buf.append(byte)
            # Trim to W bytes (rolling window keeps the most recent).
            if len(self._buf) > self.W_ctx:
                del self._buf[0 : len(self._buf) - self.W_ctx]


# ---------------------------------------------------------------------------
# train() entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[krr_ngram] device={device}  F={F_FEATS}  N={N_SAMPLES}  "
          f"W={W_CONTEXT}  ngrams=1..{NGRAM_MAX}  lambdas={LAMBDAS}")

    t_start = time.monotonic()

    # 1) Move training bytes to GPU.
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    print(f"[krr_ngram] train bytes: {train_bytes.numel():,}")

    # 2) Sample N (context, target) pairs.
    contexts, targets = _sample_contexts(
        train_bytes, N_SAMPLES, W_CONTEXT, device, seed=seed
    )
    # Split: heldout for lambda selection.
    n_heldout = max(1024, int(N_SAMPLES * HELDOUT_FRAC))
    n_train = N_SAMPLES - n_heldout
    train_ctx, val_ctx = contexts[:n_train], contexts[n_train:]
    train_tgt, val_tgt = targets[:n_train], targets[n_train:]
    print(f"[krr_ngram] sampled {N_SAMPLES:,} pairs "
          f"(train={n_train:,}, heldout={n_heldout:,})")

    # 3) Build sparse Phi for the training pairs.
    t0 = time.monotonic()
    phi = _build_sparse_phi(train_ctx, F_FEATS, device)
    print(f"[krr_ngram] built sparse Phi (nnz={phi._nnz():,}) "
          f"in {time.monotonic()-t0:.2f}s")

    # 4) Compute PtP and PtY once.
    t0 = time.monotonic()
    PtP, PtY = _compute_gram_and_phity(phi, train_tgt, F_FEATS, device)
    print(f"[krr_ngram] computed PtP {tuple(PtP.shape)} and PtY "
          f"{tuple(PtY.shape)} in {time.monotonic()-t0:.2f}s")

    # 5) Sweep lambda — solve each ridge problem, score on heldout.
    eye = torch.eye(F_FEATS, dtype=torch.float32, device=device)
    best_acc = -1.0
    best_W: torch.Tensor | None = None
    best_lam: float | None = None
    for lam in LAMBDAS:
        t0 = time.monotonic()
        A = PtP + lam * eye
        # Solve A @ W = PtY for W ∈ (F, 256).
        try:
            W_mat = torch.linalg.solve(A, PtY)
        except RuntimeError as e:
            # Singular — jitter and retry.
            print(f"[krr_ngram] lambda={lam:g} solve failed ({e!r}); jittering")
            A = A + 1e-3 * eye
            W_mat = torch.linalg.solve(A, PtY)
        t_solve = time.monotonic() - t0

        # Heldout accuracy (proxy for val acc).
        t0 = time.monotonic()
        acc = _eval_acc(W_mat, val_ctx, val_tgt, F_FEATS, device)
        t_eval = time.monotonic() - t0
        print(f"[krr_ngram] lambda={lam:>8.2e}  solve={t_solve:.2f}s  "
              f"heldout_acc={acc:.4f}  ({t_eval:.2f}s eval)")
        if acc > best_acc:
            best_acc = acc
            best_W = W_mat
            best_lam = lam

    assert best_W is not None
    print(f"[krr_ngram] best lambda={best_lam:g}  heldout_acc={best_acc:.4f}  "
          f"total_train_time={time.monotonic()-t_start:.2f}s")

    return KRRNgramCharModel(best_W, F_FEATS, W_CONTEXT)
