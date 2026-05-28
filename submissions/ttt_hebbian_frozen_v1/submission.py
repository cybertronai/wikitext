"""TTT-Linear with Hebbian inner loop + frozen random outer projections.

Per `research/outer_aggressive_gradfree/03_ttt_hebbian_inner_loop_no_outer.md`.

This is the most aggressive simplification possible: every "learning" is a
single Hebbian outer product write. No backprop, no SGD on any "outer"
parameter. The only thing that adapts is the in-context fast-weight matrix W.

Per-byte dynamics (Schlag-Schmidhuber 2021 equivalence: delta-rule = SGD-on-MSE):
    z_t = phi(window_t)                        # frozen RFF feature on last K bytes
    pred_t = W_{t-1} z_t                       # READ
    target_t = psi(byte_t) = E_targ[byte_t]    # frozen random target embedding
    W_t = W_{t-1} + eta * (target_t - pred_t) z_t^T   # WRITE (delta rule)

Frozen outer projections:
    - phi: byte_window (K bytes) -> R^d via cos(R_phi @ onehot + b_phi)/sqrt(d).
    - psi: byte -> R^d via a fixed random Gaussian E_targ: (256, d).

Readout R: (d, 256) — fit by closed-form RIDGE REGRESSION on
(W_t z_t, one_hot(byte_{t+1})) pairs collected by streaming the training text
through the dynamics. Ridge lambda selected on a 5% held-out split.
NO SGD anywhere.

Architecture variant `ttt_hebbian_frozen_v1`:
    d = 512, K = 64, eta = 0.1, single layer.
"""
from __future__ import annotations

__author__ = "@armin-claude-1m"

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

D_MODEL = 512          # fast-weight matrix is (d, d) = 512x512 fp32 = 1 MB
K_CTX = 64             # rolling byte-window length
ETA = 0.1              # Hebbian learning rate (delta-rule step size)
LAMBDAS = (1e-2, 1e0, 1e2)   # ridge regularization sweep
HELDOUT_FRAC = 0.05

# Streaming dynamics collection
B_STREAM = 128         # parallel windows per scan call
T_STREAM = 512         # window length per scan call
N_PAIRS_TARGET = 200_000     # collected (pred, next_byte) pairs for ridge fit
PAIRS_PER_SCAN = 96    # subsample positions per (B, T) scan -> B * pairs_per_scan
TRAIN_BUDGET_S = 250.0       # leave 50s safety margin under task.MAX_TRAIN_SECONDS=300
COLLECT_BUDGET_S = 150.0     # phase A
LOG_EVERY_S = 10.0

# Seed handling
def _seed() -> int:
    env = os.environ.get("SEED")
    return int(env) if env else 0


# ---------------------------------------------------------------------------
# Frozen projections (RFF feature map + target embedding)
# ---------------------------------------------------------------------------

class FrozenProjections:
    """Deterministic random projections seeded once at construction.

    phi(window_bytes) = cos(R_phi @ onehot_concat + b_phi) * sqrt(2/d)
        where onehot_concat ∈ R^{K*256} is concat of K one-hot byte vectors.

    psi(byte) = E_targ[byte] where E_targ ∈ R^{256, d} ~ N(0, 1/d).

    Implementation note: we never materialize the (K*256)-dim one-hot vector.
    Instead R_phi is reshaped to (d, K, 256) and we sum the per-position
    embeddings R_phi[:, i, byte_i] over i — exactly equivalent to the
    matmul against the one-hot concat.
    """

    def __init__(self, d: int, K: int, device: torch.device, seed: int):
        self.d = d
        self.K = K
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(seed)

        # R_phi has shape (d, K*256); we factor it as (d, K, 256). Gaussian
        # entries with std 1/sqrt(K*256/2) such that the linear preactivation
        # has unit variance under iid uniform bytes (rough — exact sigma
        # doesn't matter much for RFF cosines downstream).
        # Use Bochner-style RFF: sigma chosen so cos features have meaningful
        # variance. Std = 1/sqrt(K) is a robust default.
        std = 1.0 / math.sqrt(K)
        R = torch.randn(d, K, 256, generator=g, dtype=torch.float32) * std
        b = (torch.rand(d, generator=g, dtype=torch.float32) * 2 * math.pi)
        self.R_phi = R.to(device)  # (d, K, 256)
        self.b_phi = b.to(device)  # (d,)

        # Target embedding: (256, d), iid N(0, 1/d) so ||target|| ~ 1.
        E = torch.randn(256, d, generator=g, dtype=torch.float32) / math.sqrt(d)
        self.E_targ = E.to(device)  # (256, d)

        # Normalization factor for cosine feature (sqrt(2/d) per RFF).
        self.phi_scale = math.sqrt(2.0 / d)

    @torch.no_grad()
    def phi_window_batch(self, windows: Tensor) -> Tensor:
        """Compute phi for a batch of byte windows.

        Args:
            windows: (..., K) int64 byte ids (values 0..255).

        Returns:
            features: (..., d) float32.
        """
        # Gather R_phi[:, i, windows[..., i]] -> (..., K, d), then sum over K.
        # We use advanced indexing: for each position i in [0,K), pick the
        # column 'byte' from R_phi[:, i, :].
        # R_phi is (d, K, 256). Transpose to (K, 256, d).
        # Then for each batch element, for each position i, look up row
        # R_phi_kd[i, byte_i, :] and sum across i.
        d, K, _ = self.R_phi.shape
        assert windows.shape[-1] == K
        # Permute R_phi to (K, 256, d) for advanced indexing on (K, 256).
        R_kd = self.R_phi.permute(1, 2, 0).contiguous()  # (K, 256, d)
        # Build position-index tensor matching windows shape.
        # Use gather along an expanded view.
        # Flatten batch dims.
        batch_shape = windows.shape[:-1]
        flat = windows.reshape(-1, K).to(torch.int64)  # (N, K)
        N = flat.shape[0]
        # For each (n, i), pick R_kd[i, flat[n, i], :] -> (N, K, d)
        # Vectorize: linear-index R_kd as a (K*256, d) matrix.
        R_flat = R_kd.reshape(K * 256, d)  # (K*256, d)
        i_idx = torch.arange(K, device=flat.device).unsqueeze(0).expand(N, K)  # (N, K)
        lin = i_idx * 256 + flat  # (N, K)
        gathered = R_flat[lin]  # (N, K, d)
        preact = gathered.sum(dim=1) + self.b_phi  # (N, d)
        feats = torch.cos(preact) * self.phi_scale
        return feats.reshape(*batch_shape, d)

    @torch.no_grad()
    def phi_window_single(self, window: Tensor) -> Tensor:
        """phi for a single (K,) byte window. Returns (d,) fp32."""
        return self.phi_window_batch(window.unsqueeze(0)).squeeze(0)

    @torch.no_grad()
    def psi(self, bytes_t: Tensor) -> Tensor:
        """Target embedding for a batch of byte ids (..., ) -> (..., d)."""
        return self.E_targ[bytes_t.to(torch.int64)]


# ---------------------------------------------------------------------------
# Phase A: streaming dynamics collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_pairs(
    train_bytes: Tensor,           # (T_total,) uint8 on device
    proj: FrozenProjections,
    eta: float,
    budget_s: float,
    n_pairs_target: int,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Run the delta-rule scan over random (B, T) byte chunks and collect
    (pred, next_byte) pairs at randomly chosen positions.

    Returns:
        X: (N, d) fp32 — pred vectors W_t z_t at sampled positions.
        Y: (N,) int64 — next byte at sampled positions.
    """
    d = proj.d
    K = proj.K
    T_total = train_bytes.numel()
    if T_total < K + T_STREAM + 1:
        raise ValueError(
            f"train text too short: need at least {K + T_STREAM + 1} bytes, "
            f"got {T_total}"
        )

    # Preallocate output buffers — slight overshoot okay, we trim at end.
    cap = n_pairs_target + B_STREAM * PAIRS_PER_SCAN
    X_buf = torch.empty(cap, d, dtype=torch.float32, device=device)
    Y_buf = torch.empty(cap, dtype=torch.int64, device=device)
    n_collected = 0

    t_start = time.monotonic()
    t_last_log = t_start
    n_scans = 0

    # Precompute time index arange for window indexing.
    arange_T = torch.arange(T_STREAM, device=device)
    arange_K = torch.arange(K, device=device)

    while n_collected < n_pairs_target:
        elapsed = time.monotonic() - t_start
        if elapsed > budget_s:
            print(
                f"[ttt_hebbian] phase A budget hit ({elapsed:.1f}s); "
                f"collected {n_collected:,} pairs in {n_scans} scans"
            )
            break

        # Sample B starting positions; window [start, start + K + T_STREAM)
        # so byte at absolute pos (start + K + t) is the target byte for the
        # window ending at (start + K + t - 1).
        max_start = T_total - (K + T_STREAM) - 1
        starts = torch.randint(
            0, max_start, (B_STREAM,), generator=gen, device=device,
        )  # (B,)

        # Build windowed-byte tensor: windows[b, t, k] = byte at start[b] + t + k
        # of length K, for t in [0, T_STREAM). This is the byte window FEEDING
        # position t.
        # Shape (B, T, K)
        offs = (
            starts.unsqueeze(1).unsqueeze(2)        # (B, 1, 1)
            + arange_T.unsqueeze(0).unsqueeze(2)    # (1, T, 1)
            + arange_K.unsqueeze(0).unsqueeze(0)    # (1, 1, K)
        )  # (B, T, K)
        windows_BTK = train_bytes[offs].to(torch.int64)  # (B, T, K)

        # next_byte at time t (the target/Hebbian write target) is the byte
        # AFTER the window, i.e. train_bytes[start + K + t].
        next_byte_BT = train_bytes[
            starts.unsqueeze(1) + K + arange_T.unsqueeze(0)
        ].to(torch.int64)  # (B, T)

        # Compute z_seq = phi(window_t) for all (b, t).
        z_seq = proj.phi_window_batch(windows_BTK)  # (B, T, d) fp32

        # Compute target_seq = psi(next_byte_t) for delta-rule writes.
        # The TTT/Hebbian write at step t targets the embedding of the byte
        # being predicted.
        target_seq = proj.psi(next_byte_BT)  # (B, T, d) fp32

        # Run the sequential delta-rule scan and collect a sample of
        # (pred_t, next_byte_{t+1}) pairs. Per spec, we want pairs where the
        # output is W_t z_t and the supervision is the NEXT byte after t.
        # We log pairs at PAIRS_PER_SCAN random positions per batch row.
        # To predict next_byte_{t+1}, we need pred at time t+1 (i.e.,
        # after seeing window ending at t), supervised by byte at t+1.
        # The natural alignment in this scan: at step t we have
        # pred_t = W_{t-1} z_t (using window through t-1's byte), and the
        # target/write is byte at position t (next_byte_BT[b, t]). So
        # pred_t already targets next_byte_BT[b, t] — perfect.
        pos_indices = torch.randint(
            1, T_STREAM, (B_STREAM, PAIRS_PER_SCAN),
            generator=gen, device=device,
        )  # (B, P) — skip t=0 because W=0 there yields all-zero pred

        # Build sample masks for which (b, t) to record.
        sample_mask = torch.zeros(B_STREAM, T_STREAM, dtype=torch.bool, device=device)
        b_idx = torch.arange(B_STREAM, device=device).unsqueeze(1).expand(B_STREAM, PAIRS_PER_SCAN)
        sample_mask[b_idx.reshape(-1), pos_indices.reshape(-1)] = True

        # Sequential scan.
        W = torch.zeros(B_STREAM, d, d, device=device, dtype=torch.float32)
        # Collect into per-step buffers; concat at end.
        sampled_preds: list[Tensor] = []
        sampled_targets: list[Tensor] = []
        for t in range(T_STREAM):
            z = z_seq[:, t]                              # (B, d)
            pred = torch.bmm(W, z.unsqueeze(-1)).squeeze(-1)  # (B, d)

            # Record samples at this t for any batch elements flagged.
            if sample_mask[:, t].any():
                rows = sample_mask[:, t].nonzero(as_tuple=True)[0]  # (k,)
                sampled_preds.append(pred[rows].detach())
                sampled_targets.append(next_byte_BT[rows, t].detach())

            # Hebbian delta-rule WRITE — target is psi(next_byte_t).
            target = target_seq[:, t]                    # (B, d)
            delta = target - pred                        # (B, d)
            # W += eta * outer(delta, z)
            W.add_(torch.einsum("bi,bj->bij", delta, z), alpha=eta)

        if sampled_preds:
            preds_cat = torch.cat(sampled_preds, dim=0)
            targets_cat = torch.cat(sampled_targets, dim=0)
            n_new = preds_cat.shape[0]
            n_take = min(n_new, cap - n_collected)
            X_buf[n_collected:n_collected + n_take] = preds_cat[:n_take]
            Y_buf[n_collected:n_collected + n_take] = targets_cat[:n_take]
            n_collected += n_take

        n_scans += 1
        now = time.monotonic()
        if now - t_last_log > LOG_EVERY_S:
            t_last_log = now
            print(
                f"[ttt_hebbian] phase A scan={n_scans} "
                f"pairs={n_collected:,}/{n_pairs_target:,} "
                f"elapsed={now - t_start:.1f}s",
                flush=True,
            )

    print(
        f"[ttt_hebbian] phase A done: {n_collected:,} pairs "
        f"in {time.monotonic() - t_start:.1f}s ({n_scans} scans)"
    )
    X = X_buf[:n_collected]
    Y = Y_buf[:n_collected]
    return X, Y


# ---------------------------------------------------------------------------
# Phase B: closed-form ridge regression for readout R
# ---------------------------------------------------------------------------

@torch.no_grad()
def _fit_ridge(
    X: Tensor,        # (N, d) fp32
    Y: Tensor,        # (N,) int64
    lambdas: tuple[float, ...],
    heldout_frac: float,
    device: torch.device,
) -> tuple[Tensor, float, float]:
    """Solve R = argmin ||X R - one_hot(Y)||^2 + lambda * ||R||^2 in closed form.

    Sweeps lambda on a held-out split; picks best by held-out argmax accuracy.

    Returns:
        R: (d, 256) fp32
        best_lambda
        best_heldout_acc
    """
    N, d = X.shape
    n_hold = max(1024, int(N * heldout_frac))
    n_fit = N - n_hold
    X_fit, X_hold = X[:n_fit], X[n_fit:]
    Y_fit, Y_hold = Y[:n_fit], Y[n_fit:]

    # Build one-hot Y_fit (n_fit, 256) — float32. 200K * 256 * 4 = 200 MB ok.
    Y_oh = torch.zeros(n_fit, 256, dtype=torch.float32, device=device)
    Y_oh.scatter_(1, Y_fit.unsqueeze(1), 1.0)

    # Gram matrices: (d, d) and (d, 256).
    XtX = X_fit.T @ X_fit         # (d, d)
    XtY = X_fit.T @ Y_oh          # (d, 256)
    eye = torch.eye(d, dtype=torch.float32, device=device)

    best_acc = -1.0
    best_R: Tensor | None = None
    best_lam = lambdas[0]
    for lam in lambdas:
        t0 = time.monotonic()
        A = XtX + lam * eye
        try:
            R = torch.linalg.solve(A, XtY)
        except RuntimeError as e:
            print(f"[ttt_hebbian] ridge solve failed lam={lam}: {e!r}; jittering")
            A = A + 1e-3 * eye
            R = torch.linalg.solve(A, XtY)
        # Held-out accuracy.
        logits = X_hold @ R  # (n_hold, 256)
        pred = logits.argmax(dim=-1)
        acc = float((pred == Y_hold).float().mean().item())
        dt = time.monotonic() - t0
        print(
            f"[ttt_hebbian] ridge lam={lam:>8.2e}  solve={dt:.2f}s  "
            f"heldout_acc={acc:.4f}"
        )
        if acc > best_acc:
            best_acc = acc
            best_R = R
            best_lam = lam

    assert best_R is not None
    return best_R, best_lam, best_acc


# ---------------------------------------------------------------------------
# Streaming CharModel
# ---------------------------------------------------------------------------

class TTTHebbianCharModel(CharModel):
    """Streaming inference: maintain rolling K-byte buffer + fast-weight W.

    Per byte b at position t:
        push b into buffer
        z = phi(buffer)
        pred = W @ z
        target = E_targ[b]
        W += eta * (target - pred) z^T

    predict() uses the CURRENT W and CURRENT buffer (i.e., the byte just
    observed has already been integrated). logits = (W @ z) @ R.
    """

    def __init__(
        self,
        proj: FrozenProjections,
        R_readout: Tensor,
        eta: float,
    ):
        self.proj = proj
        self.R_readout = R_readout       # (d, 256) fp32
        self.eta = float(eta)
        self.d = proj.d
        self.K = proj.K
        self.device = proj.device

        # Streaming state — initialized in reset().
        self._W: Tensor | None = None
        self._buf: Tensor | None = None
        self.reset()

    def reset(self) -> None:
        self._W = torch.zeros(
            self.d, self.d, device=self.device, dtype=torch.float32,
        )
        self._buf = torch.zeros(self.K, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        assert self._W is not None and self._buf is not None
        z = self.proj.phi_window_single(self._buf)            # (d,)
        pred = self._W @ z                                     # (d,)
        logits = pred @ self.R_readout                         # (256,)
        probs = torch.softmax(logits.float(), dim=-1)
        out: dict[str, float] = {}
        for byte_id, p in enumerate(probs.tolist()):
            try:
                ch = bytes([byte_id]).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out[ch] = p
        return out

    @torch.no_grad()
    def observe(self, char: str) -> None:
        assert self._W is not None and self._buf is not None
        for byte in char.encode("utf-8"):
            # Roll the buffer left, append the new byte at the end.
            self._buf[:-1] = self._buf[1:].clone()
            self._buf[-1] = int(byte)
            # Compute z from the updated window.
            z = self.proj.phi_window_single(self._buf)         # (d,)
            pred = self._W @ z                                  # (d,)
            target = self.proj.E_targ[int(byte)]                # (d,)
            delta = target - pred                               # (d,)
            # W += eta * outer(delta, z)
            self._W.add_(torch.outer(delta, z), alpha=self.eta)


# ---------------------------------------------------------------------------
# train() entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    t_train_start = time.monotonic()
    seed = _seed()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[ttt_hebbian] device={device}  d={D_MODEL}  K={K_CTX}  eta={ETA}  "
        f"B={B_STREAM}  T={T_STREAM}  n_pairs={N_PAIRS_TARGET}  "
        f"lambdas={LAMBDAS}  seed={seed}"
    )

    # Move training bytes to device.
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    print(f"[ttt_hebbian] train bytes: {train_bytes.numel():,}")

    # Build frozen projections (deterministic via seed).
    proj = FrozenProjections(d=D_MODEL, K=K_CTX, device=device, seed=seed)
    print(
        f"[ttt_hebbian] frozen proj: R_phi {tuple(proj.R_phi.shape)}  "
        f"E_targ {tuple(proj.E_targ.shape)}"
    )

    # Phase A: collect (pred, next_byte) pairs.
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 17)
    X, Y = _collect_pairs(
        train_bytes=train_bytes,
        proj=proj,
        eta=ETA,
        budget_s=COLLECT_BUDGET_S,
        n_pairs_target=N_PAIRS_TARGET,
        device=device,
        gen=gen,
    )

    if X.shape[0] < 1024:
        raise RuntimeError(
            f"phase A collected only {X.shape[0]} pairs — too few to fit ridge"
        )

    # Phase B: closed-form ridge solve for readout R.
    t_b_start = time.monotonic()
    print(
        f"[ttt_hebbian] phase B: ridge solve on X{tuple(X.shape)} Y{tuple(Y.shape)}"
    )
    R, best_lam, best_acc = _fit_ridge(
        X=X,
        Y=Y,
        lambdas=LAMBDAS,
        heldout_frac=HELDOUT_FRAC,
        device=device,
    )
    print(
        f"[ttt_hebbian] phase B done in {time.monotonic() - t_b_start:.1f}s  "
        f"best_lambda={best_lam:g}  heldout_acc={best_acc:.4f}"
    )

    total = time.monotonic() - t_train_start
    print(f"[ttt_hebbian] total train time: {total:.1f}s")

    return TTTHebbianCharModel(proj=proj, R_readout=R, eta=ETA)
