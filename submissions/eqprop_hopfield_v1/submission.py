"""Equilibrium Propagation on a Modern Hopfield Energy (Ramsauer 2020).

Per research/outer_aggressive_gradfree/09_eqprop_modern_hopfield_energy.md.

A fully gradient-free char-LM. The ONLY learnable object is a pattern
matrix Xi in R^{(d_c + 256) x M}; updates are pure outer-product
differences (Scellier-Bengio EqProp, 2017), with no autograd, no
optimizer state, no backprop in any direction.

Pipeline:
  1. Frozen Random-Fourier featurizer phi: last-256-bytes one-hot -> R^{d_c}.
     One fixed random projection matrix R_phi in R^{d_c x (256 * 256)},
     bias b in R^{d_c}, phi(x) = sqrt(2/d_c) * cos(R_phi x + b).
  2. State sigma = (context_block in R^{d_c}, target_block in R^{256}).
  3. Energy E(sigma; Xi, beta) = -(1/beta) lse(beta Xi^T sigma) + (1/2)||sigma||^2
     (Ramsauer 2020 modern Hopfield energy). The fixed-point retrieval
     dynamics are exactly one softmax-attention call:
         sigma_new = Xi @ softmax(beta * Xi^T @ sigma).
  4. Free phase:   start from (phi, zeros_256), 2 relaxation steps.
  5. Nudged phase: start from (phi, (1 - beta_nudge)*zeros + beta_nudge*onehot),
                   2 relaxation steps.
  6. EqProp update: at a fixed-point sigma*, the gradient of E w.r.t. Xi
     is sigma* @ softmax(beta Xi^T sigma*)^T (an outer product). The
     contrastive update is
         Xi += eta * (1/beta_nudge) * (grad_nudge - grad_free)
     - i.e. the *difference of two batched outer products*. NO backprop.

At inference time we run the free phase (2 relaxation steps from
(phi(rolling-256-byte-buffer), zeros_256)), read out the target block,
softmax it, and emit a per-byte distribution.

Budget: 250s training cap, ~3000 steps at B=64, M=4096.

Expected val char-acc: 0.30-0.50 (DQ below 0.70 is the EXPECTED
outcome; this is a capability demonstration, not a leaderboard push).
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
# Hyperparameters (per spec)
# ---------------------------------------------------------------------------

CTX_BYTES = 256          # rolling context window in bytes
D_C = 512                # RFF context-feature dim
M = 4096                 # number of Hopfield patterns
BETA = 4.0               # Ramsauer inverse temperature for retrieval softmax
BETA_NUDGE = 0.5         # EqProp nudge strength (target-block soft-clamp)
N_RELAX = 2              # relaxation steps per phase (free, nudged)
BATCH_SIZE = 64          # training batch size
ETA = 0.1                # EqProp learning rate
N_STEPS_MAX = 3000       # training-step ceiling (also capped by wall clock)
TRAIN_BUDGET_S = 250.0   # leave 50s of headroom under MAX_TRAIN_SECONDS=300
LOG_EVERY = 100
XI_INIT_STD = 0.02       # small-random-Gaussian init for the pattern matrix


# Derived
TARGET_DIM = 256                  # one byte = 256 classes
STATE_DIM = D_C + TARGET_DIM      # full sigma dim
ONEHOT_DIM = CTX_BYTES * 256      # flattened one-hot input to the RFF


# ---------------------------------------------------------------------------
# Frozen RFF featurizer
# ---------------------------------------------------------------------------

class RFFFeaturizer:
    """Random Fourier Features over the last CTX_BYTES-byte one-hot.

    Frozen — never updated. The projection matrix R_phi in R^{d_c x 65536}
    would be ~128 MB in fp32 (256*256*512*4 bytes). Instead of materializing
    the full one-hot vector x in R^{65536} (sparse with CTX_BYTES non-zeros)
    and doing a dense matmul, we gather the (1, d_c) column for each byte
    position * byte-value combination and sum -> exactly equivalent to
    R_phi @ x but avoids materializing the sparse input.

    Specifically: x is the flat one-hot of the (pos, byte) pairs. Its
    j-th non-zero index is pos * 256 + byte at position pos. So
    (R_phi @ x)[k] = sum_{pos} R_phi[k, pos * 256 + buf[pos]].

    We pre-materialize R_phi as a (D_C, 65536) fp32 tensor. 128 MB is
    fine on A100-80GB.
    """

    def __init__(self, device: torch.device, seed: int = 0):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        # Standard RFF bandwidth: for one-hot inputs, ||x||_2 = sqrt(CTX_BYTES)
        # so a unit-variance Gaussian projection gives features at the natural
        # scale. We use std = 1 / sqrt(CTX_BYTES) so R_phi @ x has unit-ish
        # variance across rows -> cos saturates less.
        self.scale = 1.0 / math.sqrt(CTX_BYTES)
        self.R = torch.randn(
            D_C, ONEHOT_DIM, generator=g, device=device, dtype=torch.float32
        ) * self.scale
        self.b = torch.rand(
            D_C, generator=g, device=device, dtype=torch.float32
        ) * (2.0 * math.pi)
        self.norm = math.sqrt(2.0 / D_C)
        self.device = device

    @torch.no_grad()
    def features_batch(self, contexts: Tensor) -> Tensor:
        """Compute phi(x) for a batch of byte windows.

        Args:
            contexts: (B, CTX_BYTES) uint8 or int64. Each row is the
                last-CTX_BYTES-byte history.

        Returns:
            (B, D_C) float32 tensor of RFF features.
        """
        B, T = contexts.shape
        assert T == CTX_BYTES, f"expected CTX_BYTES={CTX_BYTES}, got {T}"
        ctx_i64 = contexts.to(torch.int64)
        # Flat indices into R's columns. shape (B, CTX_BYTES).
        pos = torch.arange(CTX_BYTES, device=contexts.device, dtype=torch.int64)
        flat_idx = pos.unsqueeze(0) * 256 + ctx_i64  # (B, CTX_BYTES)
        # Gather columns: R[:, flat_idx] -> (D_C, B, CTX_BYTES), then sum
        # over the CTX_BYTES axis to mimic R @ x.
        # Equivalent (and much cheaper-memory): einsum-style index_select.
        # R.T is (ONEHOT_DIM, D_C); we want (B, D_C) = sum over t of R[:, idx[b,t]].
        # Use embedding lookup over R.T (treating each column of R as a "token
        # embedding").
        # R.T has shape (ONEHOT_DIM, D_C).
        # F.embedding(flat_idx, R.T) -> (B, CTX_BYTES, D_C); sum over CTX_BYTES.
        cols = F.embedding(flat_idx, self.R.t())  # (B, CTX_BYTES, D_C)
        proj = cols.sum(dim=1)  # (B, D_C)
        out = self.norm * torch.cos(proj + self.b)  # (B, D_C)
        return out

    @torch.no_grad()
    def features_single(self, context: Tensor) -> Tensor:
        """Single-row variant for streaming inference. Returns (D_C,) fp32."""
        return self.features_batch(context.unsqueeze(0))[0]


# ---------------------------------------------------------------------------
# Hopfield relaxation
# ---------------------------------------------------------------------------

@torch.no_grad()
def hopfield_relax(
    sigma: Tensor,       # (B, STATE_DIM) fp32
    Xi: Tensor,          # (STATE_DIM, M) fp32
    beta: float,
    n_steps: int,
    clamp_context: Tensor | None = None,   # (B, D_C) fp32, re-clamped each step
    clamp_target: Tensor | None = None,    # (B, TARGET_DIM) fp32, re-clamped each step
) -> Tensor:
    """Run n_steps of Ramsauer Hopfield retrieval.

    Each step:
        attn = softmax(beta * Xi^T @ sigma)        in R^M
        sigma = Xi @ attn                           in R^STATE_DIM

    If clamp_context is provided, the first D_C entries of sigma are
    overwritten with clamp_context after every step (hard clamp on the
    context block - it's the "input" of the EBM).
    If clamp_target is provided, the last TARGET_DIM entries are
    overwritten with clamp_target after every step.
    """
    for _ in range(n_steps):
        # (B, M) = (B, STATE_DIM) @ (STATE_DIM, M)
        scores = sigma @ Xi
        attn = F.softmax(beta * scores, dim=-1)
        # (B, STATE_DIM) = (B, M) @ (M, STATE_DIM)
        sigma = attn @ Xi.t()
        if clamp_context is not None:
            sigma[:, :D_C] = clamp_context
        if clamp_target is not None:
            sigma[:, D_C:] = clamp_target
    return sigma


@torch.no_grad()
def hopfield_attn(
    sigma: Tensor,       # (B, STATE_DIM)
    Xi: Tensor,          # (STATE_DIM, M)
    beta: float,
) -> Tensor:
    """Return the (B, M) softmax-attention vector at sigma."""
    return F.softmax(beta * (sigma @ Xi), dim=-1)


# ---------------------------------------------------------------------------
# EqProp training
# ---------------------------------------------------------------------------

@torch.no_grad()
def eqprop_step(
    Xi: Tensor,          # (STATE_DIM, M) fp32 -- updated in-place
    phi: Tensor,         # (B, D_C) fp32, context features (frozen, clamped)
    targets: Tensor,     # (B,) int64, true next-byte
    beta: float,
    beta_nudge: float,
    n_relax: int,
    eta: float,
) -> None:
    """One EqProp update of Xi from a batch of (phi, target) pairs.

    Free phase:
        sigma_0 = (phi, zeros_256)
        sigma_free = hopfield_relax(sigma_0, Xi, n_relax) with context clamped.

    Nudged phase:
        target_block_0 = (1 - beta_nudge)*zeros + beta_nudge*onehot(target)
        sigma_0_n = (phi, target_block_0)
        sigma_nudge = hopfield_relax(sigma_0_n, Xi, n_relax) with context clamped.

    Update:
        attn_free  = softmax(beta Xi^T sigma_free)
        attn_nudge = softmax(beta Xi^T sigma_nudge)
        grad_free  = sigma_free^T  @ attn_free    # (STATE_DIM, M)  outer-product sum
        grad_nudge = sigma_nudge^T @ attn_nudge   # (STATE_DIM, M)
        Xi += eta * (1 / beta_nudge) * (grad_nudge - grad_free)

    All in fp32. NO backprop, NO autograd, NO optimizer state.
    """
    B = phi.shape[0]
    device = Xi.device

    # ---- Free phase ----
    zeros_tgt = torch.zeros(B, TARGET_DIM, device=device, dtype=torch.float32)
    sigma_free = torch.cat([phi, zeros_tgt], dim=-1)  # (B, STATE_DIM)
    sigma_free = hopfield_relax(
        sigma_free, Xi, beta, n_relax,
        clamp_context=phi, clamp_target=None,
    )

    # ---- Nudged phase ----
    onehot = F.one_hot(targets, num_classes=TARGET_DIM).to(torch.float32)  # (B, 256)
    target_init = beta_nudge * onehot  # (1 - beta_nudge)*0 + beta_nudge*onehot
    sigma_nudge = torch.cat([phi, target_init], dim=-1)
    sigma_nudge = hopfield_relax(
        sigma_nudge, Xi, beta, n_relax,
        clamp_context=phi, clamp_target=None,
    )

    # ---- Outer-product gradients ----
    attn_free = hopfield_attn(sigma_free, Xi, beta)    # (B, M)
    attn_nudge = hopfield_attn(sigma_nudge, Xi, beta)  # (B, M)

    # grad = sum_b sigma_b outer attn_b   ->  (STATE_DIM, M)
    grad_free = sigma_free.t() @ attn_free    # (STATE_DIM, B) @ (B, M)
    grad_nudge = sigma_nudge.t() @ attn_nudge

    # EqProp update. /B normalizes the batch sum to a mean.
    scale = eta * (1.0 / beta_nudge) / float(B)
    Xi.add_(grad_nudge - grad_free, alpha=scale)


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

def _make_train_bytes(train_text: str, device: torch.device) -> Tensor:
    raw = train_text.encode("utf-8")
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)


@torch.no_grad()
def _sample_windows(
    train_bytes: Tensor,
    batch_size: int,
    ctx_bytes: int,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample batch_size (context, target) pairs uniformly at random.

    context: (B, ctx_bytes) uint8, target: (B,) int64.
    Pads with zeros at the start if needed (when sampled index < ctx_bytes).
    """
    n = train_bytes.numel()
    # Valid target positions: ctx_bytes ... n - 1 (so that bytes[t-ctx_bytes:t]
    # is a full ctx_bytes window and bytes[t] is the target).
    # We instead sample target positions in [0, n - 1], and left-pad with zeros
    # for the early positions to make zero-padding match streaming inference.
    target_pos = torch.randint(0, n, (batch_size,), generator=gen, device=device)

    # Build (B, ctx_bytes) windows: bytes[target_pos - ctx_bytes : target_pos],
    # with zero-padding for indices < 0.
    offsets = target_pos.unsqueeze(1) - ctx_bytes + torch.arange(
        ctx_bytes, device=device, dtype=torch.int64
    ).unsqueeze(0)  # (B, ctx_bytes)
    valid = offsets >= 0
    clamped = offsets.clamp(min=0)
    ctx = train_bytes[clamped]
    # Zero-out the padded positions.
    ctx = torch.where(valid, ctx, torch.zeros_like(ctx))
    targets = train_bytes[target_pos].to(torch.int64)
    return ctx, targets


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[eqprop_hopfield_v1] device={device}  M={M}  D_C={D_C}  "
        f"BETA={BETA}  BETA_NUDGE={BETA_NUDGE}  N_RELAX={N_RELAX}  "
        f"BATCH={BATCH_SIZE}  ETA={ETA}  SEED={seed}",
        flush=True,
    )

    t_start = time.monotonic()

    # Frozen RFF featurizer (use a separate seed deterministically).
    featurizer = RFFFeaturizer(device=device, seed=seed + 1)
    print(
        f"[eqprop_hopfield_v1] RFF R shape={tuple(featurizer.R.shape)}  "
        f"(~{featurizer.R.numel() * 4 / 1e6:.1f} MB)",
        flush=True,
    )

    # Initialize pattern matrix Xi (the ONLY learnable object).
    g_xi = torch.Generator(device=device)
    g_xi.manual_seed(seed + 2)
    Xi = torch.randn(
        STATE_DIM, M, generator=g_xi, device=device, dtype=torch.float32
    ) * XI_INIT_STD
    print(
        f"[eqprop_hopfield_v1] Xi shape={tuple(Xi.shape)}  "
        f"(~{Xi.numel() * 4 / 1e6:.1f} MB)  init_std={XI_INIT_STD}",
        flush=True,
    )

    # Training bytes.
    train_bytes = _make_train_bytes(train_text, device)
    print(f"[eqprop_hopfield_v1] train bytes: {train_bytes.numel():,}", flush=True)

    g_sample = torch.Generator(device=device)
    g_sample.manual_seed(seed + 3)

    step = 0
    last_log_t = t_start
    while step < N_STEPS_MAX:
        elapsed = time.monotonic() - t_start
        if elapsed > TRAIN_BUDGET_S:
            print(
                f"[eqprop_hopfield_v1] time budget hit at step {step} "
                f"({elapsed:.1f}s); stopping training",
                flush=True,
            )
            break

        ctx, targets = _sample_windows(
            train_bytes, BATCH_SIZE, CTX_BYTES, device, g_sample
        )
        phi = featurizer.features_batch(ctx)  # (B, D_C)
        eqprop_step(Xi, phi, targets, BETA, BETA_NUDGE, N_RELAX, ETA)

        if step % LOG_EVERY == 0 or step == N_STEPS_MAX - 1:
            now = time.monotonic()
            # Cheap train-batch loss / acc diagnostic: re-run the free phase
            # and compare its target block to the true targets.
            zeros_tgt = torch.zeros(
                BATCH_SIZE, TARGET_DIM, device=device, dtype=torch.float32
            )
            sigma_eval = torch.cat([phi, zeros_tgt], dim=-1)
            sigma_eval = hopfield_relax(
                sigma_eval, Xi, BETA, N_RELAX,
                clamp_context=phi, clamp_target=None,
            )
            logits = sigma_eval[:, D_C:]  # (B, 256)
            acc = (logits.argmax(dim=-1) == targets).float().mean().item()
            ce = F.cross_entropy(logits, targets).item()
            xi_norm = Xi.norm().item()
            print(
                f"[eqprop_hopfield_v1] step {step:5d}  "
                f"train_acc={acc:.4f}  train_ce={ce:.4f}  "
                f"||Xi||={xi_norm:.3f}  "
                f"elapsed={now - t_start:.1f}s  "
                f"step_dt={(now - last_log_t) / max(1, LOG_EVERY) * 1000:.1f}ms",
                flush=True,
            )
            last_log_t = now

        step += 1

    total_elapsed = time.monotonic() - t_start
    print(
        f"[eqprop_hopfield_v1] training done: {step} steps in "
        f"{total_elapsed:.1f}s ({total_elapsed / max(1, step) * 1000:.1f}ms/step)",
        flush=True,
    )

    return EqPropHopfieldCharModel(Xi=Xi, featurizer=featurizer, device=device)


# ---------------------------------------------------------------------------
# Streaming CharModel
# ---------------------------------------------------------------------------

class EqPropHopfieldCharModel(CharModel):
    """Streaming inference: rolling 256-byte buffer -> RFF context features ->
    2-step free-phase Hopfield relaxation -> read out target block ->
    softmax -> per-byte distribution.
    """

    def __init__(self, Xi: Tensor, featurizer: RFFFeaturizer, device: torch.device):
        self.Xi = Xi              # (STATE_DIM, M)
        self.featurizer = featurizer
        self.device = device
        # Rolling buffer of last CTX_BYTES bytes, zero-padded at start.
        self._buf = bytearray(CTX_BYTES)

    def reset(self) -> None:
        self._buf = bytearray(CTX_BYTES)

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        ctx = torch.frombuffer(bytes(self._buf), dtype=torch.uint8).to(self.device)
        phi = self.featurizer.features_single(ctx)  # (D_C,)
        phi_b = phi.unsqueeze(0)  # (1, D_C)
        zeros_tgt = torch.zeros(1, TARGET_DIM, device=self.device, dtype=torch.float32)
        sigma = torch.cat([phi_b, zeros_tgt], dim=-1)  # (1, STATE_DIM)
        sigma = hopfield_relax(
            sigma, self.Xi, BETA, N_RELAX,
            clamp_context=phi_b, clamp_target=None,
        )
        logits = sigma[0, D_C:]  # (256,)
        probs = F.softmax(logits.float(), dim=-1)
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
            if len(self._buf) > CTX_BYTES:
                del self._buf[0 : len(self._buf) - CTX_BYTES]
