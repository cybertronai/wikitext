"""Causal Forward-Forward + Cascaded Ridge Readout (pass 2).

Train a narrower (5x384) FF stack with Hinton's local goodness rule,
then *discard* the goodness-as-likelihood predictor. Instead, fit a
closed-form ridge regression from concat-of-LayerNorm'd layer
activations (layers 2..5) to next-byte one-hot.

The eval-time win: ONE forward per context (zero-candidate slot) then a
linear matvec, instead of 256 forwards per char. Frees budget for a
longer FF training schedule and per-char streaming with batched
windows.

No backprop across layers (FF rule: local per-layer Adam on a detached
input). Ridge is closed-form on GPU — no gradient at all.

Spec: .survey/designs/method_forward-forward-causal_pass_2.md
"""
from __future__ import annotations

__author__ = "@survey-ff-p2"

import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Hyperparameters (from spec — DO NOT modify)
# ---------------------------------------------------------------------------

VOCAB = 256
K = 24                        # context window
INPUT_DIM = (K + 1) * VOCAB   # 6400 — (K context + 1 candidate slot) one-hots
N_LAYERS = 5                  # layer 1 frozen, layers 2..5 trained by FF
WIDTH = 384
THETA = 2.0
BATCH = 256
N_STEPS = 14000
LR = 3e-4
BETAS = (0.9, 0.99)

# Hard-negative refresh schedule (spec §3.1)
HARD_NEG_EVERY = 500
HARD_NEG_FRACTION = 0.5
HARD_NEG_TOPK = 5
# Ridge readout used for hard-negative sampling is re-fit on a smaller cache.
HARD_NEG_REFIT_N = 20_000

# Ridge fit (spec §3.2)
RIDGE_N_FIT = 80_000
RIDGE_LAMBDA = 1.0
# Feature concat = LN(a_2..a_5) -> 4 layers * 384 = 1536.
FEATURE_DIM = (N_LAYERS - 1) * WIDTH

# Eval batching (spec §5)
EVAL_BATCH = 256

# Forward extraction batch when building Phi (spec §5)
RIDGE_FORWARD_BATCH = 512


# ---------------------------------------------------------------------------
# FF stack
# ---------------------------------------------------------------------------

class FFLayer(nn.Module):
    """Linear (no bias) + ReLU. Standard FF unit."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.lin(x))


def l2_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """L2-normalize along the feature axis (strip magnitude)."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class FFStack(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [FFLayer(INPUT_DIM, WIDTH)]
        for _ in range(N_LAYERS - 1):
            layers.append(FFLayer(WIDTH, WIDTH))
        self.layers = nn.ModuleList(layers)
        # Layer 1 is a frozen random projection (Hinton's recipe).
        for p in self.layers[0].parameters():
            p.requires_grad_(False)

    def forward_all(self, x: Tensor) -> list[Tensor]:
        """Run x through every layer, .detach()-ing between layers.

        A backward on any layer's local loss touches only that layer's
        weights — FF local-credit-assignment rule.
        """
        acts: list[Tensor] = []
        h = x
        for layer in self.layers:
            h_in = h.detach()
            a = layer(h_in)
            acts.append(a)
            h = l2_normalize(a)
        return acts

    @torch.no_grad()
    def features(self, x: Tensor) -> Tensor:
        """Concat-of-LN(a_2..a_N) -> (B, FEATURE_DIM). Used for the
        ridge readout. Inference-only (no grads needed).
        """
        h = x
        feats = []
        for li, layer in enumerate(self.layers):
            a = layer(h)
            if li >= 1:  # skip layer 1 (frozen random) — spec §2
                feats.append(l2_normalize(a))
            h = l2_normalize(a)
        return torch.cat(feats, dim=-1)


# ---------------------------------------------------------------------------
# Negative samplers
# ---------------------------------------------------------------------------

class UnigramSampler:
    """Rejection-sampled unigram next-byte sampler that excludes the true byte."""

    def __init__(self, byte_counts: Tensor, device: torch.device, generator: torch.Generator):
        probs = byte_counts.float() / byte_counts.float().sum()
        self.probs = probs.to(device)
        self.device = device
        self.generator = generator

    def sample(self, true_bytes: Tensor) -> Tensor:
        n = true_bytes.numel()
        neg = torch.multinomial(self.probs, n, replacement=True, generator=self.generator)
        for _ in range(8):
            mask = neg == true_bytes
            if not mask.any():
                break
            resample = torch.multinomial(
                self.probs, int(mask.sum()), replacement=True, generator=self.generator,
            )
            neg[mask] = resample
        mask = neg == true_bytes
        if mask.any():
            neg[mask] = (true_bytes[mask] + 1) % VOCAB
        return neg


# ---------------------------------------------------------------------------
# Input construction (one-hot)
# ---------------------------------------------------------------------------

def build_input(context_bytes: Tensor, candidate_byte: Tensor | None) -> Tensor:
    """(B, INPUT_DIM) one-hot of (K context, 1 candidate slot).

    If ``candidate_byte`` is None, the candidate slot is all zeros — used
    at ridge feature-extraction and eval time per spec §2.
    """
    B = context_bytes.size(0)
    ctx_oh = F.one_hot(context_bytes, VOCAB).float()           # (B, K, 256)
    if candidate_byte is None:
        cand_oh = torch.zeros(B, 1, VOCAB, device=context_bytes.device)
    else:
        cand_oh = F.one_hot(candidate_byte, VOCAB).float().unsqueeze(1)
    full = torch.cat([ctx_oh, cand_oh], dim=1)                  # (B, K+1, 256)
    return full.reshape(B, INPUT_DIM)


# ---------------------------------------------------------------------------
# Ridge fit (closed-form normal equations on GPU)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_features(
    model: FFStack,
    train_bytes: Tensor,
    sample_idx: Tensor,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """For each starting index in ``sample_idx``, build a zero-candidate
    input from the K-byte context ending right before the target byte, and
    extract the FF feature vector. Returns (Phi, true_byte).
    """
    n_fit = sample_idx.numel()
    feats = torch.empty(n_fit, FEATURE_DIM, device=train_bytes.device, dtype=torch.float32)
    targets = torch.empty(n_fit, device=train_bytes.device, dtype=torch.long)
    arange_k = torch.arange(K, device=train_bytes.device)
    model.eval()
    for start in range(0, n_fit, batch_size):
        end = min(start + batch_size, n_fit)
        idx = sample_idx[start:end]
        offsets = idx[:, None] + arange_k[None, :]
        ctx = train_bytes[offsets].long()                        # (b, K)
        tgt = train_bytes[idx + K].long()                        # (b,)
        x = build_input(ctx, None)                                # (b, INPUT_DIM)
        feats[start:end] = model.features(x)
        targets[start:end] = tgt
    return feats, targets


@torch.no_grad()
def _solve_ridge(phi: Tensor, targets: Tensor, lam: float) -> Tensor:
    """W = (Phi^T Phi + lam I)^-1 Phi^T Y, with Y the one-hot target matrix.

    Computed without materialising the full (N,256) Y: Phi^T Y is a class-
    indexed scatter-sum over rows of Phi. Returns W of shape (D, 256).
    """
    D = phi.shape[1]
    device = phi.device
    dtype = torch.float32
    phi32 = phi.to(dtype)
    A = phi32.T @ phi32                                          # (D, D)
    A = A + lam * torch.eye(D, device=device, dtype=dtype)
    # B = Phi^T Y : (D, 256). For one-hot Y, this is a per-class sum of
    # the corresponding rows of Phi.
    B = torch.zeros(D, VOCAB, device=device, dtype=dtype)
    B.index_add_(1, targets, phi32.T)
    W = torch.linalg.solve(A, B)
    return W


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_negatives(
    true_bytes: Tensor,
    unigram_sampler: UnigramSampler,
    hard_logits: Tensor | None,
    hard_fraction: float,
    topk: int,
    generator: torch.Generator,
) -> Tensor:
    """Negatives = `hard_fraction` from ridge top-K (excluding true byte)
    + remainder from unigram. ``hard_logits`` is (B, 256) or None.
    """
    B = true_bytes.numel()
    if hard_logits is None or hard_fraction <= 0.0:
        return unigram_sampler.sample(true_bytes)

    n_hard = int(round(B * hard_fraction))
    neg = unigram_sampler.sample(true_bytes)
    if n_hard == 0:
        return neg

    # Mask out the true byte before top-K.
    masked = hard_logits.clone()
    masked.scatter_(1, true_bytes.unsqueeze(1), float("-inf"))
    topk_vals, topk_idx = masked.topk(topk, dim=-1)              # (B, topk)
    # Sample one of the top-K uniformly per row.
    pick = torch.randint(0, topk, (B,), device=true_bytes.device, generator=generator)
    hard_neg = topk_idx.gather(1, pick.unsqueeze(1)).squeeze(1)  # (B,)
    # Overwrite the first n_hard slots (random permutation for unbias).
    perm = torch.randperm(B, device=true_bytes.device, generator=generator)
    swap_pos = perm[:n_hard]
    neg[swap_pos] = hard_neg[swap_pos]
    # Final safety: any remaining collisions -> bump.
    mask = neg == true_bytes
    if mask.any():
        neg[mask] = (true_bytes[mask] + 1) % VOCAB
    return neg


def _train_ff(train_text: str, device: torch.device, seed: int) -> tuple[FFStack, Tensor]:
    """Train the FF backbone, then fit + return the ridge readout W.

    Returns (model, W) where W has shape (FEATURE_DIM, VOCAB).
    """
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < K + 1:
        raise ValueError(f"need at least {K+1} bytes; got {n}")

    byte_counts = torch.bincount(train_bytes.long(), minlength=VOCAB).cpu()
    byte_counts = byte_counts + 1  # avoid all-zero rows

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    model = FFStack().to(device)

    # Per-layer Adam — one optimiser per trained layer (layers 2..N).
    optimizers = []
    for i in range(1, N_LAYERS):
        opt = torch.optim.Adam(
            model.layers[i].parameters(),
            lr=LR,
            betas=BETAS,
            weight_decay=0.0,
        )
        optimizers.append(opt)

    unigram = UnigramSampler(byte_counts, device, gen)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ff] {n_params/1e6:.2f}M params  layers={N_LAYERS} width={WIDTH} "
          f"K={K} theta={THETA} bs={BATCH} steps={N_STEPS}")

    model.train()
    t0 = time.monotonic()
    log_every = 1000

    max_start = n - K - 1
    if max_start < 1:
        raise ValueError(f"corpus too short for K={K}: {n} bytes")

    # Hard-negative ridge readout (built on the fly).
    hard_W: Tensor | None = None

    for step in range(N_STEPS):
        # Optional periodic refit of the ridge-readout used for hard negatives.
        if step > 0 and step % HARD_NEG_EVERY == 0:
            t_re = time.monotonic()
            sample_idx = torch.randint(
                0, max_start, (HARD_NEG_REFIT_N,), device=device, generator=gen,
            )
            phi, tgt = _extract_features(
                model, train_bytes, sample_idx, RIDGE_FORWARD_BATCH,
            )
            hard_W = _solve_ridge(phi, tgt, RIDGE_LAMBDA)
            model.train()
            re_dt = time.monotonic() - t_re
            print(f"[ff] step {step:5d}  hard-neg ridge refit "
                  f"({HARD_NEG_REFIT_N} samples, {re_dt:.2f}s)", flush=True)

        # Sample a minibatch.
        idx = torch.randint(0, max_start, (BATCH,), device=device, generator=gen)
        offsets = idx[:, None] + torch.arange(K + 1, device=device)[None, :]
        windows = train_bytes[offsets].long()
        ctx = windows[:, :K]
        true_byte = windows[:, K]

        # Score the context with the *current* hard-neg ridge readout, if any,
        # to seed the hard-negative pool.
        if hard_W is not None:
            with torch.no_grad():
                phi_ctx = model.features(build_input(ctx, None))     # (B, FEATURE_DIM)
                hard_logits = phi_ctx @ hard_W                       # (B, 256)
            model.train()
        else:
            hard_logits = None

        neg_byte = _build_negatives(
            true_byte, unigram, hard_logits, HARD_NEG_FRACTION, HARD_NEG_TOPK, gen,
        )

        x_pos = build_input(ctx, true_byte)
        x_neg = build_input(ctx, neg_byte)
        x = torch.cat([x_pos, x_neg], dim=0)
        acts = model.forward_all(x)

        total_diag = 0.0
        for li in range(1, N_LAYERS):
            a = acts[li]
            a_pos, a_neg = a[:BATCH], a[BATCH:]
            g_pos = (a_pos ** 2).sum(dim=-1)
            g_neg = (a_neg ** 2).sum(dim=-1)
            loss_l = F.softplus(THETA - g_pos).mean() + F.softplus(g_neg - THETA).mean()

            opt = optimizers[li - 1]
            opt.zero_grad(set_to_none=True)
            loss_l.backward()
            opt.step()
            total_diag += loss_l.item()

        if step % log_every == 0 or step == N_STEPS - 1:
            elapsed = time.monotonic() - t0
            with torch.no_grad():
                a_last = acts[-1]
                g_pos = (a_last[:BATCH] ** 2).sum(dim=-1).mean().item()
                g_neg = (a_last[BATCH:] ** 2).sum(dim=-1).mean().item()
            print(
                f"[ff] step {step:5d}/{N_STEPS}  "
                f"loss(sum) {total_diag:.4f}  "
                f"G_pos {g_pos:.3f}  G_neg {g_neg:.3f}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    # ----- Final ridge fit on N_fit samples -----
    t_ridge = time.monotonic()
    sample_idx = torch.randint(
        0, max_start, (RIDGE_N_FIT,), device=device, generator=gen,
    )
    phi, tgt = _extract_features(model, train_bytes, sample_idx, RIDGE_FORWARD_BATCH)
    W = _solve_ridge(phi, tgt, RIDGE_LAMBDA)
    print(f"[ridge] N_fit={RIDGE_N_FIT}  D={FEATURE_DIM}  lam={RIDGE_LAMBDA}  "
          f"W.shape={tuple(W.shape)}  fit_s={time.monotonic()-t_ridge:.1f}",
          flush=True)
    # Quick training-set diagnostic: how well does W classify Phi?
    with torch.no_grad():
        # Use a small subset to avoid materialising the full (N_fit, 256) matrix.
        diag_n = min(20_000, RIDGE_N_FIT)
        pred = (phi[:diag_n] @ W).argmax(dim=-1)
        train_acc = (pred == tgt[:diag_n]).float().mean().item()
        print(f"[ridge] train_subset_acc={train_acc:.4f}  (n={diag_n})", flush=True)

    return model, W


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper — eval is one forward per char + matvec
# ---------------------------------------------------------------------------

class FFRidgeCharModel(CharModel):
    """Streaming char model with FF features + ridge readout.

    Each predict() call:
      1. Build (1, INPUT_DIM) zero-candidate input from current rolling
         K-byte context.
      2. Forward through FF -> phi (1, 1536).
      3. logits = phi @ W -> softmax over 256 bytes.

    Single forward per char (vs 256 in pass 1).
    """

    def __init__(self, model: FFStack, W: Tensor, device: torch.device):
        self.model = model
        self.W = W
        self.device = device
        self.model.eval()
        self._ctx: list[int] = []

    @torch.no_grad()
    def reset(self) -> None:
        self._ctx = []

    @torch.no_grad()
    def _build_one(self) -> Tensor:
        pad = K - len(self._ctx)
        if pad > 0:
            ctx_bytes = [0] * pad + self._ctx
        else:
            ctx_bytes = self._ctx[-K:]
        ctx_t = torch.tensor(ctx_bytes, dtype=torch.long, device=self.device).unsqueeze(0)
        return build_input(ctx_t, None)                              # (1, INPUT_DIM)

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        x = self._build_one()
        phi = self.model.features(x)                                  # (1, FEATURE_DIM)
        logits = phi @ self.W                                          # (1, 256)
        probs = F.softmax(logits.squeeze(0), dim=-1)
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
        for byte in char.encode("utf-8"):
            self._ctx.append(byte)
            if len(self._ctx) > K:
                self._ctx = self._ctx[-K:]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[ff] SEED={seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, W = _train_ff(train_text, device, seed)
    return FFRidgeCharModel(model, W, device)
