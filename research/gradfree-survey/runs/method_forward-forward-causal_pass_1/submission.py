"""Causal Forward-Forward char-LM (pass 1).

Hinton's Forward-Forward rule (local per-layer goodness-vs-threshold loss)
applied to a 6-layer fully-connected stack scoring (K=24 context, candidate
next-byte) one-hot inputs. Goodness = sum of squared activations per layer.

No backprop across layer boundaries: each layer's input is .detach()-ed; a
per-layer Adam optimizer steps only that layer's weights from the local loss.
Layer 1 is a frozen random projection (per Hinton's recipe).

At eval, for each query position we batch 256 candidate next-byte hypotheses
through the stack as one forward pass and softmax their summed goodness
(layers 2-6) to produce the next-byte distribution.

Spec: .survey/designs/method_forward-forward-causal_pass_1.md
"""
from __future__ import annotations

__author__ = "@survey-ff"

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
INPUT_DIM = (K + 1) * VOCAB   # 6400 — concat of context one-hots and candidate
N_LAYERS = 6                  # layer 1 frozen, layers 2..6 trained
WIDTH = 512
THETA = 2.0
BATCH = 256
N_STEPS = 8000
LR = 3e-4
BETAS = (0.9, 0.99)


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
    """L2-normalize along the feature axis (strips magnitude)."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class FFStack(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: INPUT_DIM -> WIDTH. Layers 2..6: WIDTH -> WIDTH.
        layers = [FFLayer(INPUT_DIM, WIDTH)]
        for _ in range(N_LAYERS - 1):
            layers.append(FFLayer(WIDTH, WIDTH))
        self.layers = nn.ModuleList(layers)
        # Layer 1 frozen (random projection, per Hinton's recipe).
        for p in self.layers[0].parameters():
            p.requires_grad_(False)

    def forward_all(self, x: Tensor) -> list[Tensor]:
        """Run x through every layer with detach+L2-norm between layers.

        Returns per-layer post-ReLU activations. Each layer's input is
        the L2-normalized, detached activation of the previous layer —
        so a backward pass on any layer touches only that layer's
        weights (FF local-credit-assignment rule).
        """
        acts: list[Tensor] = []
        h = x  # x is already a leaf; no gradient should flow into it
        for layer in self.layers:
            # Detach the input so gradients of this layer's local loss
            # cannot cross the boundary into earlier layers.
            h_in = h.detach()
            a = layer(h_in)
            acts.append(a)
            # L2-normalize before feeding to next layer (strip magnitude).
            h = l2_normalize(a)
        return acts


# ---------------------------------------------------------------------------
# Negative sampler (unigram, rejection-sampled to exclude true byte)
# ---------------------------------------------------------------------------

class UnigramSampler:
    def __init__(self, byte_counts: Tensor, device: torch.device, generator: torch.Generator):
        # byte_counts: (256,) long tensor of training byte frequencies.
        probs = byte_counts.float() / byte_counts.float().sum()
        self.probs = probs.to(device)
        self.device = device
        self.generator = generator

    def sample(self, true_bytes: Tensor) -> Tensor:
        """Sample a negative byte for each entry in true_bytes (1-D long).

        Rejection-resamples positions that collide with the true byte.
        """
        n = true_bytes.numel()
        neg = torch.multinomial(self.probs, n, replacement=True, generator=self.generator)
        # Resample collisions until none remain (loop is bounded; with
        # 256 symbols the expected number of iterations is ~1.0).
        for _ in range(8):
            mask = neg == true_bytes
            if not mask.any():
                break
            resample = torch.multinomial(self.probs, int(mask.sum()), replacement=True, generator=self.generator)
            neg[mask] = resample
        # Any remaining collisions: fall back to (true_byte + 1) % 256.
        mask = neg == true_bytes
        if mask.any():
            neg[mask] = (true_bytes[mask] + 1) % VOCAB
        return neg


# ---------------------------------------------------------------------------
# One-hot input construction
# ---------------------------------------------------------------------------

def build_input(context_bytes: Tensor, candidate_byte: Tensor) -> Tensor:
    """Construct the (B, INPUT_DIM) one-hot input.

    context_bytes: (B, K) long
    candidate_byte: (B,) long
    Returns: (B, (K+1)*VOCAB) float
    """
    B = context_bytes.size(0)
    ctx_oh = F.one_hot(context_bytes, VOCAB).float()           # (B, K, 256)
    cand_oh = F.one_hot(candidate_byte, VOCAB).float().unsqueeze(1)  # (B, 1, 256)
    full = torch.cat([ctx_oh, cand_oh], dim=1)                  # (B, K+1, 256)
    return full.reshape(B, INPUT_DIM)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_ff(train_text: str, device: torch.device, seed: int) -> FFStack:
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < K + 1:
        raise ValueError(f"need at least {K+1} bytes; got {n}")

    # Unigram byte-frequency table over the training corpus.
    byte_counts = torch.bincount(train_bytes.long(), minlength=VOCAB).cpu()
    # Ensure no zero so multinomial doesn't see an all-zero column for
    # bytes that never appear (rare but possible for some control bytes).
    # Add a tiny floor of 1 to avoid degenerate sampling.
    byte_counts = byte_counts + 1

    # Deterministic RNG for minibatch indexing and negative sampling.
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    model = FFStack().to(device)

    # Per-layer Adam — one optimizer per trained layer (layers 2..6).
    optimizers = []
    for i in range(1, N_LAYERS):
        opt = torch.optim.Adam(
            model.layers[i].parameters(),
            lr=LR,
            betas=BETAS,
            weight_decay=0.0,
        )
        optimizers.append(opt)

    sampler = UnigramSampler(byte_counts, device, gen)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ff] {n_params/1e6:.2f}M params  layers={N_LAYERS} width={WIDTH} "
          f"K={K} theta={THETA} bs={BATCH} steps={N_STEPS}")

    model.train()
    t0 = time.monotonic()
    log_every = 500

    # Max valid starting index for a (K-context, target-byte) window.
    max_start = n - K - 1
    if max_start < 1:
        raise ValueError(f"corpus too short for K={K}: {n} bytes")

    for step in range(N_STEPS):
        # Sample BATCH starting positions; each yields a K-byte context
        # plus the true next byte at offset K.
        idx = torch.randint(0, max_start, (BATCH,), device=device, generator=gen)
        offsets = idx[:, None] + torch.arange(K + 1, device=device)[None, :]
        windows = train_bytes[offsets].long()              # (B, K+1)
        ctx = windows[:, :K]                               # (B, K)
        true_byte = windows[:, K]                          # (B,)
        neg_byte = sampler.sample(true_byte)               # (B,)

        x_pos = build_input(ctx, true_byte)                # (B, INPUT_DIM)
        x_neg = build_input(ctx, neg_byte)                 # (B, INPUT_DIM)

        # Stack into a single (2B, INPUT_DIM) for one fused forward.
        x = torch.cat([x_pos, x_neg], dim=0)
        acts = model.forward_all(x)

        # Per-layer local loss + step (skip layer 0 — frozen).
        total_diag = 0.0
        for li in range(1, N_LAYERS):
            a = acts[li]
            a_pos, a_neg = a[:BATCH], a[BATCH:]
            g_pos = (a_pos ** 2).sum(dim=-1)
            g_neg = (a_neg ** 2).sum(dim=-1)
            # FF loss (softplus form for numerical stability).
            loss_l = F.softplus(THETA - g_pos).mean() + F.softplus(g_neg - THETA).mean()

            opt = optimizers[li - 1]
            opt.zero_grad(set_to_none=True)
            loss_l.backward()
            opt.step()

            total_diag += loss_l.item()

        if step % log_every == 0 or step == N_STEPS - 1:
            elapsed = time.monotonic() - t0
            # Quick goodness gap diagnostic (last layer).
            with torch.no_grad():
                a_last = acts[-1]
                g_pos = (a_last[:BATCH] ** 2).sum(dim=-1).mean().item()
                g_neg = (a_last[BATCH:] ** 2).sum(dim=-1).mean().item()
            print(
                f"[ff] step {step:5d}/{N_STEPS}  "
                f"loss(sum) {total_diag:.4f}  "
                f"G_pos(L{N_LAYERS}) {g_pos:.3f}  "
                f"G_neg(L{N_LAYERS}) {g_neg:.3f}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    return model


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper
# ---------------------------------------------------------------------------

class FFCharModel(CharModel):
    """Streaming char model over UTF-8 bytes.

    `predict()` returns a distribution over bytes (each decodable byte
    becomes a 1-char str key). At each step we score all 256 candidate
    next-bytes by running the FF stack on a B=256 batch and softmaxing
    the sum of goodnesses across layers 2..6.

    `observe(char)` appends the char's bytes to the rolling context.
    """

    def __init__(self, model: FFStack, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        # Rolling K-byte context as a python list of ints (small, cheap).
        self._ctx: list[int] = []
        # Precompute candidate-byte one-hot (256, INPUT_DIM-K*256) — but
        # we build the full (256, INPUT_DIM) tensor lazily on each
        # predict() call because the context portion changes each step.
        self._cand_oh = F.one_hot(
            torch.arange(VOCAB, device=device), VOCAB,
        ).float()  # (256, 256)
        # Cached softmax of last predict() to convert into char-keyed dict.
        self._last_probs: Tensor | None = None

    @torch.no_grad()
    def reset(self) -> None:
        self._ctx = []
        self._last_probs = None

    @torch.no_grad()
    def _build_batch(self) -> Tensor:
        """(256, INPUT_DIM) — one row per candidate next-byte."""
        # Left-pad the context with zeros (byte 0) when shorter than K.
        pad = K - len(self._ctx)
        if pad > 0:
            ctx_bytes = [0] * pad + self._ctx
        else:
            ctx_bytes = self._ctx[-K:]
        ctx_t = torch.tensor(ctx_bytes, dtype=torch.long, device=self.device)
        ctx_oh = F.one_hot(ctx_t, VOCAB).float()          # (K, 256)
        ctx_flat = ctx_oh.reshape(-1)                     # (K*256,)
        # Broadcast: same context for every candidate.
        ctx_block = ctx_flat.unsqueeze(0).expand(VOCAB, -1)  # (256, K*256)
        full = torch.cat([ctx_block, self._cand_oh], dim=-1)  # (256, INPUT_DIM)
        return full

    @torch.no_grad()
    def predict(self) -> str:
        x = self._build_batch()
        acts = self.model.forward_all(x)
        # Score = sum of goodnesses over layers 2..6 (skip layer 1).
        score = torch.zeros(VOCAB, device=self.device)
        for li in range(1, N_LAYERS):
            score = score + (acts[li] ** 2).sum(dim=-1)
        probs = F.softmax(score, dim=-1)
        out: dict[str, float] = {}
        for byte_id, p in enumerate(probs.tolist()):
            try:
                ch = bytes([byte_id]).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out[ch] = p
        return max(out, key=lambda c: out[c]) if out else ""

    @torch.no_grad()
    def observe(self, char: str) -> None:
        for byte in char.encode("utf-8"):
            self._ctx.append(byte)
            if len(self._ctx) > K:
                # Trim — only the last K bytes matter.
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
    model = _train_ff(train_text, device, seed)
    return FFCharModel(model, device)
