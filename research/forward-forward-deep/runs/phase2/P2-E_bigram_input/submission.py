"""P2-E — Bigram-aware Input Encoding (FF investigation diagnostic).

Pass-2's submission with the input encoding swapped from K=24 char
one-hots to a denser bigram-aware encoding:
  - Last byte one-hot (256).
  - Last bigram hashed via sign-hash to 2048 dims.
  - Last 8 chars one-hot concat (2048).
  - Candidate-byte slot (256).
INPUT_DIM = 4608 (vs 6400 in pass-2).

Rule, schedule, width, depth, readout, hard-neg logic — all match
pass-2. Only the input encoding changes. Diagnostic value: does the FF
stack benefit from local n-gram structure being made explicit?

Spec: .survey/ff_runs/phase2/P2-E_bigram_input/design.md
"""
from __future__ import annotations

__author__ = "@survey-ff-p2e"

import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wikitext import CharModel


VOCAB = 256
RECENT_K = 8                              # one-hot recent window
BIGRAM_HASH_DIM = 2048
LAST_BYTE_DIM = VOCAB
CANDIDATE_DIM = VOCAB
INPUT_DIM = LAST_BYTE_DIM + BIGRAM_HASH_DIM + RECENT_K * VOCAB + CANDIDATE_DIM

# Context length the streaming model needs to remember = max of:
#   - bigram lookback (2)
#   - recent K chars (8)
# Use 8.
CTX_BUFFER = RECENT_K

N_LAYERS = 5
WIDTH = 384
THETA = 2.0
BATCH = 256
N_STEPS = 14000
LR = 3e-4
BETAS = (0.9, 0.99)

HARD_NEG_EVERY = 500
HARD_NEG_FRACTION = 0.5
HARD_NEG_TOPK = 5
HARD_NEG_REFIT_N = 20_000

RIDGE_N_FIT = 80_000
RIDGE_LAMBDA = 1.0
FEATURE_DIM = (N_LAYERS - 1) * WIDTH

EVAL_BATCH = 256
RIDGE_FORWARD_BATCH = 512


# ---------------------------------------------------------------------------
# Bigram sign-hash (deterministic from SEED).
# ---------------------------------------------------------------------------

def _build_bigram_hash(seed: int, device: torch.device) -> Tensor:
    """A (65536, BIGRAM_HASH_DIM) sign-hash matrix in {-1, +1}, sparse-ish:
    each bigram maps to exactly one hash-dim (the count-sketch part) with a
    sign in {-1, +1}. Returned as a dense (65536, hash_dim) tensor for
    simplicity (~128 MB for hash_dim=2048; fine on A100-80GB).

    The choice of count-sketch vs random projection: count-sketch keeps
    feature magnitudes bounded and is cheaper to apply (one scatter per
    bigram). Sign-hash mitigates collision bias.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed * 7919 + 1)
    # Bigram-id -> hash-index, in [0, BIGRAM_HASH_DIM).
    idx = torch.randint(0, BIGRAM_HASH_DIM, (65536,), generator=g)
    sign = torch.randint(0, 2, (65536,), generator=g).float() * 2.0 - 1.0  # {-1, +1}
    # Build (65536, hash_dim) one-hot * sign for fast lookup. Sparse layout
    # is not worth it at this size; dense is fine.
    H = torch.zeros(65536, BIGRAM_HASH_DIM)
    H[torch.arange(65536), idx] = sign
    return H.to(device)


# ---------------------------------------------------------------------------
# FF stack
# ---------------------------------------------------------------------------

class FFLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.lin(x))


def l2_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class FFStack(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [FFLayer(INPUT_DIM, WIDTH)]
        for _ in range(N_LAYERS - 1):
            layers.append(FFLayer(WIDTH, WIDTH))
        self.layers = nn.ModuleList(layers)
        for p in self.layers[0].parameters():
            p.requires_grad_(False)

    def forward_all(self, x: Tensor) -> list[Tensor]:
        acts: list[Tensor] = []
        h = x
        for layer in self.layers:
            a = layer(h.detach())
            acts.append(a)
            h = l2_normalize(a)
        return acts

    @torch.no_grad()
    def features(self, x: Tensor) -> Tensor:
        h = x
        feats = []
        for li, layer in enumerate(self.layers):
            a = layer(h)
            if li >= 1:
                feats.append(l2_normalize(a))
            h = l2_normalize(a)
        return torch.cat(feats, dim=-1)


class UnigramSampler:
    def __init__(self, byte_counts: Tensor, device: torch.device, generator: torch.Generator):
        probs = byte_counts.float() / byte_counts.float().sum()
        self.probs = probs.to(device)
        self.generator = generator

    def sample(self, true_bytes: Tensor) -> Tensor:
        n = true_bytes.numel()
        neg = torch.multinomial(self.probs, n, replacement=True, generator=self.generator)
        for _ in range(8):
            mask = neg == true_bytes
            if not mask.any():
                break
            neg[mask] = torch.multinomial(
                self.probs, int(mask.sum()), replacement=True, generator=self.generator,
            )
        mask = neg == true_bytes
        if mask.any():
            neg[mask] = (true_bytes[mask] + 1) % VOCAB
        return neg


# ---------------------------------------------------------------------------
# Input construction — bigram-aware
# ---------------------------------------------------------------------------

def build_input(
    recent_bytes: Tensor,        # (B, RECENT_K) — last RECENT_K bytes incl. last
    bigram_ids: Tensor,          # (B,) — last bigram id in [0, 65536)
    last_byte: Tensor,           # (B,) — last byte (= recent_bytes[:, -1])
    candidate_byte: Tensor | None,
    H: Tensor,                   # (65536, BIGRAM_HASH_DIM)
) -> Tensor:
    B = recent_bytes.size(0)
    device = recent_bytes.device
    last_oh = F.one_hot(last_byte, VOCAB).float()                   # (B, 256)
    bigram_feat = H[bigram_ids]                                      # (B, BIGRAM_HASH_DIM)
    recent_oh = F.one_hot(recent_bytes, VOCAB).float()              # (B, RECENT_K, 256)
    recent_flat = recent_oh.reshape(B, RECENT_K * VOCAB)
    if candidate_byte is None:
        cand = torch.zeros(B, CANDIDATE_DIM, device=device)
    else:
        cand = F.one_hot(candidate_byte, VOCAB).float()
    return torch.cat([last_oh, bigram_feat, recent_flat, cand], dim=-1)


def _windows_to_features(
    train_bytes: Tensor,
    idx: Tensor,
    H: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Given starting indices ``idx`` such that the next-byte target lives at
    ``idx + RECENT_K``, return (recent, bigram_id, last_byte).
    """
    arange = torch.arange(RECENT_K, device=train_bytes.device)
    offsets = idx[:, None] + arange[None, :]
    recent = train_bytes[offsets].long()                             # (B, RECENT_K)
    last_byte = recent[:, -1]
    second_last = recent[:, -2] if RECENT_K >= 2 else recent[:, -1]
    bigram_id = second_last * 256 + last_byte                        # (B,)
    return recent, bigram_id, last_byte


@torch.no_grad()
def _extract_features(
    model: FFStack,
    train_bytes: Tensor,
    sample_idx: Tensor,
    batch_size: int,
    H: Tensor,
) -> tuple[Tensor, Tensor]:
    n_fit = sample_idx.numel()
    feats = torch.empty(n_fit, FEATURE_DIM, device=train_bytes.device, dtype=torch.float32)
    targets = torch.empty(n_fit, device=train_bytes.device, dtype=torch.long)
    model.eval()
    for start in range(0, n_fit, batch_size):
        end = min(start + batch_size, n_fit)
        sub = sample_idx[start:end]
        recent, bigram_id, last_byte = _windows_to_features(train_bytes, sub, H)
        tgt = train_bytes[sub + RECENT_K].long()
        x = build_input(recent, bigram_id, last_byte, None, H)
        feats[start:end] = model.features(x)
        targets[start:end] = tgt
    return feats, targets


@torch.no_grad()
def _solve_ridge(phi: Tensor, targets: Tensor, lam: float) -> Tensor:
    D = phi.shape[1]
    device = phi.device
    dtype = torch.float32
    phi32 = phi.to(dtype)
    A = phi32.T @ phi32 + lam * torch.eye(D, device=device, dtype=dtype)
    B = torch.zeros(D, VOCAB, device=device, dtype=dtype)
    B.index_add_(1, targets, phi32.T)
    return torch.linalg.solve(A, B)


def _build_negatives(
    true_bytes: Tensor,
    unigram_sampler: UnigramSampler,
    hard_logits: Tensor | None,
    hard_fraction: float,
    topk: int,
    generator: torch.Generator,
) -> Tensor:
    B = true_bytes.numel()
    if hard_logits is None or hard_fraction <= 0.0:
        return unigram_sampler.sample(true_bytes)
    n_hard = int(round(B * hard_fraction))
    neg = unigram_sampler.sample(true_bytes)
    if n_hard == 0:
        return neg
    masked = hard_logits.clone()
    masked.scatter_(1, true_bytes.unsqueeze(1), float("-inf"))
    _, topk_idx = masked.topk(topk, dim=-1)
    pick = torch.randint(0, topk, (B,), device=true_bytes.device, generator=generator)
    hard_neg = topk_idx.gather(1, pick.unsqueeze(1)).squeeze(1)
    perm = torch.randperm(B, device=true_bytes.device, generator=generator)
    swap_pos = perm[:n_hard]
    neg[swap_pos] = hard_neg[swap_pos]
    mask = neg == true_bytes
    if mask.any():
        neg[mask] = (true_bytes[mask] + 1) % VOCAB
    return neg


def _train_ff(train_text: str, device: torch.device, seed: int) -> tuple[FFStack, Tensor, Tensor]:
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < RECENT_K + 1:
        raise ValueError(f"need at least {RECENT_K+1} bytes; got {n}")

    H = _build_bigram_hash(seed, device)
    print(f"[p2e] bigram hash: shape={tuple(H.shape)}  bytes={H.numel()*4//1024//1024} MiB")

    byte_counts = torch.bincount(train_bytes.long(), minlength=VOCAB).cpu() + 1
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    model = FFStack().to(device)
    optimizers = [
        torch.optim.Adam(model.layers[i].parameters(), lr=LR, betas=BETAS, weight_decay=0.0)
        for i in range(1, N_LAYERS)
    ]
    unigram = UnigramSampler(byte_counts, device, gen)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[p2e] {n_params/1e6:.2f}M params  layers={N_LAYERS} width={WIDTH} "
          f"INPUT_DIM={INPUT_DIM} steps={N_STEPS}")

    model.train()
    t0 = time.monotonic()
    log_every = 1000
    max_start = n - RECENT_K - 1
    hard_W: Tensor | None = None

    for step in range(N_STEPS):
        if step > 0 and step % HARD_NEG_EVERY == 0:
            t_re = time.monotonic()
            sample_idx = torch.randint(0, max_start, (HARD_NEG_REFIT_N,), device=device, generator=gen)
            phi, tgt = _extract_features(model, train_bytes, sample_idx, RIDGE_FORWARD_BATCH, H)
            hard_W = _solve_ridge(phi, tgt, RIDGE_LAMBDA)
            model.train()
            print(f"[p2e] step {step:5d}  hard-neg ridge refit ({HARD_NEG_REFIT_N} samples, "
                  f"{time.monotonic()-t_re:.2f}s)", flush=True)

        idx = torch.randint(0, max_start, (BATCH,), device=device, generator=gen)
        recent, bigram_id, last_byte = _windows_to_features(train_bytes, idx, H)
        true_byte = train_bytes[idx + RECENT_K].long()

        if hard_W is not None:
            with torch.no_grad():
                x_ctx = build_input(recent, bigram_id, last_byte, None, H)
                phi_ctx = model.features(x_ctx)
                hard_logits = phi_ctx @ hard_W
            model.train()
        else:
            hard_logits = None

        neg_byte = _build_negatives(
            true_byte, unigram, hard_logits, HARD_NEG_FRACTION, HARD_NEG_TOPK, gen,
        )

        x_pos = build_input(recent, bigram_id, last_byte, true_byte, H)
        x_neg = build_input(recent, bigram_id, last_byte, neg_byte, H)
        x = torch.cat([x_pos, x_neg], dim=0)
        acts = model.forward_all(x)

        total = 0.0
        for li in range(1, N_LAYERS):
            a = acts[li]
            g_pos = (a[:BATCH] ** 2).sum(dim=-1)
            g_neg = (a[BATCH:] ** 2).sum(dim=-1)
            loss = F.softplus(THETA - g_pos).mean() + F.softplus(g_neg - THETA).mean()
            opt = optimizers[li - 1]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()

        if step % log_every == 0 or step == N_STEPS - 1:
            elapsed = time.monotonic() - t0
            with torch.no_grad():
                a_last = acts[-1]
                g_pos = (a_last[:BATCH] ** 2).sum(dim=-1).mean().item()
                g_neg = (a_last[BATCH:] ** 2).sum(dim=-1).mean().item()
            print(f"[p2e] step {step:5d}/{N_STEPS}  loss(sum) {total:.4f}  "
                  f"G_pos {g_pos:.3f}  G_neg {g_neg:.3f}  elapsed {elapsed:.0f}s",
                  flush=True)

    t_ridge = time.monotonic()
    sample_idx = torch.randint(0, max_start, (RIDGE_N_FIT,), device=device, generator=gen)
    phi, tgt = _extract_features(model, train_bytes, sample_idx, RIDGE_FORWARD_BATCH, H)
    W = _solve_ridge(phi, tgt, RIDGE_LAMBDA)
    print(f"[ridge] N_fit={RIDGE_N_FIT} D={FEATURE_DIM} lam={RIDGE_LAMBDA} "
          f"fit_s={time.monotonic()-t_ridge:.1f}", flush=True)

    with torch.no_grad():
        diag_n = min(20_000, RIDGE_N_FIT)
        pred = (phi[:diag_n] @ W).argmax(dim=-1)
        train_acc = (pred == tgt[:diag_n]).float().mean().item()
        print(f"[ridge] train_subset_acc={train_acc:.4f}  (n={diag_n})", flush=True)
    return model, W, H


class BigramFFCharModel(CharModel):
    def __init__(self, model: FFStack, W: Tensor, H: Tensor, device: torch.device):
        self.model = model
        self.W = W
        self.H = H
        self.device = device
        self.model.eval()
        self._ctx: list[int] = []

    @torch.no_grad()
    def reset(self) -> None:
        self._ctx = []

    @torch.no_grad()
    def _build_one(self) -> Tensor:
        pad = RECENT_K - len(self._ctx)
        if pad > 0:
            recent_bytes = [0] * pad + self._ctx
        else:
            recent_bytes = self._ctx[-RECENT_K:]
        recent = torch.tensor(recent_bytes, dtype=torch.long, device=self.device).unsqueeze(0)
        last_byte = recent[:, -1]
        second_last = recent[:, -2] if RECENT_K >= 2 else recent[:, -1]
        bigram_id = second_last * 256 + last_byte
        return build_input(recent, bigram_id, last_byte, None, self.H)

    @torch.no_grad()
    def predict(self) -> str:
        x = self._build_one()
        phi = self.model.features(x)
        logits = phi @ self.W
        probs = F.softmax(logits.squeeze(0), dim=-1)
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
            if len(self._ctx) > RECENT_K:
                self._ctx = self._ctx[-RECENT_K:]


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[p2e] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, W, H = _train_ff(train_text, device, seed)
    return BigramFFCharModel(model, W, H, device)
