"""P2-C — Width 1024 (FF investigation diagnostic).

Pass-2's submission with only two changes: WIDTH 384 -> 1024, and
N_STEPS 14000 -> 5000 (per-step cost scales with width^2; budget capped
at ~250 s). Everything else — rule, schedule, negatives, readout — is
identical to pass-2.

Diagnostic value: slope of val char-acc vs width informs Phase 7's
width prioritisation.

Spec: .survey/ff_runs/phase2/P2-C_width_1024/design.md
"""
from __future__ import annotations

__author__ = "@survey-ff-p2c"

import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wikitext import CharModel


VOCAB = 256
K = 24
INPUT_DIM = (K + 1) * VOCAB
N_LAYERS = 5
WIDTH = 1024                  # was 384
THETA = 2.0
BATCH = 256
N_STEPS = 5000                # was 14000 — width^2 cost trade
LR = 3e-4
BETAS = (0.9, 0.99)

HARD_NEG_EVERY = 250          # scaled from 500/14000 -> 250/5000
HARD_NEG_FRACTION = 0.5
HARD_NEG_TOPK = 5
HARD_NEG_REFIT_N = 20_000

RIDGE_N_FIT = 80_000
RIDGE_LAMBDA = 1.0
FEATURE_DIM = (N_LAYERS - 1) * WIDTH    # 4096 now

EVAL_BATCH = 256
RIDGE_FORWARD_BATCH = 512


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
        self.device = device
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


def build_input(context_bytes: Tensor, candidate_byte: Tensor | None) -> Tensor:
    B = context_bytes.size(0)
    ctx_oh = F.one_hot(context_bytes, VOCAB).float()
    if candidate_byte is None:
        cand_oh = torch.zeros(B, 1, VOCAB, device=context_bytes.device)
    else:
        cand_oh = F.one_hot(candidate_byte, VOCAB).float().unsqueeze(1)
    full = torch.cat([ctx_oh, cand_oh], dim=1)
    return full.reshape(B, INPUT_DIM)


@torch.no_grad()
def _extract_features(
    model: FFStack, train_bytes: Tensor, sample_idx: Tensor, batch_size: int,
) -> tuple[Tensor, Tensor]:
    n_fit = sample_idx.numel()
    feats = torch.empty(n_fit, FEATURE_DIM, device=train_bytes.device, dtype=torch.float32)
    targets = torch.empty(n_fit, device=train_bytes.device, dtype=torch.long)
    arange_k = torch.arange(K, device=train_bytes.device)
    model.eval()
    for start in range(0, n_fit, batch_size):
        end = min(start + batch_size, n_fit)
        idx = sample_idx[start:end]
        offsets = idx[:, None] + arange_k[None, :]
        ctx = train_bytes[offsets].long()
        tgt = train_bytes[idx + K].long()
        x = build_input(ctx, None)
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


def _train_ff(train_text: str, device: torch.device, seed: int) -> tuple[FFStack, Tensor]:
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < K + 1:
        raise ValueError(f"need at least {K+1} bytes; got {n}")

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
    print(f"[p2c] {n_params/1e6:.2f}M params  layers={N_LAYERS} width={WIDTH} K={K} "
          f"theta={THETA} bs={BATCH} steps={N_STEPS}")

    model.train()
    t0 = time.monotonic()
    log_every = 500
    max_start = n - K - 1
    hard_W: Tensor | None = None

    for step in range(N_STEPS):
        if step > 0 and step % HARD_NEG_EVERY == 0:
            t_re = time.monotonic()
            sample_idx = torch.randint(0, max_start, (HARD_NEG_REFIT_N,), device=device, generator=gen)
            phi, tgt = _extract_features(model, train_bytes, sample_idx, RIDGE_FORWARD_BATCH)
            hard_W = _solve_ridge(phi, tgt, RIDGE_LAMBDA)
            model.train()
            print(f"[p2c] step {step:5d}  hard-neg ridge refit ({HARD_NEG_REFIT_N} samples, "
                  f"{time.monotonic()-t_re:.2f}s)", flush=True)

        idx = torch.randint(0, max_start, (BATCH,), device=device, generator=gen)
        offsets = idx[:, None] + torch.arange(K + 1, device=device)[None, :]
        windows = train_bytes[offsets].long()
        ctx = windows[:, :K]
        true_byte = windows[:, K]

        if hard_W is not None:
            with torch.no_grad():
                phi_ctx = model.features(build_input(ctx, None))
                hard_logits = phi_ctx @ hard_W
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
            print(f"[p2c] step {step:5d}/{N_STEPS}  loss(sum) {total:.4f}  "
                  f"G_pos {g_pos:.3f}  G_neg {g_neg:.3f}  elapsed {elapsed:.0f}s",
                  flush=True)

    t_ridge = time.monotonic()
    sample_idx = torch.randint(0, max_start, (RIDGE_N_FIT,), device=device, generator=gen)
    phi, tgt = _extract_features(model, train_bytes, sample_idx, RIDGE_FORWARD_BATCH)
    W = _solve_ridge(phi, tgt, RIDGE_LAMBDA)
    print(f"[ridge] N_fit={RIDGE_N_FIT} D={FEATURE_DIM} lam={RIDGE_LAMBDA} "
          f"fit_s={time.monotonic()-t_ridge:.1f}", flush=True)

    with torch.no_grad():
        diag_n = min(20_000, RIDGE_N_FIT)
        pred = (phi[:diag_n] @ W).argmax(dim=-1)
        train_acc = (pred == tgt[:diag_n]).float().mean().item()
        print(f"[ridge] train_subset_acc={train_acc:.4f}  (n={diag_n})", flush=True)
    return model, W


class FFRidgeCharModel(CharModel):
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
        ctx_bytes = ([0] * pad + self._ctx) if pad > 0 else self._ctx[-K:]
        ctx_t = torch.tensor(ctx_bytes, dtype=torch.long, device=self.device).unsqueeze(0)
        return build_input(ctx_t, None)

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
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
        return out

    @torch.no_grad()
    def observe(self, char: str) -> None:
        for byte in char.encode("utf-8"):
            self._ctx.append(byte)
            if len(self._ctx) > K:
                self._ctx = self._ctx[-K:]


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[p2c] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, W = _train_ff(train_text, device, seed)
    return FFRidgeCharModel(model, W, device)
