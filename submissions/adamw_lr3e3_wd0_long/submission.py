"""adamw_lr3e3_wd0_long — REOPEN of W4 with proper LR.

Direct fork of submissions/nanogpt_small (the E2 working config), with ONLY
the optimizer swapped: AdamW for ALL parameters at lr=1e-3, wd=0.05.

The original W4 (adamw_only) used block_lr=3e-4 — ~10× too low for d=256 +
bf16 — and DQ'd at acc=0.6038 with loss=1.39 oscillating at step 1499 (i.e.
undertrained, not "AdamW can't reach 0.70"). Karpathy nanoGPT uses 6e-4 to
3e-3 at this size; Chinchilla scaling + bf16 → lr ∈ {1e-3, 2e-3, 3e-3} with
wd ∈ {0.0, 0.05}.

This run = run 1 of the 3-run adaptive budget (iterative-research
SKILL.md). Baseline E2 (nanogpt_small) hits 14,882 J / 0.7094 with
Muon+AdamW at the SAME arch. If AdamW-only reaches ≥0.70 here, the
implication is huge: AdamW is ~1.4× cheaper per step than Muon → 1.4× J
savings across every NN-bearing submission.

Hypothesis grid:
  Run 1: lr=1e-3, wd=0.05 (this submission)
  Run 2: tune based on run 1 trajectory (lr=2e-3 if undertrained, wd↓ if good)
  Run 3: tune based on run 2

Arch is identical to nanogpt_small E2: d=256, L=4, n_steps=1500, bs=32,
T=1024, head_dim=64, ReLU^2 MLP, RoPE base=1024, half-truncate, QK RMSNorm,
soft-cap logits, stable-then-decay (cooldown_frac=0.7).
"""
from __future__ import annotations

__author__ = "@gabrielnan"

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Architecture (verbatim from nanogpt_small E2)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer(
            "angular_freq",
            torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]),
        )

    def forward(self, x_BTHD: Tensor, offset: int = 0) -> Tensor:
        T = x_BTHD.size(1)
        pos = torch.arange(T, dtype=torch.float32, device=x_BTHD.device) + offset
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 64):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = self.rotary(q, offset=offset)
        k = self.rotary(k, offset=offset)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        is_causal = (kv_cache is None) and T > 1
        y = F.scaled_dot_product_attention(q, k, v, scale=0.12, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y), (k, v)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim=head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h, new_kv = self.attn(self.norm1(x), kv_cache, offset=offset)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        head_dim: int = 64,
        max_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(
            [Block(model_dim, head_dim=head_dim) for _ in range(num_layers)]
        )
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(
        self,
        inputs: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        x = self.norm1(self.embed(inputs))
        new_caches: list[tuple[Tensor, Tensor]] = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv, offset=offset)
            new_caches.append(new_kv)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits, new_caches


# ---------------------------------------------------------------------------
# Init scheme (mirrors modded-nanogpt simple)
# ---------------------------------------------------------------------------

def _init_modded(model: GPT) -> None:
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1) ** 0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise RuntimeError(f"Uninitialized parameter: {name}")


# ---------------------------------------------------------------------------
# Training (AdamW for ALL parameters, proper LR)
# ---------------------------------------------------------------------------

class TrainConfig:
    # E2 baseline arch (verbatim). Optimizer-only delta vs nanogpt_small:
    # block_lr = 1e-3 (was Muon lr=0.035) and block_wd = 0.05 (was Muon wd=0.025).
    # 1e-3 is the canonical Karpathy nanoGPT default for d=256 + bf16 + bs=32.
    def __init__(
        self,
        model_dim=256,
        num_layers=4,
        head_dim=64,
        max_len=1024,
        batch_size=32,
        n_steps=4500,
        cooldown_frac=0.7,
        embed_lr=0.3,
        head_lr=1.0 / 320,
        scalar_lr=0.01,
        block_lr=3e-3,
        block_wd=0.0,
        log_every=100,
    ):
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.cooldown_frac = cooldown_frac
        self.embed_lr = embed_lr
        self.head_lr = head_lr
        self.scalar_lr = scalar_lr
        self.block_lr = block_lr
        self.block_wd = block_wd
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"H={self.model_dim//self.head_dim} bs={self.batch_size} "
                f"T={self.max_len} steps={self.n_steps} "
                f"block_lr={self.block_lr} block_wd={self.block_wd})")


def _train_adamw(
    text: str,
    cfg: TrainConfig,
    device: torch.device,
) -> GPT:
    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < cfg.max_len + 1:
        raise ValueError(f"need at least {cfg.max_len+1} bytes; got {n}")

    model = GPT(
        vocab_size=256,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        head_dim=cfg.head_dim,
        max_len=cfg.max_len,
    ).to(device)
    _init_modded(model)

    # AdamW for ALL parameters. Keep the same group split as nanogpt_small:
    # embed/proj/scalars at their special LRs (so the embed/proj/scalar parts
    # aren't disturbed vs E2), but replace the Muon group on 2D block weights
    # with a standard AdamW group at lr=1e-3, weight_decay=0.05.
    # betas=(0.9, 0.95) — canonical for transformer LM (β2=0.95 standard
    # nanoGPT setting).
    block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2]
    scalars = [p for p in model.parameters() if p.ndim < 2]
    optimizer = AdamW(
        [
            dict(params=[model.embed.weight], lr=cfg.embed_lr, weight_decay=0.0),
            dict(params=[model.proj.weight], lr=cfg.head_lr, weight_decay=0.0),
            dict(params=scalars, lr=cfg.scalar_lr, weight_decay=0.0),
            dict(params=block_2d, lr=cfg.block_lr, weight_decay=cfg.block_wd),
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=(device.type == "cuda"),
    )
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[adamw_lr3e3_wd0_long] {n_params/1e6:.2f}M params  cfg={cfg}")

    def set_lr(step: int) -> None:
        progress = step / cfg.n_steps
        if progress < 1 - cfg.cooldown_frac:
            eta = 1.0
        else:
            eta = max(0.0, (1 - progress) / cfg.cooldown_frac)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * eta

    model.train()
    use_amp = device.type == "cuda"
    t0 = time.monotonic()
    for step in range(cfg.n_steps):
        set_lr(step)
        idx = torch.randint(0, n - cfg.max_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(cfg.max_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        optimizer.step()

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            print(
                f"[adamw_lr3e3_wd0_long] step {step:5d}/{cfg.n_steps}  "
                f"loss {loss.item():.4f}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    return model


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper (KV-cached, RoPE-offset-aware)
# ---------------------------------------------------------------------------

class AdamWCharModel(CharModel):
    def __init__(self, model: GPT, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self._kv: list[tuple[Tensor, Tensor]] | None = None
        self._next_logits: Tensor | None = None
        self._pos: int = 0

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = None
        self._pos = 0
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        logits, self._kv = self.model(x, None, offset=self._pos)
        self._next_logits = logits[0, -1]
        self._pos = 1

    @torch.no_grad()
    def predict(self) -> str:
        if self._next_logits is None:
            raise RuntimeError("predict() called before reset()")
        probs = F.softmax(self._next_logits.float(), dim=-1)
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
        if self._kv is None:
            raise RuntimeError("observe() called before reset()")
        for byte in char.encode("utf-8"):
            self._maybe_trim_cache()
            x = torch.tensor([[byte]], dtype=torch.long, device=self.device)
            logits, self._kv = self.model(x, self._kv, offset=self._pos)
            self._next_logits = logits[0, -1]
            self._pos += 1

    def _maybe_trim_cache(self) -> None:
        if self._kv is None:
            return
        cur = self._kv[0][0].shape[2]
        if cur < self.model.max_len:
            return
        keep = self.model.max_len - 1
        self._kv = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in self._kv]


class _EmptyCharModel(CharModel):
    def reset(self) -> None:
        pass

    def predict(self) -> str:
        return " "

    def observe(self, char: str) -> None:
        pass


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    if os.environ.get("SMOKE_TEST_ONLY") == "1":
        print("[adamw_lr3e3_wd0_long] SMOKE_TEST_ONLY=1 — returning EmptyCharModel "
              "without training.")
        return _EmptyCharModel()

    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[adamw_lr3e3_wd0_long] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run 4 of the adaptive budget (extension granted: lr=3e-3 wd=0.0 trajectory
    # was substantially better than lr=1e-3, 2e-3 — see SHARED.md). Same winning
    # recipe (lr=3e-3, wd=0.0) but **n_steps=4500 instead of 1500**: 3x more
    # training. lr=3e-3 wd=0.0 at 1500 steps reached loss 1.16 at step 1499 still
    # descending — not plateaued, so more steps should help.
    # Total wall-clock estimated: 1500 steps = 60s, so 4500 ≈ 180s, leaving
    # 120s for eval. Within 300s cap.
    block_lr = float(os.environ.get("BLOCK_LR", "3e-3"))
    block_wd = float(os.environ.get("BLOCK_WD", "0.0"))
    cfg = TrainConfig(block_lr=block_lr, block_wd=block_wd)
    model = _train_adamw(train_text, cfg, device)
    return AdamWCharModel(model)
