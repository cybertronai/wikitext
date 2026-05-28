"""DiffusionBlocks AR for byte-level WikiText-103 — v4, σ_max=5.

v3 diagnostic: per-block CE plateaued near identical levels to v2
(b0=0.23, b1=0.58, b2=1.80) and val acc dropped to 0.221. The σ-invariant
CE head normalization wasn't the bottleneck. The remaining hypothesis is
that the high-σ block trains on out-of-distribution σ at inference time:
LogNormal(P_mean=-1.2, P_std=1.2) puts <5% mass above σ=5, yet the v2/v3
schedule started Euler at σ_max=80. The first denoising step is then
extrapolating wildly, propagating noise through the rest of the chain
regardless of how well the lower-σ blocks were trained.

This variant only changes σ_max=5, keeping the rest identical (including
the v3 normalised CE head and the v2 logit_scale). If val acc moves
materially, the schedule was the problem; if not, the cross-attention
architecture itself is the ceiling.

Earlier σ-invariant CE head context:

Diagnostic from v2 + N=50 parallel run: val acc plateaued at 0.236
regardless of inference resolution. The training-time CE head had a
σ-dependent magnitude bug: pred_y has norm ≈ 1 at σ→0 (where pred_y≈z
= unit-norm y_emb) but ≈ σ_data ≈ 0.05 at σ→∞ (where pred_y ≈ c_out·out
= σ_data·out). With logit_scale=√d=20, low-σ logits sit in [-20, 20]
and the softmax sharpens fine, but high-σ logits collapse back to
[-1, 1] and CE saturates near 3.58 over 256 classes. The high-σ block
therefore never learns to discriminate, capping the whole denoising
chain.

Fix: L2-normalise pred_y before projecting onto E_out. With both
operands unit-norm and logit_scale ≈ √d, the effective logit range is
σ-independent. The L2 denoising loss still operates on un-normalised
pred_y so the probability-flow ODE interpretation is preserved.

Per-block, single-block-gradient training (Shing, Koyama, Akiba — ICLR 2026,
arxiv:2506.14202). 6-layer/384-dim transformer partitioned into B=3 blocks
of 2 layers each. Each block is a stack of cross-attention layers:

  - K, V are projected from the prefix token embedding x_emb (independent
    of z, so the KV cache is stable across Euler steps and never has to
    be rebuilt).
  - Q is projected from the noised target embedding z; cross-attention
    is causally masked at training (Q at position t attends to K/V at
    positions ≤ t).
  - AdaLN(c_noise) modulates each RMSNorm so the block is conditioned on
    σ. c_noise = 0.25 * log σ (EDM convention).

At each training step a single block b is sampled and a single σ is drawn
from its equi-probability noise range. Only block b's forward + backward
runs; the other blocks are not touched. The shared embed / E_out / cond
embedder do step on every iteration since they sit on the gradient path
of every block.

Inference replaces left-to-right argmax with B Euler steps per byte
through the EDM probability-flow ODE, starting from N(0, σ_max²·I) and
ending at a denoised next-byte embedding. Greedy argmax of (z @ E_out.T)
gives the committed byte. The per-block KV cache is extended from
x_emb on every observe() and reused for all subsequent predict() calls.
"""
from __future__ import annotations

__author__ = "@ab-10"

import math
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Noise schedule (equi-probability partitioning of LogNormal(P_mean, P_std))
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Acklam's inverse standard-normal CDF. Relative error < 1.15e-9 over
    # (0, 1) — plenty for boundary computation. Local impl so the
    # submission doesn't need scipy in the runtime image.
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    p_low = 0.02425
    p_high = 1.0 - p_low
    def _num_c(q: float) -> float:
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])

    def _den_d(q: float) -> float:
        return ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return _num_c(q) / _den_d(q)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -_num_c(q) / _den_d(q)


def get_block_sigmas(B: int, sigma_min: float = 0.002, sigma_max: float = 80.0,
                     P_mean: float = -1.2, P_std: float = 1.2) -> list[float]:
    """Equi-probability partitioning: σ_b such that ∫_{σ_b}^{σ_{b+1}} p_noise = 1/B.

    Returns B+1 boundaries, ascending: σ_0=σ_min ... σ_B=σ_max.
    Block b ∈ [0, B-1] covers [σ_b, σ_{b+1}]; block 0 is low noise, block
    B-1 is high noise.
    """
    cdf_min = _norm_cdf((math.log(sigma_min) - P_mean) / P_std)
    cdf_max = _norm_cdf((math.log(sigma_max) - P_mean) / P_std)
    out = []
    for b in range(B + 1):
        q = cdf_min + (cdf_max - cdf_min) * (b / B)
        out.append(math.exp(P_mean + P_std * _norm_ppf(q)))
    return out


def sample_sigma_in_range(sigma_lo: float, sigma_hi: float,
                          P_mean: float = -1.2, P_std: float = 1.2) -> float:
    """Single σ from LogNormal(P_mean, P_std) truncated to [σ_lo, σ_hi]."""
    cdf_lo = _norm_cdf((math.log(sigma_lo) - P_mean) / P_std)
    cdf_hi = _norm_cdf((math.log(sigma_hi) - P_mean) / P_std)
    u = random.uniform(cdf_lo, cdf_hi)
    return math.exp(P_mean + P_std * _norm_ppf(u))


# ---------------------------------------------------------------------------
# c_noise sinusoidal embedder
# ---------------------------------------------------------------------------

class CondEmbed(nn.Module):
    """Map c_noise scalar (∈ ~[-2, 1.5]) to a cond_dim vector via
    sinusoidal features + 2-layer MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim),
        )

    def forward(self, c_noise: Tensor) -> Tensor:
        if c_noise.dim() == 0:
            c_noise = c_noise.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=c_noise.device, dtype=torch.float32) / max(1, half - 1)
        )
        args = c_noise.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# AdaLN
# ---------------------------------------------------------------------------

class AdaRMSNorm(nn.Module):
    """Pre-norm RMSNorm with affine (γ, β) predicted from c_noise embedding."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.dim = dim
        self.cond_proj = nn.Linear(cond_dim, 2 * dim)
        # Initialise so AdaLN is the identity at start (γ=0, β=0). Keeps
        # the model well-behaved before c_noise has any learned signal.
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: Tensor, cond_emb: Tensor) -> Tensor:
        gb = self.cond_proj(cond_emb.to(x.dtype))
        gamma, beta = gb.chunk(2, dim=-1)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        x_normed = F.rms_norm(x, (x.size(-1),))
        return x_normed * (1 + gamma) + beta


# ---------------------------------------------------------------------------
# RoPE (half-truncate, base=1024 — same as modded_nanogpt)
# ---------------------------------------------------------------------------

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer(
            "angular_freq",
            torch.cat([freq, freq.new_zeros(dim // 4)]),
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


# ---------------------------------------------------------------------------
# Cross-attention layer (Q ← z, K/V ← x_emb)
# ---------------------------------------------------------------------------

class CrossLayer(nn.Module):
    """One layer of: causal cross-attn from z to x_emb, ReLU²-MLP on z,
    AdaLN(c_noise) on both pre-norms.

    Two forward variants:
      * forward_train: T_q == T_k == T (causal mask).
      * forward_infer: T_q == 1; Q's RoPE is offset by the cache length.

    The K/V projections are exposed as ``extend_kv`` (inference) and
    ``project_kv`` (training) so the block can build the prefix K/V from
    x_emb either incrementally (observe) or in bulk (training batch).
    """

    def __init__(self, dim: int, head_dim: int, cond_dim: int):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp_fc = nn.Linear(dim, 4 * dim)
        self.mlp_proj = nn.Linear(4 * dim, dim)
        self.norm1 = AdaRMSNorm(dim, cond_dim)
        self.norm2 = AdaRMSNorm(dim, cond_dim)
        self.rotary = Rotary(head_dim)

    def project_kv(self, x_emb: Tensor, offset: int) -> tuple[Tensor, Tensor]:
        B, T = x_emb.shape[:2]
        k = self.k(x_emb).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x_emb).view(B, T, self.num_heads, self.head_dim)
        k = F.rms_norm(k, (k.size(-1),))
        k = self.rotary(k, offset=offset)
        return k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

    def extend_kv(self, x_emb: Tensor,
                  kv_cache: tuple[Tensor, Tensor] | None,
                  offset: int) -> tuple[Tensor, Tensor]:
        k_new, v_new = self.project_kv(x_emb, offset=offset)
        if kv_cache is None:
            return (k_new, v_new)
        k_old, v_old = kv_cache
        return (torch.cat([k_old, k_new], dim=2),
                torch.cat([v_old, v_new], dim=2))

    def _qproj(self, z: Tensor, offset: int) -> Tensor:
        B, T = z.shape[:2]
        q = self.q(z).view(B, T, self.num_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        q = self.rotary(q, offset=offset)
        return q.transpose(1, 2).contiguous()

    def forward_train(self, z: Tensor, k: Tensor, v: Tensor,
                      cond_emb: Tensor) -> Tensor:
        z_in = self.norm1(z, cond_emb)
        q = self._qproj(z_in, offset=0)
        # Causal: Q at position t attends to K at positions ≤ t. Q and K
        # share the same length T, so is_causal=True applies the standard
        # lower-triangular mask.
        attn = F.scaled_dot_product_attention(q, k, v, scale=0.12, is_causal=True)
        B, T = z.shape[:2]
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        z = z + self.proj(attn)
        h = self.mlp_fc(self.norm2(z, cond_emb))
        h = h.relu().square()
        z = z + self.mlp_proj(h)
        return z

    def forward_infer(self, z: Tensor, kv_cache: tuple[Tensor, Tensor],
                      q_offset: int, cond_emb: Tensor) -> Tensor:
        z_in = self.norm1(z, cond_emb)
        q = self._qproj(z_in, offset=q_offset)  # [B, H, 1, D]
        k, v = kv_cache
        # No is_causal: Q is a single position and all positions in the
        # cache are "past" by construction (observe() commits before
        # predict()). The kernel does standard dot-product attention.
        attn = F.scaled_dot_product_attention(q, k, v, scale=0.12)
        B = z.size(0)
        attn = attn.transpose(1, 2).contiguous().view(B, 1, -1)
        z = z + self.proj(attn)
        h = self.mlp_fc(self.norm2(z, cond_emb))
        h = h.relu().square()
        z = z + self.mlp_proj(h)
        return z


# ---------------------------------------------------------------------------
# DBlock (L/B cross-attention layers + final AdaLN)
# ---------------------------------------------------------------------------

class DBlock(nn.Module):
    def __init__(self, dim: int, head_dim: int, n_layers: int, cond_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossLayer(dim, head_dim, cond_dim) for _ in range(n_layers)
        ])
        self.norm_out = AdaRMSNorm(dim, cond_dim)

    def forward_train(self, z: Tensor, x_emb: Tensor, cond_emb: Tensor) -> Tensor:
        for layer in self.layers:
            k, v = layer.project_kv(x_emb, offset=0)
            z = layer.forward_train(z, k, v, cond_emb)
        return self.norm_out(z, cond_emb)

    def extend_kv(self, x_emb: Tensor,
                  kv_caches: list[tuple[Tensor, Tensor]] | None,
                  offset: int) -> list[tuple[Tensor, Tensor]]:
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        return [layer.extend_kv(x_emb, cache, offset)
                for layer, cache in zip(self.layers, kv_caches)]

    def forward_infer(self, z: Tensor,
                      kv_caches: list[tuple[Tensor, Tensor]],
                      q_offset: int, cond_emb: Tensor) -> Tensor:
        for layer, cache in zip(self.layers, kv_caches):
            z = layer.forward_infer(z, cache, q_offset, cond_emb)
        return self.norm_out(z, cond_emb)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class DBlocksAR(nn.Module):
    def __init__(self, vocab_size: int = 256, num_layers: int = 6,
                 model_dim: int = 384, head_dim: int = 64,
                 num_blocks: int = 3, cond_dim: int = 128,
                 max_len: int = 1024):
        super().__init__()
        assert num_layers % num_blocks == 0, "num_layers must divide num_blocks"
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.cond_dim = cond_dim
        self.max_len = max_len
        self.layers_per_block = num_layers // num_blocks
        self.embed = nn.Embedding(vocab_size, model_dim)
        # E_out: output table, L2-normalized at use time. Weight-tied
        # across {noising-target lookup, byte logits projection}.
        self.E_out = nn.Parameter(torch.empty(vocab_size, model_dim))
        nn.init.normal_(self.E_out, std=1.0 / model_dim**0.5)
        # Learned temperature for the CE head. E_out rows are L2-normalised
        # at use time and pred_y has magnitude ~σ_data, so the raw
        # pred_y @ E_out.T logits sit in roughly [-1, 1]. A softmax over
        # 256 classes saturates at CE ≈ ln(256·exp(-1)/(exp(1)+255·exp(-1)))
        # ≈ 3.58 even for a perfect one-vs-rest classifier. Multiplying by
        # a learned scalar (init 1/σ_data ≈ 20) lifts that ceiling.
        self.logit_scale = nn.Parameter(torch.tensor(float(model_dim) ** 0.5))
        self.cond_embed = CondEmbed(cond_dim)
        self.blocks = nn.ModuleList([
            DBlock(model_dim, head_dim, self.layers_per_block, cond_dim)
            for _ in range(num_blocks)
        ])
        # σ_max=5 (vs EDM's 80) — for unit-norm byte embeddings with
        # σ_data ≈ 0.05, σ_max=80 is 1600× σ_data and well past the
        # LogNormal(P_mean=-1.2, P_std=1.2) training distribution's tail
        # (P(σ>5) ≈ 0.05, P(σ>80) ≈ 1e-8). v2/v3 trained the high-σ
        # block on σ ∈ [0.51, 80] but mass concentrates below ~2, so the
        # inference Euler trajectory's first step at σ_max=80 is far
        # out-of-distribution and the network's denoising is near-random
        # there. Shrinking σ_max to 5 keeps the schedule inside the
        # training support.
        boundaries = get_block_sigmas(num_blocks, sigma_max=5.0)
        self.register_buffer("block_sigmas", torch.tensor(boundaries, dtype=torch.float32))
        # σ_data is calibrated at start of training; placeholder for now.
        self.register_buffer("sigma_data", torch.tensor(1.0 / model_dim**0.5, dtype=torch.float32))

    def normalize_eout(self) -> Tensor:
        return F.normalize(self.E_out, dim=-1)

    def block_range(self, b: int, gamma: float = 0.10) -> tuple[float, float]:
        s_lo = float(self.block_sigmas[b])
        s_hi = float(self.block_sigmas[b + 1])
        if gamma > 0.0:
            log_lo, log_hi = math.log(s_lo), math.log(s_hi)
            rng = log_hi - log_lo
            s_lo = max(math.exp(log_lo - gamma * rng), float(self.block_sigmas[0]))
            s_hi = min(math.exp(log_hi + gamma * rng), float(self.block_sigmas[-1]))
        return s_lo, s_hi


def _init_model(model: DBlocksAR) -> None:
    """Zero-init output projections so each block starts as ≈identity."""
    for name, p in model.named_parameters():
        if not name.endswith("weight"):
            continue
        if "proj" in name or "mlp_proj" in name:
            nn.init.zeros_(p)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainConfig:
    def __init__(self,
                 model_dim: int = 384,
                 num_layers: int = 6,
                 head_dim: int = 64,
                 num_blocks: int = 3,
                 cond_dim: int = 128,
                 max_len: int = 1024,
                 batch_size: int = 32,
                 baseline_steps: int = 2150,
                 n_steps: int | None = None,
                 gamma: float = 0.10,
                 lambda_ce: float = 1.0,
                 lr: float = 5e-4,
                 weight_decay: float = 0.0,
                 cooldown_frac: float = 0.7,
                 log_every: int = 200):
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.cond_dim = cond_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.baseline_steps = baseline_steps
        # Paper Appendix D.1 "Fair comparison": B × baseline so total
        # layer-updates match end-to-end.
        self.n_steps = n_steps if n_steps is not None else num_blocks * baseline_steps
        self.gamma = gamma
        self.lambda_ce = lambda_ce
        self.lr = lr
        self.weight_decay = weight_decay
        self.cooldown_frac = cooldown_frac
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"B={self.num_blocks} bs={self.batch_size} T={self.max_len} "
                f"steps={self.n_steps} γ={self.gamma} λ_ce={self.lambda_ce})")


def calibrate_sigma_data(model: DBlocksAR) -> float:
    """Empirical per-coord std of L2-normalized E_out rows (≈ 1/√d)."""
    with torch.no_grad():
        E = model.normalize_eout()
        return float(E.std().item())


@contextmanager
def _maybe_autocast(device: torch.device):
    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def _train(text: str, cfg: TrainConfig, device: torch.device) -> DBlocksAR:
    # Bytes held on-device as uint8 (the modded_nanogpt trick).
    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < cfg.max_len + 1:
        raise ValueError(f"need at least {cfg.max_len+1} bytes; got {n}")

    model = DBlocksAR(
        vocab_size=256, num_layers=cfg.num_layers, model_dim=cfg.model_dim,
        head_dim=cfg.head_dim, num_blocks=cfg.num_blocks,
        cond_dim=cfg.cond_dim, max_len=cfg.max_len,
    ).to(device)
    _init_model(model)

    sigma_data = calibrate_sigma_data(model)
    model.sigma_data.fill_(sigma_data)
    print(f"[dblocks] σ_data calibrated to {sigma_data:.4f}", flush=True)

    boundaries = model.block_sigmas.tolist()
    print(f"[dblocks] block boundaries: "
          f"[{', '.join(f'{s:.4f}' for s in boundaries)}]", flush=True)

    # One AdamW per block (so we can step only the active block per
    # iteration) + one shared optimizer for embed / E_out / cond_embed,
    # which sit on every block's gradient path and step every iteration.
    fused = (device.type == "cuda")
    block_opts = [
        AdamW(blk.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
              betas=(0.9, 0.95), eps=1e-10, fused=fused)
        for blk in model.blocks
    ]
    shared_params = (
        list(model.embed.parameters())
        + list(model.cond_embed.parameters())
        + [model.E_out, model.logit_scale]
    )
    shared_opt = AdamW(shared_params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                       betas=(0.9, 0.95), eps=1e-10, fused=fused)
    all_opts = block_opts + [shared_opt]
    for opt in all_opts:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dblocks] {n_params/1e6:.2f}M params  cfg={cfg}", flush=True)

    def set_lr(step: int) -> None:
        progress = step / max(1, cfg.n_steps)
        if progress < 1 - cfg.cooldown_frac:
            eta = 1.0
        else:
            eta = max(0.0, (1 - progress) / cfg.cooldown_frac)
        for opt in all_opts:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * eta

    model.train()
    t0 = time.monotonic()
    block_loss_ema: list[float | None] = [None] * cfg.num_blocks
    block_count = [0] * cfg.num_blocks
    sd = float(sigma_data)

    for step in range(cfg.n_steps):
        set_lr(step)
        idx = torch.randint(0, n - cfg.max_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(cfg.max_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]   # [B, T]
        y = flat[:, 1:]    # [B, T]

        b = random.randrange(cfg.num_blocks)
        s_lo, s_hi = model.block_range(b, gamma=cfg.gamma)
        sigma = sample_sigma_in_range(s_lo, s_hi)
        sigma_t = max(sigma, 1e-8)
        c_skip = sd**2 / (sigma_t**2 + sd**2)
        c_out = sigma_t * sd / math.sqrt(sigma_t**2 + sd**2)
        c_in = 1.0 / math.sqrt(sigma_t**2 + sd**2)
        c_noise = 0.25 * math.log(sigma_t)
        weight = (sigma_t**2 + sd**2) / (sigma_t * sd) ** 2

        # Only block b's optimizer + the shared optimizer participate in
        # this step. The other blocks see no gradient (they aren't even
        # forwarded), but zero_grad on the inactive opts is unnecessary —
        # set_to_none keeps their grads at None.
        block_opts[b].zero_grad(set_to_none=True)
        shared_opt.zero_grad(set_to_none=True)

        cond_in = torch.tensor([c_noise], device=device, dtype=torch.float32)

        with _maybe_autocast(device):
            x_emb = model.embed(x)             # [B, T, d]
            E = model.normalize_eout()         # [V, d]  (recomputed each step)
            y_emb = E[y]                        # [B, T, d]
            eps = torch.randn_like(y_emb)
            z = y_emb + sigma_t * eps
            cond = model.cond_embed(cond_in)    # [1, cond_dim]
            cond = cond.expand(cfg.batch_size, -1)

            z_in = z * c_in
            out = model.blocks[b].forward_train(z_in, x_emb, cond)
            pred_y = out * c_out + z * c_skip   # [B, T, d]

            loss_l2 = weight * (pred_y - y_emb).pow(2).mean()
            # Normalize pred_y before the CE head so logit magnitude is
            # independent of σ. Without this, at high σ pred_y has norm
            # ≈ σ_data and logit_scale·√d·σ_data ≈ 1 → softmax saturates
            # near CE ≈ 3.58 over 256 classes. With normalisation, logit
            # magnitude is ~logit_scale ≈ √d ≈ 20 regardless of σ.
            pred_y_n = F.normalize(pred_y.float(), dim=-1)
            logits = model.logit_scale * (pred_y_n @ E.float().t())
            loss_ce = F.cross_entropy(
                logits.reshape(-1, model.vocab_size), y.reshape(-1),
            )
            loss = loss_l2 + cfg.lambda_ce * loss_ce

        loss.backward()
        block_opts[b].step()
        shared_opt.step()

        block_count[b] += 1
        l = float(loss.item())
        prev = block_loss_ema[b]
        block_loss_ema[b] = l if prev is None else 0.95 * prev + 0.05 * l

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            per_block = "  ".join(
                f"b{i}={(block_loss_ema[i] if block_loss_ema[i] is not None else float('nan')):.3f}"
                f"({block_count[i]})"
                for i in range(cfg.num_blocks)
            )
            print(
                f"[dblocks] step {step:5d}/{cfg.n_steps}  "
                f"b={b} σ={sigma_t:.3f}  "
                f"loss={l:.4f} (l2={float(loss_l2):.3f} ce={float(loss_ce):.3f})  "
                f"{per_block}  elapsed={elapsed:.0f}s",
                flush=True,
            )

    return model


# ---------------------------------------------------------------------------
# Streaming inference (CharModel)
# ---------------------------------------------------------------------------

class DBlocksCharModel(CharModel):
    """B-step Euler inference over the per-block KV-cache.

    - reset(): seeds a single zero byte through all blocks' KV-caches.
    - observe(c): for each byte of c, projects K/V from x_emb and appends
      to every block's cache (B layer-stacks × L/B layers = L layer-forwards
      total per byte).
    - predict(): starting at σ_max and going down through B noise levels,
      Euler-steps z. Step i uses block (B-1-i), reading its own KV-cache.
      Final argmax over (z @ E_out.T) commits a byte.

    Inference cost per byte ≈ 2L layer-forwards (L for observe + L for
    predict). Matches paper Appendix A.2 footnote.
    """

    def __init__(self, model: DBlocksAR, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self.B = model.num_blocks
        self.dim = model.model_dim
        self.max_len = model.max_len
        self.sigma_data = float(model.sigma_data.item())
        boundaries = model.block_sigmas.tolist()
        # Inference σ schedule: walk boundaries from high to low. Step i
        # goes σ_schedule[i] → σ_schedule[i+1] and is routed to the block
        # whose [σ_b, σ_{b+1}] range contains σ_schedule[i].
        self.sigma_schedule = list(reversed(boundaries))  # [σ_max, ..., σ_min]
        # Block index for step i: σ_schedule[i] = boundaries[B-i], which
        # is the upper endpoint of block (B-1-i)'s range.
        self._kv: list | None = None
        self._pos: int = 0

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = [None] * self.B
        self._pos = 0
        # Seed with a single sentinel byte (0) so predict() has a valid
        # prefix in every block's KV cache before any real char arrives.
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        x_emb = self.model.embed(x)
        for b in range(self.B):
            self._kv[b] = self.model.blocks[b].extend_kv(
                x_emb, self._kv[b], offset=self._pos,
            )
        self._pos = 1

    @torch.no_grad()
    def predict(self) -> str:
        if self._kv is None:
            raise RuntimeError("predict() called before reset()")
        E = self.model.normalize_eout()
        q_offset = self._pos - 1  # Most recent observed position
        sigma_hi = self.sigma_schedule[0]
        z = torch.randn(1, 1, self.dim, device=self.device) * sigma_hi
        sd = self.sigma_data
        for i in range(self.B):
            sigma = self.sigma_schedule[i]
            sigma_next = self.sigma_schedule[i + 1]
            block_idx = self.B - 1 - i
            sigma_t = max(sigma, 1e-8)
            c_skip = sd**2 / (sigma_t**2 + sd**2)
            c_out = sigma_t * sd / math.sqrt(sigma_t**2 + sd**2)
            c_in = 1.0 / math.sqrt(sigma_t**2 + sd**2)
            c_noise = 0.25 * math.log(sigma_t)
            cond = self.model.cond_embed(
                torch.tensor([c_noise], device=self.device, dtype=torch.float32),
            )  # [1, cond_dim]
            z_in = z * c_in
            out = self.model.blocks[block_idx].forward_infer(
                z_in, self._kv[block_idx], q_offset=q_offset, cond_emb=cond,
            )
            denoised = out * c_out + z * c_skip
            d = (z - denoised) / sigma_t
            z = z + d * (sigma_next - sigma)
        # Use the last block's pred_y (= c_skip·z + c_out·out from the
        # final Euler step at σ→σ_min), L2-normalised to match the CE
        # head's training-time normalisation. For greedy argmax this is
        # invariant under positive scaling, so logit_scale is cosmetic at
        # inference; the normalisation is what matters for consistency
        # with the trained head.
        final = F.normalize(z.squeeze(0).squeeze(0).float(), dim=-1)
        logits = self.model.logit_scale * (final @ E.float().t())  # [V]
        byte_id = int(logits.argmax().item())
        try:
            return bytes([byte_id]).decode("utf-8")
        except UnicodeDecodeError:
            return ""

    @torch.no_grad()
    def observe(self, char: str) -> None:
        if self._kv is None:
            raise RuntimeError("observe() called before reset()")
        for byte in char.encode("utf-8"):
            self._maybe_trim_cache()
            x = torch.tensor([[byte]], dtype=torch.long, device=self.device)
            x_emb = self.model.embed(x)
            for b in range(self.B):
                self._kv[b] = self.model.blocks[b].extend_kv(
                    x_emb, self._kv[b], offset=self._pos,
                )
            self._pos += 1

    def _maybe_trim_cache(self) -> None:
        if self._kv is None:
            return
        # Each cache entry is (k, v) with k: [1, H, T_cached, D].
        cur = self._kv[0][0][0].shape[2]
        if cur < self.max_len:
            return
        keep = self.max_len - 1
        for b in range(self.B):
            self._kv[b] = [(k[:, :, -keep:], v[:, :, -keep:])
                           for (k, v) in self._kv[b]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[dblocks] SEED={seed}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    model = _train(train_text, cfg, device)
    return DBlocksCharModel(model)
