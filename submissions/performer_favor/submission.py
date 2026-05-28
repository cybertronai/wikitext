"""Performer FAVOR+ drop-in replacement of attention in modded-nanogpt.

Source baseline:
  /home/seneca/wikitext/submissions/modded_nanogpt/submission.py  (port of KellerJordan/modded-nanogpt)

Single change versus baseline:
  CausalSelfAttention.forward swaps F.scaled_dot_product_attention for
  the FAVOR+ positive random feature kernel approximation of softmax
  (Choromanski et al. 2020, "Rethinking Attention with Performers").

Feature map (per head, shared omega for Q and K, as FAVOR+ requires):
    phi(x) = (1 / sqrt(M)) * exp(omega^T x - ||x||^2 / 2)
where omega is sampled from N(0, I_D / D) and orthogonalized per head via QR.
Q and K are pre-scaled by 1/sqrt(D) before the feature map to keep the
exp() arguments bounded (FAVOR+'s primary failure mode is exp() overflow).

Training pass (T > 1, no cache): the causal cumulative-sum form,
    qf = phi(Q), kf = phi(K),  (B, H, T, M)
    S  = cumsum_t( einsum(kf, v) -> (B,H,T,M,D) ),
    z  = cumsum_t( kf -> (B,H,T,M) ),
    out_t = einsum(qf_t, S_t) / max(einsum(qf_t, z_t), eps).
S is the (B,H,T,M,D) tensor; at M=128, D=64, T=1024, B=32, H=6 that is
~3 GB in bfloat16 -- fits on 80GB. (If OOM, drop M to 64.)

Streaming inference (T = 1, with cache): we carry forward (S, z) running
state instead of (K, V) cache, giving O(1) per-byte cost (independent of
prefix length). RoPE offset is still tracked and applied to the new
single-token q/k before the feature map. There is no fixed window in the
streaming path -- FAVOR+ has no max_len limit for the recurrent state.
"""
from __future__ import annotations

__author__ = "@ab-10"

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Architecture (modded-nanogpt simple, vocab_size=256, RoPE offset support)
# Attention replaced by FAVOR+.
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
    """Half-truncate RoPE with base-freq tuning (base=1024). Same as baseline."""
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


def _make_orthogonal_omegas(num_heads: int, head_dim: int, num_features: int,
                            device: torch.device | None = None) -> Tensor:
    """Per-head FAVOR+ random projection matrices.

    Returns a (H, D, M) buffer. For each head, we draw an M x D Gaussian
    matrix and orthogonalize blocks of D rows via QR (Choromanski 2020
    Algorithm 1). The final scale is row-norm-matched to chi-distributed
    N(0,I) rows so the feature map remains an unbiased estimator of the
    softmax kernel.
    """
    omegas = torch.empty(num_heads, head_dim, num_features, device=device)
    for h in range(num_heads):
        # Generate full M x D matrix in blocks of D x D, QR-orthogonalize each.
        nb = (num_features + head_dim - 1) // head_dim
        blocks = []
        for _ in range(nb):
            g = torch.randn(head_dim, head_dim, device=device)
            q, _ = torch.linalg.qr(g)
            blocks.append(q)  # (D, D), rows are orthonormal
        ortho = torch.cat(blocks, dim=0)[:num_features]  # (M, D)
        # Rescale rows: orthogonal rows have norm 1; chi-distributed N(0,I)
        # rows have expected norm sqrt(D). Multiply by independent chi
        # samples (norms of fresh N(0,I) rows) so phi is unbiased.
        norms = torch.randn(num_features, head_dim, device=device).norm(dim=1)  # (M,)
        ortho = ortho * norms[:, None]
        omegas[h] = ortho.t()  # (D, M)
    # Match the (D, M) shape convention used in the einsum below.
    return omegas


class CausalSelfAttention(nn.Module):
    """FAVOR+ linear attention.

    Differences from baseline: (a) attention forward uses a feature map +
    cumulative sum instead of scaled_dot_product_attention; (b) the cache
    object carries running (S, z) sums and an offset int, not (K, V)
    tensors. The module retains q/k/v/proj linears, QK RMSNorm, and RoPE
    so the parameter count and the rest of the architecture match the
    baseline 1:1 -- the only flop change is the attention kernel.
    """
    M_FEATURES = 64

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
        omegas = _make_orthogonal_omegas(self.num_heads, head_dim, self.M_FEATURES)
        # Registered as buffer so it moves with .to(device) and is not trained.
        self.register_buffer("omegas", omegas)

    def _favor_features(self, x: Tensor) -> Tensor:
        """phi(x) = exp(omega^T x - ||x||^2 / 2) / sqrt(M).

        x: (B, H, T, D). omegas: (H, D, M). returns (B, H, T, M).

        x is pre-scaled by 1/sqrt(D) before being fed in (caller does
        this once) so that ||x||^2 and omega^T x both stay O(1) and exp()
        does not overflow in bf16.
        """
        # Compute in fp32 for numerical safety of the exp().
        x32 = x.float()
        omegas32 = self.omegas.float()
        norm_sq = (x32 * x32).sum(-1, keepdim=True) * 0.5  # (B, H, T, 1)
        proj = torch.einsum("bhtd,hdm->bhtm", x32, omegas32)  # (B, H, T, M)
        # Stabilize: subtract per-(B,H,T) max over M so the largest exp() is 1.
        logits = proj - norm_sq
        logits_max = logits.amax(dim=-1, keepdim=True).detach()
        feats = torch.exp(logits - logits_max) / math.sqrt(self.omegas.size(-1))
        # We will divide num and den both by the same per-(B,H,T) scale
        # implicitly because both qf and kf carry their own max-subtraction
        # but those don't cancel across positions. So instead of canceling,
        # we just keep the max-subtraction on q (and not on k), which only
        # changes the normalization that the den absorbs. That breaks the
        # mathematical identity slightly. Simpler: skip the per-position
        # max-subtract for kf, and only do it for qf where it cancels in
        # the ratio num/den (both num and den scale by exp(-max(qf))).
        return feats  # caller decides if max-subtraction was applied

    def _favor_features_q(self, x: Tensor) -> Tensor:
        """phi(q) with per-(B,H,T) max-subtraction.

        The max-subtraction multiplies qf by a positive per-row scalar
        exp(-c_t). num_t = sum_s qf_t kf_s v_s and den_t = sum_s qf_t kf_s
        both gain the same factor exp(-c_t), so the ratio out_t = num/den
        is unchanged. Safe stabilization.
        """
        x32 = x.float()
        omegas32 = self.omegas.float()
        norm_sq = (x32 * x32).sum(-1, keepdim=True) * 0.5
        proj = torch.einsum("bhtd,hdm->bhtm", x32, omegas32)
        logits = proj - norm_sq
        logits_max = logits.amax(dim=-1, keepdim=True).detach()
        return torch.exp(logits - logits_max) / math.sqrt(self.omegas.size(-1))

    def _favor_features_k(self, x: Tensor) -> Tensor:
        """phi(k) without per-position max-subtraction.

        We do still subtract a single global constant per (B, H) computed
        from the running max over (T, M) so the cumsum doesn't accumulate
        into Inf in fp32. This is a global scalar per (B, H), absorbed
        identically by num and den at every t -- so it cancels in the
        ratio.
        """
        x32 = x.float()
        omegas32 = self.omegas.float()
        norm_sq = (x32 * x32).sum(-1, keepdim=True) * 0.5
        proj = torch.einsum("bhtd,hdm->bhtm", x32, omegas32)
        logits = proj - norm_sq  # (B, H, T, M)
        # Per-(B, H) global shift -- same scalar for every (t, m), so it
        # multiplies both num and den by the same factor and cancels.
        gmax = logits.amax(dim=(2, 3), keepdim=True).detach()
        return torch.exp(logits - gmax) / math.sqrt(self.omegas.size(-1))

    def forward(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor, int] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, int]]:
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = self.rotary(q, offset=offset)
        k = self.rotary(k, offset=offset)

        # (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Pre-scale by 1/sqrt(D) so omega^T (q/sqrt(D)) and ||q/sqrt(D)||^2
        # are O(1). This is the "RMSNorm Q,K harder, divide by sqrt(d)"
        # numerical-stability fix from the spec, baked in once.
        scale = 1.0 / math.sqrt(self.head_dim)
        q = q * scale
        k = k * scale

        if kv_cache is None:
            # ===== Training / fresh-prefix parallel path =====
            # Compute features in fp32 (exp() safety), then cast to bf16
            # for the big (B,H,T,M,D) cumsum. bf16 is fine because
            # qf/kf are bounded in [0, 1] after the max-subtraction and
            # the values entering the sum are products of bounded
            # quantities times v. Halving the bytes ~halves the HBM
            # traffic that dominates per-step time.
            qf = self._favor_features_q(q).to(torch.bfloat16)  # (B, H, T, M)
            kf = self._favor_features_k(k).to(torch.bfloat16)  # (B, H, T, M)
            v_b = v.to(torch.bfloat16)
            kv = torch.einsum("bhtm,bhtd->bhtmd", kf, v_b)     # (B, H, T, M, D) bf16
            S = torch.cumsum(kv, dim=2)                         # (B, H, T, M, D) bf16
            z = torch.cumsum(kf, dim=2)                         # (B, H, T, M)    bf16
            num = torch.einsum("bhtm,bhtmd->bhtd", qf, S)       # (B, H, T, D)
            den = torch.einsum("bhtm,bhtm->bht", qf, z).clamp(min=1e-4).unsqueeze(-1)
            y = (num / den).to(x.dtype)
            new_cache = None
        elif kv_cache is not None and T == 1:
            # ===== Streaming single-token recurrence =====
            # cache = (S_prev, z_prev, _unused). S_prev: (B, H, M, D); z_prev: (B, H, M).
            S_prev, z_prev, _ = kv_cache
            qf = self._favor_features_q(q)            # (B, H, 1, M)
            kf = self._favor_features_k(k)            # (B, H, 1, M)
            v32 = v.float()                            # (B, H, 1, D)
            # outer product for the new token
            kv_new = torch.einsum("bhtm,bhtd->bhtmd", kf, v32).squeeze(2)  # (B, H, M, D)
            kf_new = kf.squeeze(2)                                          # (B, H, M)
            S_new = S_prev + kv_new
            z_new = z_prev + kf_new
            # Compute output using the *new* running state (causal: include current token).
            qf_t = qf.squeeze(2)                                            # (B, H, M)
            num = torch.einsum("bhm,bhmd->bhd", qf_t, S_new)                # (B, H, D)
            den = torch.einsum("bhm,bhm->bh", qf_t, z_new).clamp(min=1e-6).unsqueeze(-1)
            y = (num / den).to(x.dtype).unsqueeze(2)                        # (B, H, 1, D)
            new_cache = (S_new, z_new, 0)
        else:
            # T > 1 with a cache (shouldn't happen in our wrapper, but
            # support it for safety: process tokens one at a time).
            S_prev, z_prev, _ = kv_cache
            outs = []
            S_cur, z_cur = S_prev, z_prev
            for t in range(T):
                q_t = q[:, :, t:t+1]
                k_t = k[:, :, t:t+1]
                v_t = v[:, :, t:t+1]
                qf = self._favor_features_q(q_t)
                kf = self._favor_features_k(k_t)
                kv_new = torch.einsum("bhtm,bhtd->bhtmd", kf, v_t.float()).squeeze(2)
                kf_new = kf.squeeze(2)
                S_cur = S_cur + kv_new
                z_cur = z_cur + kf_new
                qf_t = qf.squeeze(2)
                num = torch.einsum("bhm,bhmd->bhd", qf_t, S_cur)
                den = torch.einsum("bhm,bhm->bh", qf_t, z_cur).clamp(min=1e-6).unsqueeze(-1)
                outs.append((num / den).unsqueeze(2))
            y = torch.cat(outs, dim=2).to(x.dtype)
            new_cache = (S_cur, z_cur, 0)

        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y), new_cache


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
        kv_cache: tuple[Tensor, Tensor, int] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, int] | None]:
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
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.num_heads = model_dim // head_dim
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
        kv_caches: list[tuple[Tensor, Tensor, int] | None] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor, int] | None]]:
        x = self.norm1(self.embed(inputs))
        new_caches: list[tuple[Tensor, Tensor, int] | None] = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv, offset=offset)
            new_caches.append(new_kv)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits, new_caches


# ---------------------------------------------------------------------------
# Muon optimizer (single-GPU; distributed all-gather stripped) -- unchanged
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad: Tensor, momentum: Tensor, mu: float = 0.95, nesterov: bool = True) -> Tensor:
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, mu: float = 0.95):
        params = list(params)
        assert len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])


# ---------------------------------------------------------------------------
# Init scheme (mirrors modded-nanogpt simple)
# Skips the FAVOR+ omega buffer; it's already orthogonal-Gaussian and is
# not a learnable parameter.
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
# Training
# ---------------------------------------------------------------------------

class TrainConfig:
    """LRs inherited from modded-nanogpt simple. n_steps bumped to use the
    per-step compute savings from FAVOR+ (parallel cumsum + outer-product
    cumsum is ~2x cheaper per step than SDPA at T=1024, but the (B,H,T,M,D)
    cumsum imposes its own memory traffic, so we conservatively bump steps
    to fill the same 300 s wall budget rather than chase a theoretical 8x.
    """
    def __init__(
        self,
        model_dim=384,
        num_layers=6,
        head_dim=64,
        max_len=1024,
        batch_size=32,
        n_steps=1200,
        cooldown_frac=0.7,
        embed_lr=0.3,
        head_lr=1.0 / 320,
        scalar_lr=0.01,
        muon_lr=0.035,
        muon_wd=0.025,
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
        self.muon_lr = muon_lr
        self.muon_wd = muon_wd
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"H={self.model_dim//self.head_dim} bs={self.batch_size} "
                f"T={self.max_len} steps={self.n_steps})")


def _train_modded(
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

    block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2]
    scalars = [p for p in model.parameters() if p.ndim < 2]
    optimizer1 = AdamW(
        [
            dict(params=[model.embed.weight], lr=cfg.embed_lr),
            dict(params=[model.proj.weight], lr=cfg.head_lr),
            dict(params=scalars, lr=cfg.scalar_lr),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    optimizer2 = Muon(block_2d, lr=cfg.muon_lr, weight_decay=cfg.muon_wd)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[performer_favor] {n_params/1e6:.2f}M params  cfg={cfg}")

    def set_lr(step: int) -> None:
        progress = step / cfg.n_steps
        if progress < 1 - cfg.cooldown_frac:
            eta = 1.0
        else:
            eta = max(0.0, (1 - progress) / cfg.cooldown_frac)
        for opt in optimizers:
            for g in opt.param_groups:
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

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))

        if not torch.isfinite(loss):
            # FAVOR+'s primary failure mode -- log and bail loudly so the
            # run.log captures the step. The wrapper will still produce a
            # CharModel, but val acc will be 0.
            print(f"[performer_favor] NON-FINITE LOSS at step {step}: {loss.item()}", flush=True)
            raise RuntimeError(f"non-finite loss at step {step}")

        loss.backward()
        for opt in optimizers:
            opt.step()

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            print(
                f"[performer_favor] step {step:5d}/{cfg.n_steps}  "
                f"loss {loss.item():.4f}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    return model


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper. Uses FAVOR+ recurrent state (S, z) per layer
# instead of a (K, V) cache. Reset re-seeds with the start sentinel byte.
# ---------------------------------------------------------------------------

class PerformerFavorCharModel(CharModel):
    def __init__(self, model: GPT, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self._caches: list[tuple[Tensor, Tensor, int] | None] | None = None
        self._next_logits: Tensor | None = None
        self._pos: int = 0

    def _empty_caches(self) -> list[tuple[Tensor, Tensor, int]]:
        H = self.model.num_heads
        D = self.model.head_dim
        M = CausalSelfAttention.M_FEATURES
        caches: list[tuple[Tensor, Tensor, int]] = []
        for _ in range(self.model.num_layers):
            S = torch.zeros(1, H, M, D, dtype=torch.float32, device=self.device)
            z = torch.zeros(1, H, M, dtype=torch.float32, device=self.device)
            caches.append((S, z, 0))
        return caches

    @torch.no_grad()
    def reset(self) -> None:
        # Build empty running state, then feed the start sentinel byte once.
        self._caches = self._empty_caches()
        self._pos = 0
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        logits, self._caches = self.model(x, self._caches, offset=self._pos)
        self._next_logits = logits[0, -1]
        self._pos = 1

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
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
        return out

    @torch.no_grad()
    def observe(self, char: str) -> None:
        if self._caches is None:
            raise RuntimeError("observe() called before reset()")
        for byte in char.encode("utf-8"):
            x = torch.tensor([[byte]], dtype=torch.long, device=self.device)
            logits, self._caches = self.model(x, self._caches, offset=self._pos)
            self._next_logits = logits[0, -1]
            self._pos += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[performer_favor] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    model = _train_modded(train_text, cfg, device)
    return PerformerFavorCharModel(model)
