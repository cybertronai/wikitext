"""NoProp-style local-denoising terminal stack on a 5-layer modded_nanogpt body.

Experiment 09 (Variant A):
  * 5-layer modded_nanogpt body (transformer trunk, SGD-trained)
  * Parallel linear classification head ("SGD head") trained with normal CE
    backprop into the body — gives the body a global supervision signal so
    the hidden state h is informative.
  * NoPropTerminalStack: T=10 chained denoising sub-blocks + label_embed
    (256 x 128) + linear readout(128, 256). Trained per Li-Teh-Pascanu
    arXiv 2503.24322 Eq. 8 (per-step SNR-weighted L2 + cross-entropy
    reconstruction + KL prior on z_0). The body's hidden state is DETACHED
    before entering the NoProp stack — no NoProp gradient flows into the
    body.
  * Inference: use ONLY the NoProp readout. For each token we run the
    reverse denoising chain z_T -> ... -> z_0 from N(0, I) and argmax over
    readout(z_0). This isolates "does NoProp produce a usable classifier".

Diagnostics logged at training time:
  * mean cos-sim(predicted z_0, label_embed(y))  — denoiser-collapse probe
  * fraction of label_embed rows with norm < 0.1 — class-collapse probe
  * SGD-head val acc (subset)                    — body-quality reference
  * NoProp-head val acc (subset)                 — the actual contribution

References:
  Li, Teh, Pascanu 2025  arXiv 2503.24322  (NoProp, Eq. 6 + Eq. 8)
  Ho, Jain, Abbeel 2020  NeurIPS           (DDPM SNR weighting)
  Nichol & Dhariwal 2021                   (cosine alpha-bar schedule)
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
# modded-nanogpt body (5-layer; lm_head kept as the SGD classification head)
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


class GPTBody(nn.Module):
    """5-layer modded body. Exposes body_hidden() for the NoProp stack and
    a classical lm_head ("proj") for the parallel SGD-trained classifier.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        num_layers: int = 5,
        model_dim: int = 384,
        head_dim: int = 64,
        max_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(
            [Block(model_dim, head_dim=head_dim) for _ in range(num_layers)]
        )
        self.proj = Linear(model_dim, vocab_size)   # SGD classification head
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def body_hidden(
        self,
        inputs: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Return the post-norm hidden state h that feeds BOTH the SGD head
        and the NoProp stack. Float32 output for stable downstream use.
        """
        x = self.norm1(self.embed(inputs))
        new_caches: list[tuple[Tensor, Tensor]] = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv, offset=offset)
            new_caches.append(new_kv)
        h = self.norm2(x).float()
        return h, new_caches

    def forward(
        self,
        inputs: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        """Return (logits_sgd, h, new_caches).

        logits_sgd is soft-capped (cap=15) to match modded-nanogpt.
        h is the hidden state passed into the NoProp stack.
        """
        h, new_caches = self.body_hidden(inputs, kv_caches, offset=offset)
        logits = self.proj(h).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits, h, new_caches


# ---------------------------------------------------------------------------
# NoProp terminal stack (T=10 chained denoisers + label_embed + readout)
# ---------------------------------------------------------------------------

def cosine_alpha_bar(T: int, s: float = 0.008, device=None, dtype=torch.float32) -> Tensor:
    """Cosine alpha_bar schedule in NoProp indexing convention.

    NoProp (Li-Teh-Pascanu 2503.24322) writes the forward noising as
        q(z_t | y) = N(sqrt(abar_t) * u_y, (1 - abar_t) * I)
    with abar_t INCREASING from ~0 at t=0 (pure noise) to ~1 at t=T
    (clean label embedding). This is the opposite indexing convention
    from standard DDPM; under it SNR(t) = abar_t / (1-abar_t) is
    monotonically increasing in t, so the per-step weight
    `SNR(t) - SNR(t-1)` (Eq. 8) is non-negative as written.

    We take the Nichol-Dhariwal cosine schedule
        f(u) = cos((u+s)/(1+s) * pi/2)^2
    and FLIP its indexing: abar_t = 1 - f(t/T)/f(0). This gives
    abar_0 = 0 (numerically clamped) and abar_T ~ 1. Returns shape (T+1,).
    Clamped to [1e-5, 1 - 1e-5] for numerical stability of SNR terms.
    """
    t = torch.arange(T + 1, device=device, dtype=dtype)
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    abar_ddpm = f / f[0]              # decreasing 1 -> ~0
    abar = 1.0 - abar_ddpm            # increasing ~0 -> 1, NoProp convention
    # Conservative endpoint clamps: abar_T=0.98 caps SNR_T at 49, abar_0=0.02
    # avoids near-singular SNR_0=0. Without these the final SNR diff
    # explodes (~1e5) and dominates the per-step L2 loss.
    return abar.clamp(0.02, 0.98)


class Denoiser(nn.Module):
    """Small MLP taking concat(h, z_prev, t_scalar) -> predicted label_embed."""
    def __init__(self, d_h: int, d_label: int, hidden: int = 384):
        super().__init__()
        self.fc1 = nn.Linear(d_h + d_label + 1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, d_label)

    def forward(self, h: Tensor, z_prev: Tensor, t_scalar: float) -> Tensor:
        t_col = torch.full(h.shape[:-1] + (1,), t_scalar,
                           device=h.device, dtype=h.dtype)
        x = torch.cat([h, z_prev, t_col], dim=-1)
        x = self.fc1(x)
        x = x.relu().square()
        x = self.fc2(x)
        x = x.relu().square()
        x = self.fc3(x)
        return x


class NoPropTerminalStack(nn.Module):
    """T=10 denoising sub-blocks + 256x128 label_embed + 128->256 readout."""

    def __init__(
        self,
        d_h: int,
        d_label: int = 128,
        vocab_size: int = 256,
        T: int = 10,
        denoiser_hidden: int = 384,
    ):
        super().__init__()
        self.T = T
        self.d_label = d_label
        self.vocab_size = vocab_size
        self.label_embed = nn.Embedding(vocab_size, d_label)
        nn.init.normal_(self.label_embed.weight, std=0.5)
        self.denoisers = nn.ModuleList(
            [Denoiser(d_h, d_label, hidden=denoiser_hidden) for _ in range(T)]
        )
        self.readout = nn.Linear(d_label, vocab_size)
        nn.init.zeros_(self.readout.bias)
        nn.init.normal_(self.readout.weight, std=0.02)

    @torch.no_grad()
    def inference_logits(self, h: Tensor, alpha_bars: Tensor) -> Tensor:
        """Run the denoising chain in NoProp convention z_0 -> z_T.

        h:           (..., d_h)
        alpha_bars:  (T+1,) increasing from ~0 (pure noise at t=0) to ~1
                     (clean label embedding at t=T)
        returns:     logits over vocab, shape (..., vocab_size)

        Each denoiser at index t-1 was trained on training pair
            z_{t-1} = sqrt(abar_{t-1}) * y_emb + sqrt(1-abar_{t-1}) * eps
            target  = y_emb
            input   = (h, z_{t-1}, t/T)
        So at inference, given the current state z_{t-1}, denoisers[t-1]
        predicts the clean target u_pred. We then re-noise deterministically
        to the next level:
            z_t = sqrt(abar_t) * u_pred
        Starting from z_0 ~ N(0, I) we iterate t = 1, 2, ..., T. The final
        u_pred at t=T is fed to the readout.
        """
        T = self.T
        z = torch.randn(h.shape[:-1] + (self.d_label,),
                        device=h.device, dtype=h.dtype)
        u_pred = z
        for t in range(1, T + 1):
            u_pred = self.denoisers[t - 1](h, z, t / T)
            abar_t = alpha_bars[t]
            z = abar_t.sqrt() * u_pred
        return self.readout(u_pred)


# ---------------------------------------------------------------------------
# Muon optimizer (unchanged from baseline)
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


def _init_modded_body(model: GPTBody) -> None:
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
    def __init__(
        self,
        model_dim=384,
        num_layers=5,                # 5-layer body per spec
        head_dim=64,
        max_len=1024,
        batch_size=32,
        n_steps=1800,                # leave headroom for NoProp overhead
        cooldown_frac=0.7,
        embed_lr=0.3,
        head_lr=1.0 / 320,
        scalar_lr=0.01,
        muon_lr=0.035,
        muon_wd=0.025,
        noprop_lr=3e-3,
        noprop_T=10,
        d_label=128,
        denoiser_hidden=384,
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
        self.noprop_lr = noprop_lr
        self.noprop_T = noprop_T
        self.d_label = d_label
        self.denoiser_hidden = denoiser_hidden
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"bs={self.batch_size} T_seq={self.max_len} "
                f"steps={self.n_steps} noprop_T={self.noprop_T} "
                f"d_label={self.d_label})")


def _train_noprop(
    text: str,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[GPTBody, NoPropTerminalStack, Tensor]:
    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < cfg.max_len + 1:
        raise ValueError(f"need at least {cfg.max_len+1} bytes; got {n}")

    body = GPTBody(
        vocab_size=256,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        head_dim=cfg.head_dim,
        max_len=cfg.max_len,
    ).to(device)
    _init_modded_body(body)

    np_stack = NoPropTerminalStack(
        d_h=cfg.model_dim,
        d_label=cfg.d_label,
        vocab_size=256,
        T=cfg.noprop_T,
        denoiser_hidden=cfg.denoiser_hidden,
    ).to(device)

    block_2d = [p for p in body.blocks.parameters() if p.ndim >= 2]
    scalars = [p for p in body.parameters() if p.ndim < 2]
    opt_body_adam = AdamW(
        [
            dict(params=[body.embed.weight], lr=cfg.embed_lr),
            dict(params=[body.proj.weight], lr=cfg.head_lr),
            dict(params=scalars, lr=cfg.scalar_lr),
        ],
        betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    opt_body_muon = Muon(block_2d, lr=cfg.muon_lr, weight_decay=cfg.muon_wd)
    opt_noprop = AdamW(
        np_stack.parameters(),
        lr=cfg.noprop_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    optimizers = [opt_body_adam, opt_body_muon, opt_noprop]
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    alpha_bars = cosine_alpha_bar(cfg.noprop_T, device=device, dtype=torch.float32)

    n_body = sum(p.numel() for p in body.parameters())
    n_np = sum(p.numel() for p in np_stack.parameters())
    print(f"[noprop] body={n_body/1e6:.2f}M  noprop={n_np/1e6:.2f}M  cfg={cfg}")
    print(f"[noprop] alpha_bars = {[round(x, 4) for x in alpha_bars.tolist()]}")

    def set_lr(step: int) -> None:
        progress = step / cfg.n_steps
        if progress < 1 - cfg.cooldown_frac:
            eta = 1.0
        else:
            eta = max(0.0, (1 - progress) / cfg.cooldown_frac)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * eta

    body.train()
    np_stack.train()
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
                logits_sgd, h, _ = body(x)
        else:
            logits_sgd, h, _ = body(x)

        # --- SGD head loss: full grad into body ---
        loss_sgd = F.cross_entropy(
            logits_sgd.reshape(-1, 256), y.reshape(-1)
        )

        # --- NoProp losses: gradient blocked into body ---
        h_det = h.detach()
        y_emb = np_stack.label_embed(y)  # (B, T_seq, d_label), float32

        # Per-step SNR-weighted L2 (Eq. 8 first term).
        loss_np = h.new_zeros((), dtype=torch.float32)
        for t in range(1, cfg.noprop_T + 1):
            abar_t = alpha_bars[t]
            abar_tm1 = alpha_bars[t - 1]
            eps = torch.randn_like(y_emb)
            z_tm1 = abar_tm1.sqrt() * y_emb + (1.0 - abar_tm1).sqrt() * eps
            u_pred = np_stack.denoisers[t - 1](h_det, z_tm1, t / cfg.noprop_T)
            snr_t = abar_t / (1.0 - abar_t)
            snr_tm1 = abar_tm1 / (1.0 - abar_tm1)
            w = (snr_t - snr_tm1).clamp(min=0.0)
            loss_np = loss_np + w * ((u_pred - y_emb) ** 2).mean()

        # Reconstruction term (Eq. 8 second term): train readout against
        # noisy z_T (ground-truth) so readout learns to recover y from a
        # near-pure-noise sample of the target embedding.
        eps_T = torch.randn_like(y_emb)
        abar_T = alpha_bars[cfg.noprop_T]
        z_T_gt = abar_T.sqrt() * y_emb + (1.0 - abar_T).sqrt() * eps_T
        logits_np = np_stack.readout(z_T_gt)
        loss_recon = F.cross_entropy(logits_np.reshape(-1, 256), y.reshape(-1))

        # KL term: q(z_0|y) ~= delta on label_embed(y); KL vs N(0,I) ~=
        # 0.5 * ||y_emb||^2 per element (Eq. 8 third term, simplified).
        loss_kl = 0.5 * (y_emb ** 2).mean()

        total = loss_sgd + loss_np + loss_recon + 1e-3 * loss_kl
        total.backward()
        for opt in optimizers:
            opt.step()

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            with torch.no_grad():
                # Diagnostic: the FINAL denoiser (t=T) sees the cleanest
                # noisy input z_{T-1}; if NoProp is learning at all it
                # should hit very high cos-sim with y_emb here. Also
                # report the t=1 (hardest, nearly pure noise) cos-sim —
                # this is the one that drives early inference quality.
                abar_T = alpha_bars[cfg.noprop_T]
                abar_Tm1 = alpha_bars[cfg.noprop_T - 1]
                eps_T = torch.randn_like(y_emb)
                z_Tm1_gt = abar_Tm1.sqrt() * y_emb + (1.0 - abar_Tm1).sqrt() * eps_T
                u_T_pred = np_stack.denoisers[-1](h_det, z_Tm1_gt, 1.0)
                cos_clean = F.cosine_similarity(
                    u_T_pred.reshape(-1, cfg.d_label),
                    y_emb.reshape(-1, cfg.d_label),
                    dim=-1,
                ).mean().item()
                abar_0 = alpha_bars[0]
                eps_0 = torch.randn_like(y_emb)
                z_0 = abar_0.sqrt() * y_emb + (1.0 - abar_0).sqrt() * eps_0
                u_1_pred = np_stack.denoisers[0](h_det, z_0, 1.0 / cfg.noprop_T)
                cos_noisy = F.cosine_similarity(
                    u_1_pred.reshape(-1, cfg.d_label),
                    y_emb.reshape(-1, cfg.d_label),
                    dim=-1,
                ).mean().item()
                row_norms = np_stack.label_embed.weight.norm(dim=-1)
                frac_dead = (row_norms < 0.1).float().mean().item()
                row_norms = np_stack.label_embed.weight.norm(dim=-1)
                frac_dead = (row_norms < 0.1).float().mean().item()
            print(
                f"[noprop] step {step:5d}/{cfg.n_steps}  "
                f"sgd {loss_sgd.item():.4f}  "
                f"np {float(loss_np):.4f}  "
                f"rec {loss_recon.item():.4f}  "
                f"kl {loss_kl.item():.4f}  "
                f"cos_clean {cos_clean:+.3f}  "
                f"cos_noisy {cos_noisy:+.3f}  "
                f"dead_emb {frac_dead:.3f}  "
                f"row_norm_mean {row_norms.mean().item():.3f}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    return body, np_stack, alpha_bars


# ---------------------------------------------------------------------------
# Streaming CharModel — inference uses the NoProp readout ONLY (Variant A)
# ---------------------------------------------------------------------------

class NoPropCharModel(CharModel):
    def __init__(
        self,
        body: GPTBody,
        np_stack: NoPropTerminalStack,
        alpha_bars: Tensor,
        device: torch.device | None = None,
    ):
        self.body = body
        self.np_stack = np_stack
        self.alpha_bars = alpha_bars
        self.device = device or next(body.parameters()).device
        self.body.eval()
        self.np_stack.eval()
        self._kv: list[tuple[Tensor, Tensor]] | None = None
        self._next_h: Tensor | None = None
        self._pos: int = 0

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = None
        self._pos = 0
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        h, self._kv = self.body.body_hidden(x, None, offset=self._pos)
        self._next_h = h[0, -1]
        self._pos = 1

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        if self._next_h is None:
            raise RuntimeError("predict() called before reset()")
        h = self._next_h.view(1, 1, -1)
        logits = self.np_stack.inference_logits(h, self.alpha_bars).view(-1)
        probs = F.softmax(logits.float(), dim=-1)
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
        if self._kv is None:
            raise RuntimeError("observe() called before reset()")
        for byte in char.encode("utf-8"):
            self._maybe_trim_cache()
            x = torch.tensor([[byte]], dtype=torch.long, device=self.device)
            h, self._kv = self.body.body_hidden(x, self._kv, offset=self._pos)
            self._next_h = h[0, -1]
            self._pos += 1

    def _maybe_trim_cache(self) -> None:
        if self._kv is None:
            return
        cur = self._kv[0][0].shape[2]
        if cur < self.body.max_len:
            return
        keep = self.body.max_len - 1
        self._kv = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in self._kv]


# ---------------------------------------------------------------------------
# Diagnostic: quick val accuracy comparison (SGD head vs NoProp head)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _diag_val_acc(
    body: GPTBody,
    np_stack: NoPropTerminalStack,
    alpha_bars: Tensor,
    valid_text: str,
    device: torch.device,
    n_chars: int = 4000,
    seq_len: int = 1024,
) -> tuple[float, float]:
    """Teacher-forced char-acc on a short val prefix, computed two ways:
      * SGD head (body.proj over h)
      * NoProp head (denoising chain from N(0, I) per position, argmax of
        readout(z_0))
    Returns (sgd_acc, np_acc).
    """
    body.eval()
    np_stack.eval()
    raw = valid_text.encode("utf-8")[: max(seq_len + 1, n_chars + 1)]
    if len(raw) < seq_len + 1:
        return float("nan"), float("nan")
    n_eval = min(n_chars, len(raw) - 1)
    x = torch.tensor(list(raw[:seq_len]), dtype=torch.long, device=device).unsqueeze(0)
    y_full = torch.tensor(list(raw[1:seq_len + 1]), dtype=torch.long, device=device)

    h, _ = body.body_hidden(x)
    h = h[0]
    logits_sgd = body.proj(h).float()
    sgd_pred = logits_sgd.argmax(dim=-1)
    sgd_acc = (sgd_pred[:n_eval] == y_full[:n_eval]).float().mean().item()
    logits_np = np_stack.inference_logits(h.unsqueeze(0), alpha_bars).squeeze(0)
    np_pred = logits_np.argmax(dim=-1)
    np_acc = (np_pred[:n_eval] == y_full[:n_eval]).float().mean().item()
    return sgd_acc, np_acc


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
        print(f"[noprop] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    body, np_stack, alpha_bars = _train_noprop(train_text, cfg, device)

    if valid_text is not None:
        sgd_acc, np_acc = _diag_val_acc(
            body, np_stack, alpha_bars, valid_text, device,
        )
        print(
            f"[noprop] diag val acc (1024-tok teacher-forced): "
            f"sgd={sgd_acc:.4f}  np={np_acc:.4f}  gap={sgd_acc - np_acc:+.4f}",
            flush=True,
        )

    return NoPropCharModel(body, np_stack, alpha_bars, device=device)
