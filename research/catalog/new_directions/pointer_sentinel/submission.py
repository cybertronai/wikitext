"""Char-level Pointer-Sentinel Mixture LM on the wikitext energy benchmark.

Per research/catalog/new_directions/spec_15_pointer_sentinel_char.md.

Architecture:
  - Parametric backbone: small causal transformer (d=256, L=4, H=4) using
    modded-nanogpt-style RMSNorm / RoPE / CausalSelfAttention / ReLU^2 MLP.
  - Pointer-sentinel module over a buffer of the last W=1024 characters.
    Attention-style scoring of a query (from the backbone's final hidden)
    against W buffered char embeddings plus one learnable sentinel key.
    The softmax mass on the sentinel becomes the mixture gate g; the
    remaining 1-g mass is scatter-added into a 256-slot vocab distribution
    indexed by buffered char IDs. Final output = g * P_vocab + (1-g) * P_ptr.

Training (the engineering crux):
  - Each minibatch sample is a (CTX + T) byte window. The first CTX bytes
    are buffer-only "warm-up" (no loss, no backbone forward); positions
    CTX..CTX+T-1 are the predictive positions and contribute to the loss.
  - At predictive position t (0-indexed within T), the pointer buffer is
    chars [(CTX+t)-W .. (CTX+t)-1], i.e. the W chars immediately before t.
    Causality holds because the buffer ends at (CTX+t)-1 < CTX+t = target's
    position.
  - The backbone runs only over the T predictive positions, with RoPE
    offset CTX so position-encoding semantics are consistent with the
    streaming inference path.

Streaming (CharModel):
  - Backbone runs with KV cache (O(1) marginal cost per byte, as in
    modded-nanogpt/lwta_k2).
  - Rolling W-buffer of (char_id, embed) tuples updated on each observe().
  - predict() runs pointer scoring across the current buffer, mixes with
    the vocab branch, returns the byte distribution.
"""
from __future__ import annotations

__author__ = "@ab-10"

import math
import os
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Architecture (modded-nanogpt-style backbone, smaller)
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


class Backbone(nn.Module):
    """Small causal transformer producing per-position hidden states.

    Returns the pre-final-norm hidden so the pointer-sentinel module can
    consume it directly (the vocab branch's own RMSNorm + linear lives
    inside CharPointerSentinel).
    """
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
        self.model_dim = model_dim
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(
            [Block(model_dim, head_dim=head_dim) for _ in range(num_layers)]
        )
        self.norm_in = RMSNorm(model_dim)
        self.norm_out = RMSNorm(model_dim)

    def forward(
        self,
        inputs: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        x = self.norm_in(self.embed(inputs))
        new_caches: list[tuple[Tensor, Tensor]] = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv, offset=offset)
            new_caches.append(new_kv)
        return self.norm_out(x), new_caches


# ---------------------------------------------------------------------------
# Pointer-Sentinel module
# ---------------------------------------------------------------------------

class CharPointerSentinel(nn.Module):
    """Pointer-sentinel mixture over a buffer of recent char embeddings.

    Two-mode forward:
      * Training (forward_local): given `full_embeds` and `full_ids` over
        a (B, L) context+target stream plus a window length W and offset
        CTX, computes pointer attention over the local W-char window
        ending at each of the last T positions using a single big bmm
        + band-mask, avoiding the (B,T,W,D) materialization.
      * Streaming (forward): given pre-built (B, T, W, D) buffer (small
        T=1 case), runs the original per-position formulation.

    Both paths produce: mixture (B, T, V) summing to 1, and sentinel_mass
    (B, T) — the gate value g.
    """
    def __init__(self, d_model: int, vocab: int = 256):
        super().__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_vocab = Linear(d_model, vocab)
        self.sentinel = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(
        self,
        hidden: Tensor,
        buffer_embeds: Tensor,
        buffer_chars: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Streaming-style path (small T, small W_eff)."""
        B, T, D = hidden.shape
        W = buffer_embeds.size(2)

        q = self.W_q(hidden)
        k = self.W_k(buffer_embeds)

        scale = 1.0 / math.sqrt(D)
        ptr_logits = (q.unsqueeze(-2) * k).sum(-1) * scale
        sent = self.sentinel.to(q.dtype)
        sent_logits = (q * sent).sum(-1, keepdim=True) * scale

        all_logits = torch.cat([ptr_logits, sent_logits], dim=-1)
        attn = all_logits.float().softmax(-1)
        ptr_attn = attn[..., :W]
        sentinel_mass = attn[..., W:W+1]

        p_pointer = torch.zeros(B, T, self.vocab, device=hidden.device, dtype=ptr_attn.dtype)
        p_pointer.scatter_add_(-1, buffer_chars, ptr_attn)

        p_vocab = self.W_vocab(hidden).float().softmax(-1)
        g = sentinel_mass
        mixture = g * p_vocab + p_pointer
        return mixture, sentinel_mass.squeeze(-1)

    def forward_local(
        self,
        hidden: Tensor,
        full_embeds: Tensor,
        full_ids: Tensor,
        ctx_len: int,
        window: int,
    ) -> tuple[Tensor, Tensor]:
        """Training path.

        hidden:      (B, T, D)         — backbone hidden for the last T positions
        full_embeds: (B, L, D)         — char embeddings for CTX+T stream (L = ctx_len+T)
        full_ids:    (B, L) int64      — corresponding char IDs
        ctx_len:     CTX, the number of warm-up positions at the front
        window:      W, pointer buffer length

        At predictive position t (0..T-1), the pointer attends over
        full positions [ctx_len + t - window + 1, ctx_len + t] inclusive.
        That is exactly one of the W slots of a band of width W in the
        (T, L) attention matrix.
        """
        B, T, D = hidden.shape
        L = full_embeds.size(1)
        assert L == ctx_len + T, f"L={L} ctx_len={ctx_len} T={T}"

        q = self.W_q(hidden)                          # (B, T, D)
        K = self.W_k(full_embeds)                     # (B, L, D)

        scale = 1.0 / math.sqrt(D)
        # Pointer logits over the full L-stream.
        ptr_logits = torch.bmm(q, K.transpose(1, 2)) * scale  # (B, T, L)

        # Build the band mask: position (t, j) is valid iff
        #     ctx_len + t - window + 1 <= j <= ctx_len + t.
        # j is global position into full stream.
        device = hidden.device
        t_idx = torch.arange(T, device=device).unsqueeze(1)        # (T, 1)
        j_idx = torch.arange(L, device=device).unsqueeze(0)        # (1, L)
        valid = (j_idx >= ctx_len + t_idx - window + 1) & (j_idx <= ctx_len + t_idx)
        # valid: (T, L); apply to logits.
        ptr_logits = ptr_logits.masked_fill(~valid.unsqueeze(0), float("-inf"))

        # Sentinel logit per query.
        sent = self.sentinel.to(q.dtype)
        sent_logits = (q * sent).sum(-1, keepdim=True) * scale     # (B, T, 1)

        all_logits = torch.cat([ptr_logits, sent_logits], dim=-1)  # (B, T, L+1)
        attn = all_logits.float().softmax(-1)                       # (B, T, L+1)
        ptr_attn = attn[..., :L]                                    # (B, T, L)
        sentinel_mass = attn[..., L:L+1]                            # (B, T, 1)

        # Scatter pointer mass into vocab slots, using full_ids broadcast.
        # full_ids: (B, L) → (B, T, L) by expansion.
        ids_btL = full_ids.unsqueeze(1).expand(B, T, L)             # (B, T, L)
        p_pointer = torch.zeros(B, T, self.vocab, device=device, dtype=ptr_attn.dtype)
        p_pointer.scatter_add_(-1, ids_btL, ptr_attn)               # (B, T, V)

        p_vocab = self.W_vocab(hidden).float().softmax(-1)          # (B, T, V)
        g = sentinel_mass
        mixture = g * p_vocab + p_pointer
        return mixture, sentinel_mass.squeeze(-1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PointerSentinelLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        num_layers: int = 4,
        model_dim: int = 256,
        head_dim: int = 64,
        max_len: int = 1024,
        buffer_len: int = 1024,
    ):
        super().__init__()
        self.backbone = Backbone(
            vocab_size=vocab_size,
            num_layers=num_layers,
            model_dim=model_dim,
            head_dim=head_dim,
            max_len=max_len,
        )
        self.pointer = CharPointerSentinel(d_model=model_dim, vocab=vocab_size)
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_len = max_len
        self.buffer_len = buffer_len

    def embed_chars(self, ids: Tensor) -> Tensor:
        """Look up the backbone's char embedding table."""
        return self.backbone.embed(ids)

    def forward_train(
        self,
        ctx_ids: Tensor,
        tgt_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Training forward.

        Inputs:
          ctx_ids: (B, CTX) — pointer-only warm-up chars
          tgt_ids: (B, T)   — predictive chars (backbone consumes these
                              with RoPE offset = CTX; pointer buffer at
                              local position t covers global chars
                              [CTX+t-W+1, CTX+t].)

        Returns:
          mixture:        (B, T, V) per-position next-char probabilities
                          (summing to 1.0 along V)
          sentinel_mass:  (B, T) per-position gate value g.
        """
        B, CTX = ctx_ids.shape
        T = tgt_ids.size(1)
        W = self.buffer_len

        full_ids = torch.cat([ctx_ids, tgt_ids], dim=1)                # (B, CTX+T)
        full_embeds = self.embed_chars(full_ids)                       # (B, CTX+T, D)

        # Backbone runs only over the T predictive positions, with RoPE
        # offset = CTX so positions are consistent with the streaming
        # inference path.
        hidden, _ = self.backbone(tgt_ids, kv_caches=None, offset=CTX)  # (B, T, D)

        mixture, sentinel_mass = self.pointer.forward_local(
            hidden, full_embeds, full_ids, ctx_len=CTX, window=W,
        )
        return mixture, sentinel_mass


# ---------------------------------------------------------------------------
# Muon optimizer (single-GPU; same as modded-nanogpt port)
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
# Init
# ---------------------------------------------------------------------------

def _init_pointer_sentinel(model: PointerSentinelLM) -> None:
    for name, p in model.named_parameters():
        w = p.data
        if name == "pointer.sentinel":
            w.normal_(mean=0.0, std=0.02)
            continue
        if name.endswith("weight"):
            if "proj" in name or "W_vocab" in name:
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
        # Per spec Remediation B: enlarge backbone to d=384/L=6 because
        # the d=256/L=4 backbone hit val_acc=0.6538 (below floor 0.70)
        # with 160s training budget unused. The pointer module is kept
        # at W=1024.
        model_dim: int = 384,
        num_layers: int = 6,
        head_dim: int = 64,
        max_len: int = 512,         # T: number of predictive positions per sample
        buffer_len: int = 1024,     # W: pointer buffer length
        batch_size: int = 24,
        n_steps: int = 2000,
        cooldown_frac: float = 0.7,
        embed_lr: float = 0.3,
        head_lr: float = 1.0 / 320,
        scalar_lr: float = 0.01,
        muon_lr: float = 0.035,
        muon_wd: float = 0.025,
        log_every: int = 100,
    ):
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.max_len = max_len
        self.buffer_len = buffer_len
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
                f"T={self.max_len} W={self.buffer_len} steps={self.n_steps})")


def _train_pointer_sentinel(
    text: str,
    cfg: TrainConfig,
    device: torch.device,
) -> PointerSentinelLM:
    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    sample_len = cfg.buffer_len + cfg.max_len  # CTX + T
    if n < sample_len + 1:
        raise ValueError(f"need at least {sample_len+1} bytes; got {n}")

    model = PointerSentinelLM(
        vocab_size=256,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        head_dim=cfg.head_dim,
        max_len=cfg.max_len,
        buffer_len=cfg.buffer_len,
    ).to(device)
    _init_pointer_sentinel(model)

    # Build optimizer param groups, ensuring no parameter appears twice.
    # Muon gets all 2-D block weights (backbone blocks + pointer Q/K projs).
    # AdamW gets:
    #   - embed weight                 (embed_lr)
    #   - vocab head weight + bias     (head_lr)
    #   - everything else 1-D / sentinel (scalar_lr)
    backbone_2d = [p for p in model.backbone.blocks.parameters() if p.ndim >= 2]
    pointer_2d = [model.pointer.W_q.weight, model.pointer.W_k.weight]
    block_2d = backbone_2d + pointer_2d

    embed_params = [model.backbone.embed.weight]
    head_params = [model.pointer.W_vocab.weight, model.pointer.W_vocab.bias]
    claimed_ids = {id(p) for p in block_2d + embed_params + head_params}

    scalar_params = [
        p for p in model.parameters() if id(p) not in claimed_ids
    ]

    optimizer1 = AdamW(
        [
            dict(params=embed_params, lr=cfg.embed_lr),
            dict(params=head_params, lr=cfg.head_lr),
            dict(params=scalar_params, lr=cfg.scalar_lr),
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
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    n_pointer = sum(p.numel() for p in model.pointer.parameters())
    print(f"[ptrsen] {n_params/1e6:.2f}M params "
          f"(backbone={n_backbone/1e6:.2f}M  pointer={n_pointer/1e6:.2f}M) "
          f"cfg={cfg}", flush=True)

    def set_lr(progress: float) -> None:
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
    # Defensive wall-clock budget: stop training in time to return cleanly
    # before the harness's 300s SIGALRM. Schedule progress is driven by
    # `elapsed / train_budget_s` so the cooldown phase always kicks in,
    # regardless of how many steps we end up doing.
    train_budget_s = 265.0
    for step in range(cfg.n_steps):
        elapsed = time.monotonic() - t0
        if elapsed > train_budget_s:
            print(f"[ptrsen] stopping early at step {step} "
                  f"(elapsed {elapsed:.0f}s > {train_budget_s:.0f}s)",
                  flush=True)
            break
        progress = max(step / cfg.n_steps, elapsed / train_budget_s)
        set_lr(progress)
        # Sample (CTX + T + 1)-byte windows: first CTX bytes are pointer
        # warm-up, the next T are inputs to the backbone, and the very
        # last byte is unused (we predict targets shifted by one).
        # Actually: input to backbone at position t is the char at offset
        # (CTX+t); we predict char at offset (CTX+t+1). So we need length
        # sample_len + 1 = CTX + T + 1 bytes.
        idx = torch.randint(0, n - sample_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(sample_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()              # (B, CTX+T+1)
        ctx_ids = flat[:, :cfg.buffer_len]              # (B, CTX)
        tgt_in  = flat[:, cfg.buffer_len:cfg.buffer_len + cfg.max_len]  # (B, T)
        tgt_out = flat[:, cfg.buffer_len + 1:cfg.buffer_len + 1 + cfg.max_len]  # (B, T)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mixture, _ = model.forward_train(ctx_ids, tgt_in)
                # mixture is a probability tensor; use NLL on log of it.
                # Use small eps to avoid log(0) in the rare case where the
                # pointer mass on the true char is exactly 0.
                log_p = torch.log(mixture.clamp_min(1e-10))
                loss = F.nll_loss(log_p.reshape(-1, 256), tgt_out.reshape(-1))
        else:
            mixture, _ = model.forward_train(ctx_ids, tgt_in)
            log_p = torch.log(mixture.clamp_min(1e-10))
            loss = F.nll_loss(log_p.reshape(-1, 256), tgt_out.reshape(-1))
        loss.backward()
        for opt in optimizers:
            opt.step()

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            print(
                f"[ptrsen] step {step:5d}/{cfg.n_steps}  "
                f"loss {loss.item():.4f}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    return model


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper
# ---------------------------------------------------------------------------

class PointerSentinelCharModel(CharModel):
    """Streaming wrapper.

    State:
      _kv:         per-layer KV cache for the backbone (KV-cached attention)
      _hidden:     final hidden state for the next predict() call
      _buf_ids:    deque of last W observed char IDs (int)
      _buf_emb:    deque of last W observed char embeddings (D-dim tensors)
      _pos:        absolute position counter (RoPE offset)
      _ptrs_sum, _ptrs_n: running sentinel-mass statistics (diagnostic).
    """
    def __init__(self, model: PointerSentinelLM, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self.W = model.buffer_len
        self.D = model.model_dim
        self.V = model.vocab_size
        self._kv: list[tuple[Tensor, Tensor]] | None = None
        self._hidden: Tensor | None = None
        self._buf_ids: deque[int] = deque(maxlen=self.W)
        self._buf_emb: deque[Tensor] = deque(maxlen=self.W)
        self._pos: int = 0
        # Diagnostic: mean sentinel mass.
        self._ptrs_sum: float = 0.0
        self._ptrs_n: int = 0

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = None
        self._hidden = None
        self._buf_ids.clear()
        self._buf_emb.clear()
        self._pos = 0
        # Seed with one zero byte so predict() always has a valid hidden.
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        hidden, self._kv = self.model.backbone(x, None, offset=self._pos)
        self._hidden = hidden[:, -1:, :]  # (1, 1, D)
        # The seed byte goes into the pointer buffer too — it's "observed"
        # context the same way the backbone treats it.
        emb = self.model.embed_chars(x)[:, -1, :]  # (1, D)
        self._buf_ids.append(0)
        self._buf_emb.append(emb.squeeze(0))
        self._pos = 1

    @torch.no_grad()
    def _mixture_from_hidden(self) -> Tensor:
        """Return P(next char) as a (256,) probability tensor."""
        assert self._hidden is not None
        hidden = self._hidden  # (1, 1, D)
        if len(self._buf_ids) == 0:
            # Should not happen post-reset, but degrade to vocab-only.
            p_vocab = self.model.pointer.W_vocab(hidden).float().softmax(-1)
            return p_vocab.squeeze(0).squeeze(0)

        # Build (1, 1, W_eff, D) buffer-embed tensor and (1, 1, W_eff)
        # buffer-id tensor from the deques. W_eff = current buffer size.
        buf_emb = torch.stack(list(self._buf_emb), dim=0)  # (W_eff, D)
        buf_ids = torch.tensor(list(self._buf_ids), dtype=torch.long, device=self.device)
        buffer_embeds = buf_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, W_eff, D)
        buffer_chars  = buf_ids.unsqueeze(0).unsqueeze(0)  # (1, 1, W_eff)

        mixture, sentinel_mass = self.model.pointer(hidden, buffer_embeds, buffer_chars)
        self._ptrs_sum += float(sentinel_mass.squeeze().item())
        self._ptrs_n += 1
        return mixture.squeeze(0).squeeze(0)  # (V,)

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        if self._hidden is None:
            raise RuntimeError("predict() called before reset()")
        probs = self._mixture_from_hidden()
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
            hidden, self._kv = self.model.backbone(x, self._kv, offset=self._pos)
            self._hidden = hidden[:, -1:, :]
            emb = self.model.embed_chars(x)[:, -1, :].squeeze(0)  # (D,)
            self._buf_ids.append(byte)
            self._buf_emb.append(emb)
            self._pos += 1

    def _maybe_trim_cache(self) -> None:
        if self._kv is None:
            return
        cur = self._kv[0][0].shape[2]
        if cur < self.model.max_len:
            return
        keep = self.model.max_len - 1
        self._kv = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in self._kv]

    def sentinel_mass_mean(self) -> float:
        return self._ptrs_sum / max(1, self._ptrs_n)


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
        print(f"[ptrsen] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    model = _train_pointer_sentinel(train_text, cfg, device)
    return PointerSentinelCharModel(model)
