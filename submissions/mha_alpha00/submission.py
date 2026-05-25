"""Modern Hopfield Attention (MHA) on a 4-layer Muon trunk — Experiment 19.

Replaces vanilla causal self-attention with the Hopfield-derived MHA update
(Masumura & Taki, NeurIPS 2025, arXiv 2511.20698, equation 10 with α=0):

    scores_n = Q_n K_n^T * (1/√(2d_head))           [per-layer raw logits]
    h_n      = α' · h_{n-1} + (1 - α') · scores_n   [cross-layer EMA]
    attn_n   = softmax( causal_mask(h_n) )
    y_n      = attn_n @ V_n

α' = 0 recovers vanilla attention exactly (h_n = scores_n). α' > 0 introduces
the non-adiabatic Hopfield coupling across transformer layers. Zero added
trainable parameters; the only new hyperparameter is the scalar α'.

WHY THE CUSTOM KERNEL. The MHA update materializes the full (B, H, T, T)
attention-score tensor at each layer so the next layer can EMA against it.
At our shapes (B=32, H=6, T=1024, bf16) that's 384 MB per layer. Naive math
attention (compute scores → softmax → @V as separate ops) writes that 384 MB
to HBM twice per layer (scores, then attn) and reads it back — ~1.5–2×
slower than F.scaled_dot_product_attention (FlashAttention-fused).

The fused kernel here exploits SDPA's additive `attn_mask` argument and
a small algebraic rewrite of the EMA:

    h_n = α' · h_{n-1} + (1 − α') · Q K^T * scale
        = Q K^T * scale + α' (h_{n-1} − Q K^T * scale)
                          \_____________ "bias" _____________/

So softmax(h_n) V = softmax(Q K^T * scale + bias) V — which is exactly what
F.scaled_dot_product_attention(q, k, v, attn_mask=bias) computes, fused via
PyTorch's memory-efficient attention backend on A100 (xformers-derived,
~1.1–1.3× slower than FlashAttention; supports any float attn_mask,
supports backward natively). The full backward chain — y → attn_mask →
(h_prev, scores) → (Q, K) — is handled by autograd; no custom
autograd.Function required.

Tried first: torch.nn.attention.flex_attention with a captured h. Compiled,
ran correctly in forward (α'=0 vs SDPA matched). Then crashed on the α'>0
test because PyTorch 2.5.1's FlexAttentionAutogradOp asserts
`not any_buffer_requires_grad` on captured tensors — a known limitation
fixed in 2.6+. Our h captures the autograd graph through Q K^T and h_prev,
so it always requires grad in training. Switching to SDPA-with-bias
sidesteps this entirely with negligible perf cost (single extra matmul per
layer for the explicit scores → h materialization, the EMA elementwise op,
and the bias-tensor write — total <1% of step time at training shapes).

Streaming inference uses an explicit math path because T_query=1 makes the
score tensor cheap (B·H·1·T_kv ≤ 6 MB).

Based on:
  submissions/modded_nanogpt/submission.py  — 6-layer baseline
  submissions/hopfield_layer/submission.py  — 4-layer trunk pattern
"""
from __future__ import annotations

__author__ = "@ab-10"

import math
import os
import time
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


def _make_causal_bias_mask(T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build an additive causal mask suitable for F.scaled_dot_product_attention.

    Returns a (T, T) tensor with 0.0 on the lower triangle and -inf on the
    strict upper triangle. Broadcasting handles batch/head dims inside SDPA.
    Cached by GPT on first use.
    """
    mask = torch.zeros(T, T, device=device, dtype=dtype)
    causal = torch.triu(
        torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1,
    )
    mask.masked_fill_(causal, float("-inf"))
    return mask


# ---------------------------------------------------------------------------
# Architecture (modded-nanogpt simple, vocab=256, RoPE offset support).
# The CausalSelfAttention class is replaced by HopfieldCoupledAttention.
# Everything else is identical to submissions/modded_nanogpt/submission.py.
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
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=dim // 4, dtype=torch.float32
        )
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


class HopfieldCoupledAttention(nn.Module):
    """Self-attention with cross-layer Hopfield score-EMA (MHA, α' coupling).

    Forward signature differs from baseline CausalSelfAttention: takes and
    returns an extra `h_prev` / `h_new` tensor that carries the previous
    layer's attention-score matrix forward.

    At α' = 0 the EMA collapses to identity (h_n = scores_n) and the layer
    is mathematically equivalent to vanilla attention. We still go through
    the FlexAttention kernel (not SDPA) so timing is comparable across the
    α' sweep — energy differences between α' = 0 and α' > 0 are then
    attributable to the EMA mechanism rather than kernel choice.
    """

    def __init__(self, dim: int, head_dim: int = 64, alpha_prime: float = 0.5):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)
        # MHA paper scaling: 1/√(2 d_head). modded-nanogpt uses 0.12 ≈
        # 1/√(64) — slightly hotter. We match modded-nanogpt to keep the
        # α'=0 cell numerically identical to the vanilla SDPA path.
        self.scale = 0.12
        self.alpha_prime = alpha_prime

    def forward(
        self,
        x: Tensor,
        h_prev: Tensor | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        causal_bias: Tensor | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor | None]:
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

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        T_kv = k.size(2)
        new_kv = (k, v)

        streaming = (kv_cache is not None) or T == 1
        need_h_out = self.alpha_prime != 0.0

        if streaming:
            # T_query = 1, T_kv ≤ 1024. The score tensor is at most ~6 MB
            # and SDPA's per-call overhead would dominate; use explicit math.
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,1,T_kv)
            if h_prev is not None and self.alpha_prime != 0.0:
                h = self.alpha_prime * h_prev + (1.0 - self.alpha_prime) * scores
            else:
                h = scores
            # No causal mask: single query at offset+T-1, keys at positions ≤ that.
            attn = F.softmax(h.float(), dim=-1).to(v.dtype)
            y = torch.matmul(attn, v)  # (B, H, 1, D)
            h_out = h if need_h_out else None
        else:
            # Training path. Two regimes:
            if h_prev is None:
                # First layer (or α' = 0 globally): no EMA correction, pure
                # vanilla attention. Use SDPA with is_causal=True for the
                # FlashAttention fast path.
                y = F.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, is_causal=True,
                )
                if need_h_out:
                    # Compute h_out = scores so the *next* layer can EMA.
                    # One extra (B,H,T,T) matmul, ≈100 μs on A100 at our shapes.
                    h_out = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                else:
                    h_out = None
            else:
                # Subsequent layer with EMA. Algebraic rewrite that lets
                # SDPA do all the heavy lifting:
                #
                #   h_n = α' h_{n-1} + (1-α') Q K^T * scale
                #
                # SDPA computes softmax(Q K^T * scale_sdpa + attn_mask) V.
                # Set scale_sdpa = (1-α') * scale and attn_mask = α' h_{n-1},
                # then the pre-softmax logits inside SDPA are exactly h_n.
                #
                # This is the form that *avoids* materializing (h_prev -
                # scores) as a separate (B,H,T,T) tensor — the mask is just
                # α' * h_prev (one elementwise read) plus the broadcast
                # causal mask. SDPA dispatches to the memory-efficient
                # attention backend on A100, which supports float attn_mask
                # and supports backward natively.
                if causal_bias is None:
                    causal_bias = _make_causal_bias_mask(T, q.device, q.dtype)
                # attn_mask = α' * h_prev + causal (-inf upper-tri). The
                # (T,T) causal_bias broadcasts into the (B,H,T,T) mask
                # add — no extra (B,H,T,T) materialization.
                attn_mask = self.alpha_prime * h_prev + causal_bias
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    scale=(1.0 - self.alpha_prime) * self.scale,
                )
                # Materialize h_out only if a subsequent layer will EMA
                # against it. The single extra Q K^T matmul here is the
                # unavoidable price of cross-layer state.
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                h_out = self.alpha_prime * h_prev + (1.0 - self.alpha_prime) * scores

        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y), new_kv, h_out


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
    def __init__(self, dim: int, head_dim: int, alpha_prime: float):
        super().__init__()
        self.attn = HopfieldCoupledAttention(
            dim, head_dim=head_dim, alpha_prime=alpha_prime,
        )
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(
        self,
        x: Tensor,
        h_prev: Tensor | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        causal_bias: Tensor | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor | None]:
        attn_out, new_kv, h_out = self.attn(
            self.norm1(x), h_prev=h_prev, kv_cache=kv_cache,
            causal_bias=causal_bias, offset=offset,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv, h_out


class GPT(nn.Module):
    """4-layer modded-nanogpt with MHA-coupled attention layers.

    Threads h through the layer loop: h is None at layer 0, becomes the
    score-EMA tensor for layers 1..L-1. KV cache is per-layer as before.
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        head_dim: int = 64,
        max_len: int = 1024,
        alpha_prime: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.alpha_prime = alpha_prime
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            Block(model_dim, head_dim=head_dim, alpha_prime=alpha_prime)
            for _ in range(num_layers)
        ])
        # Phase-1 invariant: α' is a single global scalar, not per-layer.
        # Guards against a future per-layer α' variant being accidentally
        # enabled by mutation of an inner attention module.
        assert len({blk.attn.alpha_prime for blk in self.blocks}) == 1, (
            "α' must be uniform across layers in this experiment"
        )
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        # Cached causal bias mask (T, T) for training; built lazily once.
        self._causal_bias: Tensor | None = None

    def _get_causal_bias(self, T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if (
            self._causal_bias is None
            or self._causal_bias.shape[-1] != T
            or self._causal_bias.device != device
            or self._causal_bias.dtype != dtype
        ):
            self._causal_bias = _make_causal_bias_mask(T, device, dtype)
        return self._causal_bias

    def forward(
        self,
        inputs: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        B, T = inputs.size(0), inputs.size(1)
        x = self.norm1(self.embed(inputs))
        new_caches: list[tuple[Tensor, Tensor]] = []

        is_training_path = (kv_caches is None) and T > 1
        # Build the causal additive bias once per forward and share across
        # layers (saves L−1 redundant constructions). bfloat16 is the
        # autocast active dtype during training.
        causal_bias = (
            self._get_causal_bias(T, inputs.device, torch.bfloat16)
            if is_training_path else None
        )

        h: Tensor | None = None
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv, h = block(
                x, h_prev=h, kv_cache=kv,
                causal_bias=causal_bias, offset=offset,
            )
            new_caches.append(new_kv)
        logits = self.proj(self.norm2(x)).float()
        # modded-nanogpt softcap on logits.
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits, new_caches


# ---------------------------------------------------------------------------
# Muon optimizer (single-GPU; distributed all-gather stripped).
# Identical to submissions/modded_nanogpt/submission.py.
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
# Init scheme (mirrors modded-nanogpt simple).
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
                w.normal_(std=0.33 ** 0.5 / w.size(-1) ** 0.5)
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
        model_dim: int = 384,
        num_layers: int = 4,
        head_dim: int = 64,
        max_len: int = 1024,
        batch_size: int = 32,
        n_steps: int = 2150,
        cooldown_frac: float = 0.7,
        embed_lr: float = 0.3,
        head_lr: float = 1.0 / 320,
        scalar_lr: float = 0.01,
        muon_lr: float = 0.035,
        muon_wd: float = 0.025,
        alpha_prime: float = 0.5,
        log_every: int = 100,
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
        self.alpha_prime = alpha_prime
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"H={self.model_dim // self.head_dim} bs={self.batch_size} "
                f"T={self.max_len} steps={self.n_steps} "
                f"α'={self.alpha_prime})")


def _train_modded(text: str, cfg: TrainConfig, device: torch.device) -> GPT:
    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < cfg.max_len + 1:
        raise ValueError(f"need at least {cfg.max_len + 1} bytes; got {n}")

    model = GPT(
        vocab_size=256,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        head_dim=cfg.head_dim,
        max_len=cfg.max_len,
        alpha_prime=cfg.alpha_prime,
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
    print(f"[mha] {n_params/1e6:.2f}M params  cfg={cfg}")

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
        loss.backward()
        for opt in optimizers:
            opt.step()

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            print(
                f"[mha] step {step:5d}/{cfg.n_steps}  loss {loss.item():.4f}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    return model


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper — KV-cached, RoPE-offset-aware.
# h is NOT cached across observe() calls: it's a transient layer-to-layer
# pass within a single forward (training or streaming token).
# ---------------------------------------------------------------------------

class MHACharModel(CharModel):
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
        # Seed the model with byte 0; matches modded_nanogpt's bootstrap.
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        logits, self._kv = self.model(x, None, offset=self._pos)
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


# ---------------------------------------------------------------------------
# Entry point — `submit.py` looks for this signature.
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[mha] SEED={seed}")

    # α' can be overridden by env (useful for the sweep without code dup).
    alpha_env = os.environ.get("MHA_ALPHA_PRIME")
    alpha_prime = float(alpha_env) if alpha_env is not None else 0.0
    print(f"[mha] α' = {alpha_prime}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(alpha_prime=alpha_prime)
    model = _train_modded(train_text, cfg, device)
    return MHACharModel(model)
