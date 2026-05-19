"""Schmidhuber hierarchical surprise-gated chunker.

Per research/catalog/new_directions/spec_16_chunker.md.

Two-level architecture:

  Automatizer L (2-layer, d=128, heads=4, seq_len=512):
    Byte-level autoregressive transformer. Trained with cross-entropy on
    every position. Provides per-position P_L(true_byte) used to gate
    "surprise" events.

  Surprise gate (tau=0.1):
    Position t is a surprise iff P_L(true_byte_t | context) < tau.

  Chunker H (6-layer, d=384, heads=6):
    Larger autoregressive transformer. Consumes ONLY the surprise byte
    embeddings, with sinusoidal positional encoding indexed by ORIGINAL
    byte position (not the surprise-stream index). Predicts the next
    surprise byte. Trained with cross-entropy at next-surprise targets
    only.

Joint training: loss = loss_L + alpha * loss_H, alpha=1.0. Muon for 2-D
weights, AdamW for 1-D / embeddings.

Streaming inference:
  predict():
    run L on current 512-byte buffer -> P_L.
    if max(P_L) >= 1 - tau: return argmax(P_L)  (L is confident; fast)
    else: run H on the last K=32 surprise bytes -> return argmax(P_H).
  observe(c):
    push c to L's 512-byte buffer.
    compute P_L(c); if < tau, also push (c, position) to H's buffer.

D1 (Phase 0 diagnostic) reported p_s(tau=0.1) = 0.267 after 60s training
of L alone, well below the 0.50 pass threshold. Phase 1 proceeds with
default tau=0.1.
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
# Shared primitives — RMSNorm, RoPE, attention, MLP (modded-nanogpt style)
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
    """Half-truncate RoPE, base=1024. Supports a per-row absolute-position
    offset (used by H's variable surprise-stream positions).
    """
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
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
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
        y = F.scaled_dot_product_attention(q, k, v, scale=0.12, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)


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

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Automatizer L — RoPE-based (positions are dense and contiguous)
# ---------------------------------------------------------------------------

class AutomatizerL(nn.Module):
    """Tiny byte-LM, 2-layer d=128 heads=4, seq_len=512."""
    def __init__(self, vocab_size: int = 256, num_layers: int = 2,
                 model_dim: int = 128, head_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, head_dim=head_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.norm1(self.embed(inputs))
        for blk in self.blocks:
            x = blk(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits

    def embed_only(self, inputs: Tensor) -> Tensor:
        """Return token embeddings (for sharing with H if desired)."""
        return self.embed(inputs)


# ---------------------------------------------------------------------------
# Chunker H — operates on surprise byte embeddings with sinusoidal POSITIONAL
#             encoding indexed by ORIGINAL byte position. Standard attention
#             (no RoPE on positions because positions are irregular).
# ---------------------------------------------------------------------------

class ChunkerAttention(nn.Module):
    """Plain causal self-attention (no RoPE); positions are baked into x
    by the sinusoidal positional encoding added to the embedding.
    """
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # If attn_mask provided, it's (B, T, T) additive (-inf for blocked).
        # Otherwise use is_causal=True via SDPA.
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, scale=0.12, is_causal=True)
        else:
            # Combine attn_mask with causal mask by expanding to (B, 1, T, T)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask.unsqueeze(1), scale=0.12, is_causal=False
            )
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)


class ChunkerBlock(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.attn = ChunkerAttention(dim, head_dim=head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


def _sinusoidal_position_encoding(positions: Tensor, dim: int) -> Tensor:
    """positions: (B, T) long tensor of absolute byte indices.
    Returns (B, T, dim) sinusoidal encoding in float32.
    """
    # Use float32 internally; caller casts to model dtype.
    device = positions.device
    half = dim // 2
    # frequencies: 1 / 10000^(2i/dim)
    freqs = torch.exp(
        torch.arange(0, half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, half))
    )  # (half,)
    pos = positions.float()  # (B, T)
    args = pos.unsqueeze(-1) * freqs  # (B, T, half)
    sin = args.sin()
    cos = args.cos()
    pe = torch.cat([sin, cos], dim=-1)  # (B, T, 2*half)
    if pe.size(-1) < dim:  # odd dim safety
        pe = F.pad(pe, (0, dim - pe.size(-1)))
    return pe


class ChunkerH(nn.Module):
    """6-layer transformer, d=384, heads=6 (head_dim=64), causal."""
    def __init__(self, vocab_size: int = 256, num_layers: int = 6,
                 model_dim: int = 384, head_dim: int = 64):
        super().__init__()
        self.model_dim = model_dim
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            ChunkerBlock(model_dim, head_dim=head_dim) for _ in range(num_layers)
        ])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(
        self,
        bytes_in: Tensor,
        positions: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """bytes_in: (B, T) long
        positions: (B, T) long, absolute byte indices in original stream
        attn_mask: optional (B, T, T) additive float mask (-inf to block).
        """
        pe = _sinusoidal_position_encoding(positions, self.model_dim).type_as(self.embed.weight)
        x = self.embed(bytes_in) + pe
        x = self.norm1(x)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits


# ---------------------------------------------------------------------------
# Muon optimizer (single-GPU)
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


def _init_modded(module: nn.Module) -> None:
    for name, p in module.named_parameters():
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
# Joint chunker module
# ---------------------------------------------------------------------------

class Chunker(nn.Module):
    """L + H bundled. forward() returns both losses + diagnostics."""
    def __init__(self, tau: float = 0.1, max_surprise_ctx: int = 64):
        super().__init__()
        self.tau = tau
        self.max_surprise_ctx = max_surprise_ctx
        self.L = AutomatizerL(
            vocab_size=256, num_layers=2, model_dim=128, head_dim=32,
        )
        self.H = ChunkerH(
            vocab_size=256, num_layers=6, model_dim=384, head_dim=64,
        )

    def forward_train(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, dict]:
        """Joint forward.

        x: (B, T) long input bytes
        y: (B, T) long target bytes (shifted by 1)

        Returns (loss_L, loss_H, info).
        """
        B, T = x.shape
        device = x.device
        logits_L = self.L(x)  # (B, T, 256)
        loss_L = F.cross_entropy(
            logits_L.reshape(-1, 256), y.reshape(-1), reduction="mean"
        )

        # Per-position P_L(true_byte). We use logits_L pre-true-byte.
        # Note: logits_L[:, t] predicts y[:, t]. P_L(y_t) = softmax(logits_L[:,t])[y_t].
        with torch.no_grad():
            probs_L = F.softmax(logits_L.float(), dim=-1)
            p_true = probs_L.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, T)
            surprise_mask = (p_true < self.tau)  # (B, T) bool
            n_surprise = surprise_mask.sum().item()
            n_total = surprise_mask.numel()

        # Build H's inputs. For each batch row, gather (surprise_bytes,
        # surprise_positions) chronologically, truncated to max_surprise_ctx.
        # H predicts the next-surprise byte; its target at position k is the
        # surprise byte at chronological index k+1 in the row.
        loss_H_total = torch.zeros((), device=device, dtype=torch.float32)
        n_pred = 0
        K = self.max_surprise_ctx
        # We pack a fixed (B, K) buffer using only the last K surprises in each row.
        # bytes_buf and pos_buf get padded with 0 + position=0; attention mask blocks pads.
        bytes_buf = torch.zeros((B, K), dtype=torch.long, device=device)
        pos_buf = torch.zeros((B, K), dtype=torch.long, device=device)
        valid = torch.zeros((B, K), dtype=torch.bool, device=device)
        targets = torch.zeros((B, K), dtype=torch.long, device=device)
        target_valid = torch.zeros((B, K), dtype=torch.bool, device=device)

        sm = surprise_mask
        for b in range(B):
            idx = sm[b].nonzero(as_tuple=False).flatten()  # positions where surprise
            if idx.numel() < 2:
                continue
            # Take last K+1 surprises (the last surprise is the target, we use up
            # to K context bytes preceding it).
            keep = idx[-(K + 1):]
            ctx = keep[:-1]      # context (length up to K)
            tgt_pos = keep[1:]   # target positions (the next-surprise byte at each step)
            n_ctx = ctx.numel()
            bytes_buf[b, :n_ctx] = x[b, ctx]  # input byte at ctx position
            # Actually: H consumes the SURPRISE byte. Surprise byte at position p is
            # the byte that L *should have predicted* there — i.e. y[b, p] (true byte
            # at position p). Equivalently x[b, p+1] for shifted streams; here y is
            # already the true bytes at each position. Use y[b, ctx].
            bytes_buf[b, :n_ctx] = y[b, ctx]
            pos_buf[b, :n_ctx] = ctx
            valid[b, :n_ctx] = True
            targets[b, :n_ctx] = y[b, tgt_pos]
            target_valid[b, :n_ctx] = True

        if valid.any():
            # Build attention mask: row r in step k is allowed to attend to step j
            # iff j <= k AND valid[b, j]. Causal + pad-mask.
            T2 = K
            causal = torch.tril(torch.ones(T2, T2, device=device, dtype=torch.bool))[None]
            pad_mask = valid.unsqueeze(1).expand(B, T2, T2)  # allow attending to valid keys
            mask_bool = causal & pad_mask
            # additive: 0 where allowed, -inf where blocked
            attn_mask = torch.zeros((B, T2, T2), device=device, dtype=torch.float32)
            attn_mask = attn_mask.masked_fill(~mask_bool, float("-inf"))

            logits_H = self.H(bytes_buf, pos_buf, attn_mask=attn_mask)  # (B, K, 256)
            # CE only at positions where target_valid is True
            flat_logits = logits_H.reshape(-1, 256)
            flat_targets = targets.reshape(-1)
            flat_mask = target_valid.reshape(-1)
            if flat_mask.any():
                loss_H_total = F.cross_entropy(
                    flat_logits[flat_mask], flat_targets[flat_mask], reduction="mean"
                )
                n_pred = int(flat_mask.sum().item())

        info = {
            "n_surprise": n_surprise,
            "n_total": n_total,
            "surprise_rate": float(n_surprise) / max(1, n_total),
            "n_pred_H": n_pred,
        }
        return loss_L, loss_H_total, info


# ---------------------------------------------------------------------------
# Streaming CharModel — runs L every byte; runs H only when L is unsure.
# ---------------------------------------------------------------------------

class ChunkerCharModel(CharModel):
    """Streaming inference per spec §"Streaming inference".

    Maintains:
      - last 512 bytes (L's context), a deque-like tensor.
      - last K surprise bytes + their absolute positions (H's context).
      - absolute byte position counter.

    For inference we re-run L on its full context buffer for each call.
    This is simple and correct (no KV-cache management across two models);
    the 512-token L forward is cheap (2-layer d=128) and inference is not
    energy-counted.
    """
    def __init__(
        self,
        model: Chunker,
        device: torch.device | None = None,
        max_surprise_ctx: int = 32,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self.tau = float(model.tau)
        self.fast_thresh = 1.0 - self.tau
        self.L_ctx_len = 512
        self.K = max_surprise_ctx

        # buffers
        self._L_buf: list[int] = []
        self._surp_bytes: list[int] = []
        self._surp_positions: list[int] = []
        self._pos: int = 0  # absolute position in the eval stream
        self._next_logits: Tensor | None = None

        # per-mode counters
        self.stats = {
            "n_predict": 0,
            "n_predict_L": 0,
            "n_predict_H": 0,
            "n_observe": 0,
            "n_surprise_observed": 0,
        }

    def reset(self) -> None:
        self._L_buf = [0]  # stream-start sentinel byte (matches modded baseline)
        self._surp_bytes = []
        self._surp_positions = []
        self._pos = 1
        self._next_logits = None
        self.stats = {
            "n_predict": 0,
            "n_predict_L": 0,
            "n_predict_H": 0,
            "n_observe": 0,
            "n_surprise_observed": 0,
        }
        # Pre-compute logits for position 1 by feeding the sentinel.
        self._run_L_forward()

    @torch.no_grad()
    def _run_L_forward(self) -> Tensor:
        ctx = self._L_buf[-self.L_ctx_len:]
        x = torch.tensor([ctx], dtype=torch.long, device=self.device)
        logits = self.model.L(x)  # (1, T, 256)
        return logits[0, -1]  # (256,)

    @torch.no_grad()
    def _run_H_forward(self) -> Tensor:
        if not self._surp_bytes:
            # No surprise history yet — fall back to uniform over bytes.
            return torch.zeros(256, device=self.device)
        ctx_b = self._surp_bytes[-self.K:]
        ctx_p = self._surp_positions[-self.K:]
        bytes_in = torch.tensor([ctx_b], dtype=torch.long, device=self.device)
        pos_in = torch.tensor([ctx_p], dtype=torch.long, device=self.device)
        # No padding required; the entire row is valid.
        logits = self.model.H(bytes_in, pos_in)  # (1, T, 256)
        return logits[0, -1]

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        self.stats["n_predict"] += 1
        L_logits = self._run_L_forward()
        L_probs = F.softmax(L_logits.float(), dim=-1)
        max_p, argmax_id = L_probs.max(dim=-1)
        if max_p.item() >= self.fast_thresh:
            self.stats["n_predict_L"] += 1
            # Return L's distribution.
            probs = L_probs
            self._last_pred_mode = "L"
        else:
            self.stats["n_predict_H"] += 1
            H_logits = self._run_H_forward()
            probs = F.softmax(H_logits.float(), dim=-1)
            self._last_pred_mode = "H"
        # Cache L_logits for surprise-gate evaluation in observe().
        self._last_L_probs = L_probs
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
        self.stats["n_observe"] += 1
        for byte in char.encode("utf-8"):
            # Determine if THIS byte is a surprise under L's last prediction.
            # If predict() was not called first (shouldn't happen given API),
            # compute L on the fly.
            if not hasattr(self, "_last_L_probs"):
                _ = self._run_L_forward()
                self._last_L_probs = F.softmax(_.float(), dim=-1)
            p_true = float(self._last_L_probs[byte].item())
            if p_true < self.tau:
                self.stats["n_surprise_observed"] += 1
                self._surp_bytes.append(byte)
                self._surp_positions.append(self._pos)
            self._L_buf.append(byte)
            self._pos += 1
            # Force re-derivation for the next predict() call.
            if hasattr(self, "_last_L_probs"):
                del self._last_L_probs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainConfig:
    def __init__(
        self,
        tau: float = 0.1,
        max_surprise_ctx: int = 64,
        seq_len: int = 512,
        batch_size: int = 32,
        n_steps: int = 2150,
        cooldown_frac: float = 0.7,
        embed_lr: float = 0.3,
        head_lr: float = 1.0 / 320,
        scalar_lr: float = 0.01,
        muon_lr: float = 0.035,
        muon_wd: float = 0.025,
        alpha_H: float = 1.0,
        log_every: int = 50,
    ):
        self.tau = tau
        self.max_surprise_ctx = max_surprise_ctx
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.cooldown_frac = cooldown_frac
        self.embed_lr = embed_lr
        self.head_lr = head_lr
        self.scalar_lr = scalar_lr
        self.muon_lr = muon_lr
        self.muon_wd = muon_wd
        self.alpha_H = alpha_H
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(tau={self.tau} K={self.max_surprise_ctx} "
                f"T={self.seq_len} bs={self.batch_size} steps={self.n_steps} "
                f"alpha_H={self.alpha_H})")


def _train_chunker(text: str, cfg: TrainConfig, device: torch.device) -> Chunker:
    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()

    model = Chunker(tau=cfg.tau, max_surprise_ctx=cfg.max_surprise_ctx).to(device)
    _init_modded(model.L)
    _init_modded(model.H)

    n_params_L = sum(p.numel() for p in model.L.parameters())
    n_params_H = sum(p.numel() for p in model.H.parameters())
    n_params_total = n_params_L + n_params_H
    print(f"[chunker] L params: {n_params_L/1e6:.3f}M  "
          f"H params: {n_params_H/1e6:.3f}M  "
          f"total: {n_params_total/1e6:.3f}M  "
          f"cfg={cfg}")

    # 2-D weights -> Muon; scalars + embeddings + head -> AdamW
    block_2d = (
        [p for p in model.L.blocks.parameters() if p.ndim >= 2]
        + [p for p in model.H.blocks.parameters() if p.ndim >= 2]
    )
    scalars = [p for p in model.parameters() if p.ndim < 2]
    opt_adam = AdamW(
        [
            dict(params=[model.L.embed.weight, model.H.embed.weight], lr=cfg.embed_lr),
            dict(params=[model.L.proj.weight, model.H.proj.weight], lr=cfg.head_lr),
            dict(params=scalars, lr=cfg.scalar_lr),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    opt_muon = Muon(block_2d, lr=cfg.muon_lr, weight_decay=cfg.muon_wd)
    optimizers = [opt_adam, opt_muon]
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

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
    running_surprise = 0.0
    running_n = 0
    last_loss_L = float("nan")
    last_loss_H = float("nan")
    for step in range(cfg.n_steps):
        set_lr(step)
        idx = torch.randint(0, n - cfg.seq_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(cfg.seq_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_L, loss_H, info = model.forward_train(x, y)
                total = loss_L + cfg.alpha_H * loss_H
        else:
            loss_L, loss_H, info = model.forward_train(x, y)
            total = loss_L + cfg.alpha_H * loss_H

        total.backward()
        for opt in optimizers:
            opt.step()

        last_loss_L = float(loss_L.detach().item())
        last_loss_H = float(loss_H.detach().item()) if isinstance(loss_H, Tensor) else 0.0
        running_surprise += info["surprise_rate"]
        running_n += 1

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            avg_sr = running_surprise / max(1, running_n)
            print(f"[chunker] step {step:5d}/{cfg.n_steps}  "
                  f"loss_L {last_loss_L:.4f}  loss_H {last_loss_H:.4f}  "
                  f"p_s {info['surprise_rate']:.3f} (avg {avg_sr:.3f})  "
                  f"n_pred_H {info['n_pred_H']}  "
                  f"elapsed {elapsed:.0f}s", flush=True)

    elapsed = time.monotonic() - t0
    print(f"[chunker] trained {cfg.n_steps} steps in {elapsed:.0f}s; "
          f"final loss_L={last_loss_L:.4f} loss_H={last_loss_H:.4f}; "
          f"avg surprise rate {running_surprise/max(1,running_n):.3f}")

    return model


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
        print(f"[chunker] SEED={seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    model = _train_chunker(train_text, cfg, device)
    char_model = ChunkerCharModel(model, max_surprise_ctx=32)
    # Hand off the live diagnostic counters by printing after eval (not here).
    print(f"[chunker] training complete; built ChunkerCharModel "
          f"(tau={char_model.tau}, K={char_model.K}). "
          f"Per-mode stats will accumulate during evaluation.")
    return char_model
