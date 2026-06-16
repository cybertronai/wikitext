"""modded-nanogpt + isotropic Compositional Muon on attention OV pairs (MP=1).

This forks ``modded_nanogpt`` and replaces ordinary Muon on each attention
``V``/``O`` pair with Tilde Research's cheap isotropic Compositional Muon (CM)
OV rule. CM treats the loss-visible circuit as the composed map ``W_O W_V``
rather than two unrelated matrices: V gets scaled by the current O partner norm,
and O gets a per-head partner-scaled full-matrix spectral sign. Q/K and MLP
matrices stay on ordinary Muon; embeddings/head/scalars stay on AdamW.

Reference: https://blog.tilderesearch.com/blog/compositional-muon

----------------------------------------------------------------------------
Port of KellerJordan/modded-nanogpt to the wikitext byte-level benchmark.

Source:
  https://github.com/KellerJordan/modded-nanogpt
  records/track_3_optimization/train_gpt_simple.py

Adaptations from upstream:
  * vocab_size: 50304 (GPT-2 BPE) -> 256 (raw bytes)
  * single-GPU: stripped torch.distributed all-gather from Muon
  * streaming inference: Rotary takes a position offset; CausalSelfAttention
    optionally consumes/produces a per-layer KV cache; CharModel wrapper
    drives both at O(1) marginal cost per byte.
  * forward signature: returns logits (no targets baked in) so the same
    module serves training (full prefix) and streaming (T=1 with cache).

Training tricks ported verbatim:
  * Muon optimizer (Newton-Schulz orthogonalized momentum) for 2-D block
    weights; AdamW for embeddings, lm_head, and 1-D scalars.
  * Half-truncate RoPE with base-freq tuning (base=1024, second half of
    angular_freq zeroed).
  * QK RMSNorm before RoPE; attention scale 0.12.
  * ReLU^2 MLP, RMSNorm pre-norm, soft-capped logits (cap=15).
  * Init scheme: zero proj/lm_head, default normal embed, scaled normal
    others; RMSNorm gains -> 1; biases -> 0.
  * "Stable then decay" LR schedule (cooldown_frac=0.7).
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
# Architecture (modded-nanogpt simple, vocab_size=256, RoPE offset support)
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
    """Half-truncate RoPE with base-freq tuning (base=1024).

    The second half of the head dimension is left unrotated (angular_freq
    zeroed). Accepts an absolute-position offset so streaming inference
    rotates the new query at its true position rather than always at 0.
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

        # (B, T, H, D) -> (B, H, T, D) for SDPA
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
# Muon optimizer (single-GPU; distributed all-gather stripped)
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


_POLAR_EXPRESS_COEFFS = (
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
)


def _msign_polar_express(G: Tensor, eps: float = 1e-7) -> Tensor:
    """Spectral sign used by Tilde's Compositional Muon release."""
    assert G.ndim >= 2
    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    X = X.float() / X.float().norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    X = X.bfloat16()
    fused = torch.baddbmm if X.ndim == 3 else torch.addmm
    for a, b, c in _POLAR_EXPRESS_COEFFS:
        A = X @ X.mT
        B = fused(A, A, A, alpha=c, beta=b)
        X = fused(X, B, X, alpha=1.0, beta=a)
    X = X.float()
    return X.mT if transpose else X


def _to_heads_row(W: Tensor, n_heads: int, head_dim: int) -> Tensor:
    """Split math-convention ``(d_model, n_heads * head_dim)`` into heads."""
    d = W.shape[0]
    return W.view(d, n_heads, head_dim).transpose(0, 1).contiguous()


def _cm_ov_isotropic_delta(
    W_V: Tensor,
    W_O: Tensor,
    G_V: Tensor,
    G_O: Tensor,
    head_dim: int,
    damping: float,
) -> tuple[Tensor, Tensor]:
    """Tilde isotropic CM OV direction for PyTorch Linear weights.

    Inputs use PyTorch convention: ``W_V`` is ``(d_v, d_model)`` and ``W_O``
    is ``(d_model, d_v)``. Returned directions match those shapes.
    """
    Wv = W_V.mT.float()
    Wo = W_O.mT.float()
    Gv = G_V.mT.float()
    Go = G_O.mT.float()
    d, d_v = Wv.shape
    n_heads = d_v // head_dim

    Wv_h = _to_heads_row(Wv, n_heads, head_dim)
    Gv_h = _to_heads_row(Gv, n_heads, head_dim)
    Wo_h = Wo.view(n_heads, head_dim, d)
    Go_h = Go.view(n_heads, head_dim, d)

    s_v = (Wv_h.square().sum(dim=(-2, -1)) / head_dim + damping).clamp_min(1e-12).rsqrt()
    s_o = (Wo_h.square().sum(dim=(-2, -1)) / head_dim + damping).clamp_min(1e-12).rsqrt()

    m_v = _msign_polar_express(Gv_h)
    delta_v_h = s_o[:, None, None] * m_v

    go_in = s_v[:, None, None] * Go_h
    m_o = _msign_polar_express(go_in.reshape(d_v, d)).view(n_heads, head_dim, d)
    delta_o_h = s_v[:, None, None] * m_o

    delta_v = delta_v_h.transpose(0, 1).contiguous().view(d, d_v)
    delta_o = delta_o_h.reshape(d_v, d)
    return delta_v.mT.bfloat16(), delta_o.mT.bfloat16()


class CompositionalMuonOV:
    """Optimizer-like wrapper for attention V/O pairs only."""

    def __init__(
        self,
        pairs: list[tuple[nn.Parameter, nn.Parameter]],
        *,
        lr: float,
        weight_decay: float,
        mu: float,
        head_dim: int,
        cm_mp: float,
        damping: float,
    ):
        self.pairs = pairs
        self.head_dim = head_dim
        self.cm_mp = cm_mp
        self.damping = damping
        self.momentum = {p: torch.zeros_like(p) for pair in pairs for p in pair}
        self.param_groups = [dict(lr=lr, weight_decay=weight_decay, mu=mu)]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for v_weight, o_weight in self.pairs:
            for p in (v_weight, o_weight):
                p.grad = None if set_to_none else torch.zeros_like(p)

    @torch.no_grad()
    def step(self) -> None:
        group = self.param_groups[0]
        lr = group["lr"]
        wd = group["weight_decay"]
        mu = group["mu"]
        for v_weight, o_weight in self.pairs:
            if v_weight.grad is None or o_weight.grad is None:
                continue
            mv = self.momentum[v_weight]
            mo = self.momentum[o_weight]
            mv.lerp_(v_weight.grad, 1 - mu)
            mo.lerp_(o_weight.grad, 1 - mu)
            uv = v_weight.grad.lerp(mv, mu)
            uo = o_weight.grad.lerp(mo, mu)
            dv, do = _cm_ov_isotropic_delta(
                v_weight, o_weight, uv, uo, self.head_dim, self.damping
            )
            v_weight.mul_(1 - lr * wd)
            o_weight.mul_(1 - lr * wd)
            adjusted_lr = lr * 0.5 * self.cm_mp
            v_weight.add_(dv, alpha=-adjusted_lr)
            o_weight.add_(do, alpha=-adjusted_lr)


# ---------------------------------------------------------------------------
# Init scheme (mirrors modded-nanogpt simple: zero proj, normal embed, ...)
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
    # LRs from modded-nanogpt simple (FineWeb tuning); kept as-is for the port.
    def __init__(
        self,
        model_dim=384,
        num_layers=6,
        head_dim=64,
        max_len=1024,
        batch_size=32,
        n_steps=2150,
        cooldown_frac=0.7,
        embed_lr=0.3,
        head_lr=1.0 / 320,
        scalar_lr=0.01,
        muon_lr=0.035,
        muon_wd=0.025,
        cm_mp=1.0,
        cm_damping=1e-2,
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
        self.cm_mp = cm_mp
        self.cm_damping = cm_damping
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"H={self.model_dim//self.head_dim} bs={self.batch_size} "
                f"T={self.max_len} steps={self.n_steps} "
                f"cm_mp={self.cm_mp})")


def _train_modded(
    text: str,
    cfg: TrainConfig,
    device: torch.device,
) -> GPT:
    # Hold the full corpus on GPU as uint8; cast windows to long at sample time.
    # `list(text.encode())` would balloon to ~28GB of Python ints for the full
    # 530MB train split — torch.frombuffer keeps it tight.
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

    # AdamW gets embeddings, lm_head, and 1-D scalars (RMSNorm gains, biases).
    # Compositional Muon gets attention V/O pairs; ordinary Muon gets the
    # remaining 2-D block weights (Q/K and MLP matrices).
    ov_pairs = [(block.attn.v.weight, block.attn.proj.weight) for block in model.blocks]
    ov_ids = {id(p) for pair in ov_pairs for p in pair}
    block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2 and id(p) not in ov_ids]
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
    optimizer3 = CompositionalMuonOV(
        ov_pairs,
        lr=cfg.muon_lr,
        weight_decay=cfg.muon_wd,
        mu=0.95,
        head_dim=cfg.head_dim,
        cm_mp=cfg.cm_mp,
        damping=cfg.cm_damping,
    )
    optimizers = [optimizer1, optimizer2, optimizer3]
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[comp-muon] {n_params/1e6:.2f}M params  cfg={cfg}  "
          f"ov_pairs={len(ov_pairs)} muon_params={len(block_2d)}")

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
                f"[comp-muon] step {step:5d}/{cfg.n_steps}  "
                f"loss {loss.item():.4f}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    return model


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper (KV-cached, RoPE-offset-aware)
# ---------------------------------------------------------------------------

class ModdedNanoGPTCharModel(CharModel):
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
        # Seed with a single zero byte — a stream-start sentinel that
        # gives predict() a valid distribution before any real char is
        # observed.
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
    def predict_dist(self):
        """Side-channel: return the raw next-byte distribution (length 256).

        Used by ``wikitext.evaluate`` to compute CE in bits/char alongside
        argmax accuracy. Indexed by utf-8 byte id (0..255). Returned as a
        numpy array for backend-agnostic indexing.
        """
        import numpy as np  # local — torch is already loaded above
        if self._next_logits is None:
            raise RuntimeError("predict_dist() called before reset()")
        probs = F.softmax(self._next_logits.float(), dim=-1)
        return probs.cpu().numpy()

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
    # Optional reproducibility hook for floor-calibration sweeps. CUDA
    # kernels remain partly nondeterministic even when seeded — this
    # pins the sampling indices and init RNG, which is the dominant
    # run-to-run variance source for this submission.
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[comp-muon] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs: dict = {}
    if os.environ.get("SMOKE"):
        kwargs.update(model_dim=64, num_layers=2, max_len=64, batch_size=8,
                      n_steps=int(os.environ.get("N_STEPS", "4")), log_every=1)
    elif os.environ.get("N_STEPS"):
        kwargs["n_steps"] = int(os.environ["N_STEPS"])
    if os.environ.get("CM_MP") is not None:
        kwargs["cm_mp"] = float(os.environ["CM_MP"])
    if os.environ.get("CM_DAMPING") is not None:
        kwargs["cm_damping"] = float(os.environ["CM_DAMPING"])
    cfg = TrainConfig(**kwargs)
    model = _train_modded(train_text, cfg, device)
    return ModdedNanoGPTCharModel(model)
