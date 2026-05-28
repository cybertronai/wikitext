"""Hebbian Fast-Weight Block v2 — Schlag sum-norm + WY parallel scan + T=512.

Experiment 07 v2. Changes vs v1:
  1. **Schlag sum-normalization** on ELU+1 feature map (Option B1 of the spec).
     v1 used L2-normalize on raw projections with no denominator, which is
     non-canonical. v2 implements the Schlag-Schmidhuber 2021 (arXiv 2102.11174
     Eq. 29) rule exactly:
         φ(x) = ELU(x) + 1
         φ_norm = φ / (φ.sum(-1) + ε)
  2. **WY-Householder parallel scan** as the chunkwise scan inside the FW block,
     following Yang 2024 (arXiv 2406.06484) Algorithm 2. The chunk-internal
     recurrence becomes one unit-lower triangular solve + a handful of bmm.
     (v1 already had a WY-style scan; v2 keeps the same backbone math; the
     headline change is the feature map + normalization.)
  3. **T = 512** (half v1's 1024). Halves per-step scan cost while keeping the
     per-byte training signal density identical (random window over a 540 MB
     stream).

Architecture: 4-layer modded-nanogpt body + 1 HebbianFastWeightBlock at the
last position (block index 4, i.e. 5 blocks total). Same as v1.

Gradient contract: W is a non-differentiable hidden state. Writes run under
torch.no_grad. Reads o_t = W q_t propagate gradient through q_t and `proj`
only — k, v, β projections train via dynamics only.

Based on:
  submissions/hebbian_fw_block/submission.py  (v1)
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
# Standard transformer building blocks (verbatim from modded_nanogpt)
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


# ---------------------------------------------------------------------------
# Hebbian Fast-Weight Block — Schlag sum-norm + WY parallel scan
# ---------------------------------------------------------------------------

def _phi_sumnorm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Schlag sum-normalization on ELU+1 feature map (Option B1).

    Implements:
        φ(x) = ELU(x) + 1      # strictly positive (Katharopoulos 2020)
        φ_norm = φ / (Σ φ)     # row-stochastic (Schlag 2021 Eq. 29)
    """
    phi = F.elu(x) + 1.0
    return phi / (phi.sum(dim=-1, keepdim=True) + eps)


class HebbianFastWeightBlock(nn.Module):
    """Single transformer-style block whose attention is replaced by a
    Schlag-Schmidhuber 2021 delta-rule fast-weight scan, parallelized with
    the WY-Householder form (Yang 2024 Algorithm 2).

    Parameters (trainable, SGD): q, k, v, beta, proj, norm1, norm2, mlp.
    Hidden state (NOT trainable, no backprop): W: (B, d, d).
    """
    def __init__(self, dim: int, chunk_size: int = 64, decay: float = 1.0):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.decay = decay
        self.norm1 = RMSNorm(dim)
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.beta = Linear(dim, 1)
        self.proj = Linear(dim, dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim)

    def _scan_chunk(
        self,
        q_chunk: Tensor,        # (B, Tc, d)  -- fp32, sum-normalized φ
        k_chunk: Tensor,        # (B, Tc, d)  -- fp32, sum-normalized φ
        v_chunk: Tensor,        # (B, Tc, d)  -- fp32
        beta_chunk: Tensor,     # (B, Tc)     -- fp32
        W: Tensor,              # (B, d, d)   -- fp32, no_grad detached
    ) -> tuple[Tensor, Tensor]:
        """Chunkwise parallel form of the delta-rule scan (Yang 2024 Alg. 2).

        For chunk inputs (Q, K, V) of shape (B, C, d), β of shape (B, C), and
        initial state W (B, d, d), the chunk-internal scan reduces to:

            U = (I + β · strict_lower(K K^T))^{-1} β (V - K W^T)
            O = Q W^T + strict_lower(Q K^T) U
            W_new = W + U^T K

        Mathematically equivalent to the sequential delta-rule loop for
        decay = 1.0. T inner sequential iterations collapse to T/C outer
        iterations, each issuing O(1) GPU kernels.

        Solves are forced to fp32 per Yang 2024 §4.2 (bf16 triangular solve
        is unstable on A100/H100 for moderate C).
        """
        assert self.decay == 1.0, (
            "WY parallel scan currently supports decay=1.0 only"
        )
        B, C, d = q_chunk.shape
        device = q_chunk.device
        dtype = q_chunk.dtype  # fp32

        # Writes run gradient-free (W is a non-differentiable hidden state).
        with torch.no_grad():
            k_det = k_chunk.detach()
            v_det = v_chunk.detach()
            beta_det = beta_chunk.detach()
            W_det = W.detach()

            # V_tilde[t] = V[t] - K[t] W^T — residual against chunk-start W.
            V_tilde = v_det - torch.bmm(k_det, W_det.transpose(-1, -2))

            # A[t,s] = β_t (K[t] · K[s]) for s < t. solve_triangular with
            # unitriangular=True ignores the diagonal; we add I for clarity.
            KKt = torch.bmm(k_det, k_det.transpose(-1, -2))
            A = torch.tril(beta_det.unsqueeze(-1) * KKt, diagonal=-1)
            A = A + torch.eye(C, device=device, dtype=dtype)

            RHS = beta_det.unsqueeze(-1) * V_tilde
            U = torch.linalg.solve_triangular(
                A, RHS, upper=False, unitriangular=True,
            )

            # W_new = W + Σ_t u_t k_t^T = W + U^T K
            W_new = W_det + torch.bmm(U.transpose(-1, -2), k_det)

        # Reads: gradient flows only through q_chunk. K, W, U stay detached.
        k_for_out = k_chunk.detach()
        QKt = torch.bmm(q_chunk, k_for_out.transpose(-1, -2))
        A_qk = torch.tril(QKt, diagonal=-1)
        outs = torch.bmm(q_chunk, W.detach().transpose(-1, -2)) + torch.bmm(A_qk, U)

        return outs, W_new

    def _scan(
        self,
        q: Tensor,     # (B, T, d)  fp32, sum-normalized φ
        k: Tensor,     # (B, T, d)  fp32, sum-normalized φ
        v: Tensor,     # (B, T, d)  fp32
        beta: Tensor,  # (B, T)     fp32
        W0: Tensor,    # (B, d, d)  fp32
    ) -> tuple[Tensor, Tensor]:
        B, T, d = q.shape
        outs: list[Tensor] = []
        W = W0
        cs = self.chunk_size
        for s in range(0, T, cs):
            e = min(T, s + cs)
            o_chunk, W = self._scan_chunk(
                q[:, s:e], k[:, s:e], v[:, s:e], beta[:, s:e], W,
            )
            outs.append(o_chunk)
        out = torch.cat(outs, dim=1)
        return out, W

    def forward(
        self,
        x: Tensor,
        fw_state: Tensor | None = None,
        offset: int = 0,  # signature compat; unused
    ) -> tuple[Tensor, Tensor]:
        """Apply the Hebbian fast-weight scan + MLP, residual style.

        Args:
            x: (B, T, d) hidden states.
            fw_state: optional running W (B, d, d). None -> zeros.

        Returns:
            (x_out, W_final).
        """
        B, T, d = x.shape

        # Sub-layer 1: fast-weight scan, pre-norm, residual.
        h = self.norm1(x)

        # Schlag sum-norm on ELU+1 feature map (Option B1, spec axis B).
        # All scan math in fp32 (bf16 outer-product write would lose precision
        # over T writes; bf16 triangular solve is unstable per Yang 2024 §4.2).
        q = _phi_sumnorm(self.q(h).float())
        k = _phi_sumnorm(self.k(h).float())
        v = self.v(h).float()
        beta = torch.sigmoid(self.beta(h).float()).squeeze(-1)  # (B, T)

        if fw_state is None:
            W = torch.zeros(B, d, d, device=x.device, dtype=torch.float32)
        else:
            W = fw_state.float().detach()
        out_fp32, W_new = self._scan(q, k, v, beta, W)
        out = out_fp32.type_as(x)
        x = x + self.proj(out)

        # Sub-layer 2: MLP + pre-norm + residual.
        x = x + self.mlp(self.norm2(x))
        return x, W_new


# ---------------------------------------------------------------------------
# GPT body with the Hebbian block at the last position
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """4 standard blocks + 1 HebbianFastWeightBlock at index 4 (last).

    Total = 5 blocks.
    """
    def __init__(
        self,
        vocab_size: int,
        num_std_blocks: int = 4,
        model_dim: int = 384,
        head_dim: int = 64,
        max_len: int = 512,
        fw_chunk: int = 64,
        fw_decay: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_std_blocks = num_std_blocks
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(
            [Block(model_dim, head_dim=head_dim) for _ in range(num_std_blocks)]
        )
        self.fw_block = HebbianFastWeightBlock(
            model_dim, chunk_size=fw_chunk, decay=fw_decay,
        )
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(
        self,
        inputs: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        fw_state: Tensor | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]], Tensor]:
        x = self.norm1(self.embed(inputs))
        new_caches: list[tuple[Tensor, Tensor]] = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv, offset=offset)
            new_caches.append(new_kv)
        x, W_new = self.fw_block(x, fw_state=fw_state, offset=offset)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits, new_caches, W_new


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


# ---------------------------------------------------------------------------
# Init scheme
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
    def __init__(
        self,
        model_dim=384,
        num_std_blocks=4,       # 4 std + 1 hebbian = 5 total
        head_dim=64,
        max_len=512,            # v2: halved from v1's 1024 (spec axis C)
        batch_size=32,
        n_steps=2150,
        cooldown_frac=0.7,
        embed_lr=0.3,
        head_lr=1.0 / 320,
        scalar_lr=0.01,
        muon_lr=0.035,
        muon_wd=0.025,
        fw_chunk=64,
        fw_decay=1.0,
        log_every=100,
    ):
        self.model_dim = model_dim
        self.num_std_blocks = num_std_blocks
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
        self.fw_chunk = fw_chunk
        self.fw_decay = fw_decay
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L_std={self.num_std_blocks}+1fw "
                f"H={self.model_dim//self.head_dim} bs={self.batch_size} "
                f"T={self.max_len} steps={self.n_steps} "
                f"fw_chunk={self.fw_chunk})")


def _train_hebbian(
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
        num_std_blocks=cfg.num_std_blocks,
        model_dim=cfg.model_dim,
        head_dim=cfg.head_dim,
        max_len=cfg.max_len,
        fw_chunk=cfg.fw_chunk,
        fw_decay=cfg.fw_decay,
    ).to(device)
    _init_modded(model)

    body_2d = [p for n_, p in model.named_parameters()
               if (n_.startswith("blocks.") or n_.startswith("fw_block."))
               and p.ndim >= 2]
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
    optimizer2 = Muon(body_2d, lr=cfg.muon_lr, weight_decay=cfg.muon_wd)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hebbian_fw_v2] {n_params/1e6:.2f}M params  cfg={cfg}")

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
                logits, _, _ = model(x, fw_state=None)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            logits, _, _ = model(x, fw_state=None)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        for opt in optimizers:
            opt.step()

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            print(
                f"[hebbian_fw_v2] step {step:5d}/{cfg.n_steps}  "
                f"loss {loss.item():.4f}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    return model


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper
# ---------------------------------------------------------------------------

class HebbianFWCharModel(CharModel):
    """KV-cached + fast-weight-state streaming inference.

    The KV cache holds the 4 standard self-attention layers as before.
    The Hebbian block's fast-weight matrix W (B=1, d, d) lives in
    self._W and is updated by exactly one delta step per observed byte
    (chunk-size-1 fold of the same scan).
    """
    def __init__(self, model: GPT, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self._kv: list[tuple[Tensor, Tensor]] | None = None
        self._W: Tensor | None = None
        self._next_logits: Tensor | None = None
        self._pos: int = 0

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = None
        self._W = torch.zeros(
            1, self.model.fw_block.dim, self.model.fw_block.dim,
            device=self.device, dtype=torch.float32,
        )
        self._pos = 0
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        logits, self._kv, self._W = self.model(
            x, kv_caches=None, fw_state=self._W, offset=self._pos,
        )
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
        if self._kv is None or self._W is None:
            raise RuntimeError("observe() called before reset()")
        for byte in char.encode("utf-8"):
            self._maybe_trim_cache()
            x = torch.tensor([[byte]], dtype=torch.long, device=self.device)
            logits, self._kv, self._W = self.model(
                x, kv_caches=self._kv, fw_state=self._W, offset=self._pos,
            )
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
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[hebbian_fw_v2] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    model = _train_hebbian(train_text, cfg, device)
    return HebbianFWCharModel(model)
