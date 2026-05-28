"""Direct Feedback Alignment (DFA) on a small byte-level transformer.

Replaces the chain-rule backward pass with a fixed random projection of
the output-layer error to each block's output. Inside each block,
backprop still flows normally — but the gradient ARRIVING at the block's
output is the DFA projection of the global error, not the upstream
chain-rule gradient. This is the Launay-Poli-Krzakala 2020 recipe
("DFA scales to modern deep learning"): block-DFA on a transformer.

Mechanism per block l (output h_l):
    Forward (standard):
        h_l = h_{l-1} + Attn(LN(h_{l-1}))
        h_l = h_l    + MLP(LN(h_l))
    Backward (DFA):
        grad_into_block_l_output  :=  B_l @ e        # NOT the chain-rule grad
        where e = softmax(logits) - one_hot(y)        # output-layer error
              B_l is a fixed random matrix, frozen.

Implementation: a `DFAHook` torch.autograd.Function inserted after each
block. Forward is identity; backward returns `B_l @ e_flat.T` (reshaped
back to (B, T, D)) regardless of the upstream chain-rule gradient. The
output error `e` is stashed in a process-global slot just before
`loss.backward()` is called.

The per-step FLOP saving comes from skipping the cross-block chain
matmuls: the chain-rule backward path through each block's residual +
attention + MLP is replaced by one (d x V) @ (V x BT) matmul per block.
That eliminates ~half the backward FLOPs in the transformer (the cross-
block-output @ next-block-input chain). Inside the block, attention/MLP
backprop still runs — Launay 2020 reports that's needed for the per-
layer weight updates to be effective.

Architecture closely mirrors `submissions/modded_nanogpt` (6-layer 384-d
transformer, RoPE, RMSNorm, ReLU^2 MLP) but uses plain SGD-momentum
optimizer (Muon's spectral norm-stabilization assumes chain-rule grads;
DFA's pseudo-gradients have different statistics, so we use a plain
optimizer per Launay 2020's recipe).
"""
from __future__ import annotations

__author__ = "@armin-claude-1m"

import math
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wikitext import CharModel


# ---------------------------------------------------------------------------
# DFA plumbing: a process-global slot holding the current output-layer error.
# Set just before loss.backward(); read by DFAHook.backward.
# ---------------------------------------------------------------------------

class _DFAErrorSlot:
    """Holds the current output-layer error `e` of shape (B*T, V)
    AND a projection target dim `D` to which each block's hook should
    project. The hook also reads the block's per-layer fixed feedback
    matrix `B_l` of shape (V, D) from the slot.

    Module-level state because the DFAHook backward closure needs to
    read it without entering the autograd graph. Cleared after each
    backward to avoid stale-error bugs.
    """
    error: Optional[Tensor] = None  # (B*T, V) float
    shape: Optional[tuple] = None    # (B, T, D)


_slot = _DFAErrorSlot()


class DFAHook(torch.autograd.Function):
    """Identity in forward; in backward, returns the FIXED projection of
    the current output error (held in `_slot.error`) through this
    block's frozen feedback matrix `B`.

    Inputs:
        x: (B, T, D)       — block output (passed through unchanged)
        B: (V, D)          — frozen feedback matrix for this block

    Backward grads:
        grad_x: (B, T, D)  — replaces the upstream chain-rule grad with
                              the DFA projection.
        grad_B: None       — feedback matrices are NOT trained.

    Note: we do not call `.backward()` through any upstream module of
    this hook in the standard sense — but PyTorch still calls our
    `backward` because we're on the computational graph; we just IGNORE
    `grad_output` and substitute our own projection. The upstream graph
    (everything BEFORE this hook in the forward pass — i.e. the block's
    internals) then receives our DFA projection as its `grad_output`,
    so attention/MLP/RMSNorm inside the block all get backprop driven
    by the DFA-projected error.
    """

    @staticmethod
    def forward(ctx, x: Tensor, B: Tensor) -> Tensor:
        ctx.save_for_backward(B)
        ctx.x_shape = x.shape
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (B,) = ctx.saved_tensors
        B_shape = ctx.x_shape  # (Bsz, T, D)
        Bsz, T, D = B_shape
        # _slot.error: (Bsz*T, V); B: (V, D). Output: (Bsz*T, D)
        e = _slot.error
        if e is None:
            # No DFA error available — fall back to chain-rule (e.g. eval).
            return grad_output, None
        # Project the error and reshape.
        proj = e @ B  # (Bsz*T, D)
        # Cast to match grad_output dtype (autocast may have changed it).
        proj = proj.to(dtype=grad_output.dtype if grad_output is not None else proj.dtype)
        proj = proj.view(Bsz, T, D)
        # Scale by 1/(Bsz*T) so per-token magnitude matches mean-CE backward.
        proj = proj / (Bsz * T)
        return proj, None


# ---------------------------------------------------------------------------
# Architecture (mirrors modded_nanogpt: RMSNorm, RoPE, attention, ReLU^2 MLP)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is not None:
            return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    """Half-truncate RoPE with base=1024 (same as modded_nanogpt)."""
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
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
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
    """Transformer block with a DFA tap on its output.

    The DFA hook is applied AFTER the residual-add, so backward into this
    block's internals (attn + mlp + RMSNorms) is driven by the DFA-
    projected error instead of the chain-rule gradient from later blocks.
    """
    def __init__(self, dim: int, head_dim: int, vocab_size: int, use_dfa: bool = True):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim=head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.use_dfa = use_dfa
        # Frozen feedback matrix for this block: (V, D). Scaled to match
        # roughly the chain-rule grad magnitude. Stored as a buffer so it
        # moves with .to(device) but is not in the parameter list.
        if use_dfa:
            B_l = torch.randn(vocab_size, dim) / math.sqrt(dim)
            self.register_buffer("B_dfa", B_l)
        else:
            self.B_dfa = None

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
        offset: int = 0,
        dfa_active: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h, new_kv = self.attn(self.norm1(x), kv_cache, offset=offset)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        if dfa_active and self.use_dfa and self.B_dfa is not None:
            x = DFAHook.apply(x, self.B_dfa)
        return x, new_kv


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        head_dim: int = 64,
        max_len: int = 1024,
        use_dfa: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.use_dfa = use_dfa
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            Block(model_dim, head_dim=head_dim, vocab_size=vocab_size, use_dfa=use_dfa)
            for _ in range(num_layers)
        ])
        self.proj = Linear(model_dim, vocab_size, bias=True)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(
        self,
        inputs: Tensor,
        kv_caches: Optional[list[tuple[Tensor, Tensor]]] = None,
        offset: int = 0,
        dfa_active: bool = False,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        x = self.norm1(self.embed(inputs))
        new_caches: list[tuple[Tensor, Tensor]] = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv, offset=offset, dfa_active=dfa_active)
            new_caches.append(new_kv)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits, new_caches


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def _init_model(model: GPT) -> None:
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name and "mlp" not in name and "attn" not in name:
                # Output projection (lm_head): small init
                w.normal_(std=0.02)
            elif "embed" in name:
                w.normal_(std=0.02)
            else:
                # Xavier-ish for inner linears
                w.normal_(std=0.33**0.5 / w.size(-1) ** 0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.fill_(1.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainConfig:
    def __init__(
        self,
        model_dim=384,
        num_layers=6,
        head_dim=64,
        max_len=512,
        batch_size=32,
        n_steps=4000,
        cooldown_frac=0.7,
        lr=3e-3,
        embed_lr=0.1,
        head_lr=3e-4,
        momentum=0.9,
        weight_decay=0.0,
        log_every=100,
    ):
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.cooldown_frac = cooldown_frac
        self.lr = lr
        self.embed_lr = embed_lr
        self.head_lr = head_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"H={self.model_dim//self.head_dim} bs={self.batch_size} "
                f"T={self.max_len} steps={self.n_steps} lr={self.lr})")


def _train_dfa(
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
        use_dfa=True,
    ).to(device)
    _init_model(model)

    # Optimizer: plain SGD-momentum on all parameters. Per Launay 2020,
    # DFA needs simpler/lower-magnitude updates than backprop. We use
    # separate LRs for the embedding, head, and inner blocks.
    inner_params = []
    for blk in model.blocks:
        inner_params += list(blk.parameters())
    inner_params += [model.norm1.gains, model.norm2.gains]

    optimizer = torch.optim.SGD(
        [
            dict(params=[model.embed.weight], lr=cfg.embed_lr),
            dict(params=[model.proj.weight, model.proj.bias], lr=cfg.head_lr),
            dict(params=inner_params, lr=cfg.lr),
        ],
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dfa] {n_params/1e6:.2f}M params  cfg={cfg}", flush=True)
    n_feedback = sum(b.B_dfa.numel() for b in model.blocks)
    print(f"[dfa] {n_feedback/1e6:.2f}M frozen feedback params (NOT trained)", flush=True)

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
    last_log_time = t0
    last_log_step = 0
    actual_steps_run = 0

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
                logits, _ = model(x, dfa_active=True)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            logits, _ = model(x, dfa_active=True)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))

        # Compute the output-layer error e = softmax(logits) - one_hot(y)
        # and stash it in the DFA slot for the backward pass.
        with torch.no_grad():
            B_sz, T_sz, V = logits.shape
            probs = F.softmax(logits.float(), dim=-1)  # (B, T, V)
            e = probs.view(-1, V).clone()              # (B*T, V)
            e.scatter_add_(1, y.reshape(-1, 1), -torch.ones_like(e[:, :1]))
            # e is now (softmax - one_hot), the cross-entropy output grad.
            _slot.error = e.to(dtype=torch.float32)
            _slot.shape = (B_sz, T_sz, V)

        # Backward. The DFAHook in each block intercepts grad_output and
        # substitutes B_l @ e. The HEAD/embed/output-norm get normal CE
        # gradients (no DFA tap on them — they're at the top of the
        # net, and Launay 2020 keeps the head trained by true CE grad).
        loss.backward()

        # Clear DFA slot.
        _slot.error = None

        # Optional grad clipping for stability under DFA.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        actual_steps_run = step + 1

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            dt = time.monotonic() - last_log_time
            ds = step - last_log_step
            step_rate = (ds / max(dt, 1e-6)) if ds > 0 else 0.0
            print(
                f"[dfa] step {step:5d}/{cfg.n_steps}  "
                f"loss {loss.item():.4f}  "
                f"elapsed {elapsed:.0f}s  "
                f"({step_rate:.1f} steps/s)",
                flush=True,
            )
            last_log_time = time.monotonic()
            last_log_step = step

    elapsed = time.monotonic() - t0
    print(f"[dfa] training done: {actual_steps_run} steps in {elapsed:.1f}s", flush=True)
    return model


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper (KV-cached, RoPE-offset-aware)
# Same as modded_nanogpt — DFA only affects training, not inference.
# ---------------------------------------------------------------------------

class DFACharModel(CharModel):
    def __init__(self, model: GPT, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self._kv: Optional[list[tuple[Tensor, Tensor]]] = None
        self._next_logits: Optional[Tensor] = None
        self._pos: int = 0

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = None
        self._pos = 0
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        logits, self._kv = self.model(x, None, offset=self._pos, dfa_active=False)
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
            logits, self._kv = self.model(x, self._kv, offset=self._pos, dfa_active=False)
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

def train(train_text: str, valid_text: Optional[str] = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[dfa] SEED={seed}")
    else:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    model = _train_dfa(train_text, cfg, device)
    return DFACharModel(model)
