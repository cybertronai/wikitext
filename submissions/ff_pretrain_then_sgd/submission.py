"""FF (Hinton goodness) pretrain + SGD cross-entropy fine-tune.

Two-stage training over the modded-nanogpt architecture (same Block,
init, optimizers, streaming wrapper as `submissions/modded_nanogpt/`).

Stage 1 (~80 s, ~300 steps cap):
  Per-block Hinton Forward-Forward goodness loss
    g_pos = mean(h_pos**2)      g_neg = mean(h_neg**2)
    loss  = softplus(-(g_pos - theta)) + softplus(g_neg - theta)
  Locality is enforced by detaching the block input on every step, so
  each block's local AdamW optimizer only sees gradients from its own
  goodness loss — no cross-block gradient flow.

  Negative samples: corrupt the last position of each context window to
  a uniformly random *different* byte (per the experiment design).

Stage-1 sanity gate (experiment_08 step 19 / Failure Modes):
  Compute the per-block mean activation std before FF pretrain
  (`init_std`) and after FF pretrain (`post_ff_std`). If any block's
  ratio post_ff_std / init_std > 5, re-normalize that block's residual
  output back to init scale by rescaling its mlp.proj and attn.proj
  output linears. If the rescaling itself fails (NaN, zero-div), fall
  back to the baseline (re-init the blocks from scratch). The gate
  guarantees Stage 2 starts at the same distribution scale as a fresh
  init, which is the documented FF failure mode (activations blow up
  to satisfy "make squared activations large").

Stage 2 (~220 s):
  Standard Muon + AdamW CE training over the full stack, identical to
  the modded-nanogpt baseline but with `n_steps` reduced to fit the
  remaining budget. Blocks start from the FF-pretrained init (or the
  fresh re-init if the gate triggered the fallback).
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
# Architecture (identical to submissions/modded_nanogpt/submission.py)
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
# Stage 1 — Forward-Forward (Hinton goodness) per-block local pretrain
# ---------------------------------------------------------------------------

def _measure_block_activation_stds(
    model: GPT,
    train_bytes: Tensor,
    batch_size: int,
    max_len: int,
) -> list[float]:
    """Return per-block activation std (population, over all elements) on a
    fresh held-out batch. Run under eval+autocast for parity with training.
    """
    model.eval()
    n = train_bytes.numel()
    with torch.no_grad():
        idx = torch.randint(0, n - max_len - 1, (batch_size,), device=train_bytes.device)
        offsets = idx[:, None] + torch.arange(max_len + 1, device=train_bytes.device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        use_amp = train_bytes.device.type == "cuda"
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _NullCtx()
        with ctx:
            h = model.norm1(model.embed(x))
            stds: list[float] = []
            for block in model.blocks:
                h, _ = block(h)
                stds.append(float(h.detach().float().std().item()))
    model.train()
    return stds


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def _sample_pos_neg(
    train_bytes: Tensor,
    batch_size: int,
    max_len: int,
) -> tuple[Tensor, Tensor]:
    """Positive: real (context, true-next-byte) window.
    Negative: same context with the LAST position replaced by a random
    other byte (uniform over the 255 alternatives).

    Returns two (B, T) long tensors.
    """
    device = train_bytes.device
    n = train_bytes.numel()
    idx = torch.randint(0, n - max_len - 1, (batch_size,), device=device)
    offsets = idx[:, None] + torch.arange(max_len, device=device)[None, :]
    pos = train_bytes[offsets].long()
    neg = pos.clone()
    # Random replacement byte at the last position, guaranteed different.
    last = neg[:, -1]
    rand_byte = torch.randint(0, 255, (batch_size,), device=device, dtype=last.dtype)
    # Shift to skip the original value: 0..254 -> 0..254 (skipping `last`)
    rand_byte = torch.where(rand_byte >= last, rand_byte + 1, rand_byte)
    neg[:, -1] = rand_byte
    return pos, neg


def _ff_pretrain(
    model: GPT,
    train_bytes: Tensor,
    *,
    batch_size: int,
    max_len: int,
    ff_lr: float,
    max_steps: int,
    max_seconds: float,
) -> dict:
    """Run Stage-1 FF pretrain in-place on `model.blocks`. Returns a dict
    of telemetry (per-block goodness/activation-std trajectory + final
    step count + wall-clock).
    """
    device = train_bytes.device
    model_dim = model.embed.embedding_dim
    theta = float(model_dim)

    # Per-block AdamW. Note: norm1/norm2 RMSNorm gains in each block are
    # included so the block has a chance to learn its own input scale.
    per_block_opts = [
        AdamW(b.parameters(), lr=ff_lr, betas=(0.9, 0.95), weight_decay=0.0)
        for b in model.blocks
    ]

    telemetry = {
        "steps": 0,
        "duration_s": 0.0,
        "g_pos_last": [None] * len(model.blocks),
        "g_neg_last": [None] * len(model.blocks),
        "h_std_last": [None] * len(model.blocks),
        "loss_last": [None] * len(model.blocks),
    }

    use_amp = device.type == "cuda"
    model.train()
    t_start = time.monotonic()
    print(f"[ff] Stage-1 begin  theta={theta:.1f}  "
          f"max_steps={max_steps}  budget={max_seconds:.0f}s  "
          f"batch={batch_size}  T={max_len}", flush=True)
    step = 0
    while step < max_steps:
        if time.monotonic() - t_start > max_seconds:
            break
        x_pos, x_neg = _sample_pos_neg(train_bytes, batch_size, max_len)

        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _NullCtx()
        with ctx:
            h_pos = model.norm1(model.embed(x_pos))
            h_neg = model.norm1(model.embed(x_neg))

        # Sequentially traverse blocks. For each block:
        #   * detach the inbound activation (no cross-block grad)
        #   * compute goodness on positive + negative
        #   * step that block's local optimizer
        #   * forward the (detached) clean output to feed the next block
        for i, block in enumerate(model.blocks):
            h_pos_in = h_pos.detach().requires_grad_(False)
            h_neg_in = h_neg.detach().requires_grad_(False)

            ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _NullCtx()
            with ctx:
                h_pos_out, _ = block(h_pos_in)
                h_neg_out, _ = block(h_neg_in)
                # Sum-of-squared-activations per token; mean across batch&time.
                # Operate in fp32 for numeric stability of softplus.
                g_pos = (h_pos_out.float() ** 2).sum(-1).mean()
                g_neg = (h_neg_out.float() ** 2).sum(-1).mean()
                loss = F.softplus(-(g_pos - theta)) + F.softplus(g_neg - theta)

            per_block_opts[i].zero_grad(set_to_none=True)
            loss.backward()
            per_block_opts[i].step()

            # Hand off to the next block — detach so the loop stays local.
            h_pos = h_pos_out.detach()
            h_neg = h_neg_out.detach()

            telemetry["g_pos_last"][i] = float(g_pos.detach().item())
            telemetry["g_neg_last"][i] = float(g_neg.detach().item())
            telemetry["loss_last"][i] = float(loss.detach().item())
            telemetry["h_std_last"][i] = float(
                h_pos_out.detach().float().std().item()
            )

        step += 1
        if step % 25 == 0 or step == 1:
            elapsed = time.monotonic() - t_start
            g_pos_s = "  ".join(f"{v:7.1f}" for v in telemetry["g_pos_last"])
            g_neg_s = "  ".join(f"{v:7.1f}" for v in telemetry["g_neg_last"])
            std_s = "  ".join(f"{v:6.2f}" for v in telemetry["h_std_last"])
            print(
                f"[ff] step {step:4d}  t={elapsed:5.1f}s  "
                f"g_pos=[{g_pos_s}]  g_neg=[{g_neg_s}]  "
                f"h_std=[{std_s}]",
                flush=True,
            )

    telemetry["steps"] = step
    telemetry["duration_s"] = time.monotonic() - t_start
    print(
        f"[ff] Stage-1 end    steps={step}  "
        f"duration={telemetry['duration_s']:.1f}s",
        flush=True,
    )
    return telemetry


def _rescale_block_outputs(model: GPT, scale_factors: list[float]) -> None:
    """Per-block output rescale: divide both proj output linears (attn.proj,
    mlp.proj) by `scale_factors[i]`. Since these projections are the only
    paths through which a block contributes to its residual stream output,
    rescaling them rescales the *residual contribution* of the block by
    the same factor. This brings post-FF activation std back toward init
    scale without re-initializing the trained block parameters.

    Note: the residual identity path (x + h) is NOT rescaled, only the
    block's *contribution* h. This is an approximation — the new activation
    std becomes a mix of the identity input std and the rescaled contribution
    — but it's the simplest in-place gate fix that preserves the FF init.
    """
    with torch.no_grad():
        for i, (block, s) in enumerate(zip(model.blocks, scale_factors)):
            if s <= 0 or not math.isfinite(s):
                continue
            block.attn.proj.weight.data.div_(s)
            block.attn.proj.bias.data.div_(s)
            block.mlp.proj.weight.data.div_(s)
            block.mlp.proj.bias.data.div_(s)


# ---------------------------------------------------------------------------
# Stage 2 — standard SGD CE (modded-nanogpt loop)
# ---------------------------------------------------------------------------

def _stage2_train(
    model: GPT,
    train_bytes: Tensor,
    *,
    batch_size: int,
    max_len: int,
    n_steps: int,
    cooldown_frac: float,
    embed_lr: float,
    head_lr: float,
    scalar_lr: float,
    muon_lr: float,
    muon_wd: float,
    log_every: int,
    max_seconds: float,
) -> None:
    device = train_bytes.device
    n = train_bytes.numel()
    block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2]
    scalars = [p for p in model.parameters() if p.ndim < 2]
    optimizer1 = AdamW(
        [
            dict(params=[model.embed.weight], lr=embed_lr),
            dict(params=[model.proj.weight], lr=head_lr),
            dict(params=scalars, lr=scalar_lr),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    optimizer2 = Muon(block_2d, lr=muon_lr, weight_decay=muon_wd)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    def set_lr(step: int) -> None:
        progress = step / n_steps
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = max(0.0, (1 - progress) / cooldown_frac)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * eta

    model.train()
    use_amp = device.type == "cuda"
    t0 = time.monotonic()
    print(f"[stage2] begin  n_steps={n_steps}  budget={max_seconds:.0f}s",
          flush=True)
    for step in range(n_steps):
        if time.monotonic() - t0 > max_seconds:
            print(f"[stage2] budget exhausted at step {step}", flush=True)
            break
        set_lr(step)
        idx = torch.randint(0, n - max_len - 1, (batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(max_len + 1, device=device)[None, :]
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

        if log_every and (step % log_every == 0 or step == n_steps - 1):
            elapsed = time.monotonic() - t0
            print(
                f"[stage2] step {step:5d}/{n_steps}  "
                f"loss {loss.item():.4f}  elapsed {elapsed:.0f}s",
                flush=True,
            )
    print(f"[stage2] end    duration={time.monotonic()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper
# ---------------------------------------------------------------------------

class FFPretrainCharModel(CharModel):
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
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[ff_pretrain] SEED={seed}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Architecture / training hyperparameters --------------------------
    model_dim = 384
    num_layers = 6
    head_dim = 64
    max_len = 1024
    batch_size = 32

    # Stage 1 (FF)
    ff_max_seconds = 80.0
    ff_max_steps = 300
    ff_lr = 1e-3
    ff_gate_ratio = 5.0
    # Use a smaller window for FF to fit more steps in 80 s. Hinton's
    # goodness loss does not depend on T being maxed out (signal is per-
    # token), and shorter T means more steps per second.
    ff_max_len = 256
    ff_batch_size = 32

    # Stage 2 (CE)
    stage2_budget_seconds = 215.0  # 300 - 80 budget - 5 s headroom
    # Reuse modded-nanogpt's LRs; n_steps shortened so cooldown schedule
    # still ends on time. Baseline took ~246 s for 2150 steps -> ~0.115 s/step.
    # With ~215 s budget we expect ~1850 steps to fit; pad target slightly.
    stage2_n_steps = 1900
    cooldown_frac = 0.7
    embed_lr = 0.3
    head_lr = 1.0 / 320
    scalar_lr = 0.01
    muon_lr = 0.035
    muon_wd = 0.025
    log_every = 100

    # ---- Data ---- --------------------------------------------------------
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < max_len + 1:
        raise ValueError(f"need at least {max_len+1} bytes; got {n}")

    # ---- Model + init -----------------------------------------------------
    model = GPT(
        vocab_size=256,
        num_layers=num_layers,
        model_dim=model_dim,
        head_dim=head_dim,
        max_len=max_len,
    ).to(device)
    _init_modded(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ff_pretrain] {n_params/1e6:.2f}M params  dim={model_dim} "
          f"L={num_layers} bs={batch_size} T={max_len}", flush=True)

    # ---- Init-time activation-std baseline (for sanity gate) -------------
    init_stds = _measure_block_activation_stds(
        model, train_bytes, ff_batch_size, ff_max_len,
    )
    print(
        f"[ff_pretrain] init-time per-block h_std: "
        f"[{'  '.join(f'{s:6.3f}' for s in init_stds)}]",
        flush=True,
    )

    # ---- Stage 1: FF pretrain --------------------------------------------
    ff_telemetry = _ff_pretrain(
        model,
        train_bytes,
        batch_size=ff_batch_size,
        max_len=ff_max_len,
        ff_lr=ff_lr,
        max_steps=ff_max_steps,
        max_seconds=ff_max_seconds,
    )

    # ---- Stage-1 sanity gate ---------------------------------------------
    post_stds = _measure_block_activation_stds(
        model, train_bytes, ff_batch_size, ff_max_len,
    )
    ratios = [
        (post / init) if (init > 0 and math.isfinite(post)) else float("inf")
        for post, init in zip(post_stds, init_stds)
    ]
    print(
        f"[ff_pretrain] post-FF  per-block h_std: "
        f"[{'  '.join(f'{s:6.3f}' for s in post_stds)}]",
        flush=True,
    )
    print(
        f"[ff_pretrain] gate ratio (post/init) per block: "
        f"[{'  '.join(f'{r:6.2f}' for r in ratios)}]  threshold={ff_gate_ratio:.1f}",
        flush=True,
    )

    any_bad = any(
        (not math.isfinite(r)) or r > ff_gate_ratio for r in ratios
    )
    if any_bad:
        # Try rescaling first: divide each block's *contribution* (its
        # attn.proj / mlp.proj output linears) by the ratio. Then re-
        # measure; if still bad, fall back to baseline init.
        print(
            f"[ff_pretrain] sanity gate TRIGGERED — rescaling blocks "
            f"by their ratios", flush=True,
        )
        rescale_factors = [
            max(r, 1.0) if math.isfinite(r) else float("inf")
            for r in ratios
        ]
        try:
            _rescale_block_outputs(model, rescale_factors)
            recheck_stds = _measure_block_activation_stds(
                model, train_bytes, ff_batch_size, ff_max_len,
            )
            recheck_ratios = [
                (post / init) if (init > 0 and math.isfinite(post)) else float("inf")
                for post, init in zip(recheck_stds, init_stds)
            ]
            print(
                f"[ff_pretrain] after rescale h_std: "
                f"[{'  '.join(f'{s:6.3f}' for s in recheck_stds)}]  "
                f"ratios: [{'  '.join(f'{r:6.2f}' for r in recheck_ratios)}]",
                flush=True,
            )
            still_bad = any(
                (not math.isfinite(r)) or r > ff_gate_ratio for r in recheck_ratios
            )
        except Exception as e:
            print(f"[ff_pretrain] rescale FAILED ({e!r}) — falling back to baseline init",
                  flush=True)
            still_bad = True

        if still_bad:
            print(
                "[ff_pretrain] sanity gate FALLBACK: re-init blocks from "
                "baseline init (discards FF pretrain)",
                flush=True,
            )
            # Re-init only the block parameters; keep embed/head/norms.
            for name, p in model.named_parameters():
                if "blocks." not in name:
                    continue
                w = p.data
                if name.endswith("weight"):
                    if "proj" in name:
                        w.zero_()
                    else:
                        w.normal_(std=0.33**0.5 / w.size(-1) ** 0.5)
                elif name.endswith("bias"):
                    w.zero_()
                elif name.endswith("gains"):
                    w.fill_(1.0)
    else:
        print(f"[ff_pretrain] sanity gate PASSED — proceeding to Stage 2",
              flush=True)

    # ---- Stage 2: CE fine-tune the full stack ----------------------------
    _stage2_train(
        model,
        train_bytes,
        batch_size=batch_size,
        max_len=max_len,
        n_steps=stage2_n_steps,
        cooldown_frac=cooldown_frac,
        embed_lr=embed_lr,
        head_lr=head_lr,
        scalar_lr=scalar_lr,
        muon_lr=muon_lr,
        muon_wd=muon_wd,
        log_every=log_every,
        max_seconds=stage2_budget_seconds,
    )

    return FFPretrainCharModel(model)
