"""Mono-Forward block-local pretrain + SGD cross-entropy fine-tune (v2).

v1 (`submissions/ff_pretrain_then_sgd/`) used Hinton's Forward-Forward
goodness loss with random-byte-corrupt negatives. Live telemetry from
the v1 run proved the FF stage learned nothing: g_pos and g_neg stayed
within 1 unit of each other for 300 steps; the gate ratio finished at
1.01 (vacuously below the 5.0 threshold); Stage 1's full 80 s of
wall-clock contributed zero to the final 0.7293 acc — that PASS was
just `modded_nanogpt` with 1900/2150 Stage-2 steps.

v2 (this submission) replaces Stage 1 with **Mono-Forward** (Gong, Li,
Abdulla 2025, arXiv 2501.09238): a per-block linear probe head
`H_l : R^d -> R^256` trained with standard cross-entropy on detached
block outputs. No negative samples; no goodness saturation; bounded loss.

Stage 1 (~30 s, ~150 steps cap):
  For each transformer block l in [0, L), compute h = block_l(h_in)
  where h_in is detached from the previous block. Compute next-byte
  logits = H_l(h), then CE loss against the true next byte. Each block
  has its own AdamW; the probe heads have their own joint AdamW.
  Per-block gradient flows only from H_l back through block_l (and the
  embedding receives a small share from block 0).

Stage 1 diagnostic gate:
  Compute val-set probe ensemble accuracy (average final-block logits
  on a 200-char window). If < 0.10 (>> uniform 1/256 = 0.004), gate
  PASSES and Stage 2 inherits the pretrained body. If < 0.10, the
  block-local CE produced no signal; re-init the body and run Stage 2
  from scratch (the baseline path). This gate is structured to *fail
  fast* on the v1 failure mode, unlike v1's vacuous "ratio < 5.0" gate.

Stage 2 (~250 s, n_steps=2150):
  Standard Muon + AdamW CE training, identical to the modded-nanogpt
  baseline. n_steps=2150 matches the baseline exactly (vs v1's 1900),
  so the comparison is "did Stage 1 give a better init?" — not
  "did Stage 2 have less time?". The probe heads `H_l` are discarded
  at the end of Stage 1; only the body weights carry forward.
"""
from __future__ import annotations

__author__ = "@armin-claude-1m"

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


def _init_blocks_only(model: GPT) -> None:
    """Re-init only the block parameters; keep embed/head/norms intact.
    Used by the Stage-1 gate fallback when the probes produced no signal.
    """
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


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Stage 1 — Mono-Forward per-block local pretrain via probe heads
# ---------------------------------------------------------------------------

def _mono_forward_pretrain(
    model: GPT,
    train_bytes: Tensor,
    *,
    batch_size: int,
    max_len: int,
    body_lr: float,
    probe_lr: float,
    max_steps: int,
    max_seconds: float,
    log_every: int,
) -> tuple[list[nn.Linear], dict]:
    """Run Stage-1 Mono-Forward pretrain in-place on `model.blocks`.

    For each step:
      h = embed(x); h = norm1(h)
      for ell in range(L):
          h = block_ell(h_in_detached)           # gradient enabled here
          logits_ell = probe_ell(h)              # head sees graph for block_ell
          loss_ell = CE(logits_ell, y)
          loss_ell.backward()                    # backprops into block_ell + probe_ell
          opts_block[ell].step(); opts_probe[ell].step()
          h = h.detach()                         # block ell+1 starts a fresh graph

    Note on embedding gradient: when ell=0, h_in is `model.norm1(model.embed(x))`,
    which carries gradient into the embedding + norm1. We allow this (it's
    just the natural first block) but the per-block optimizer for block 0
    is constructed to *exclude* embedding/norm1 — we don't want Stage 1 to
    train the embedding via probe gradients only.

    Returns (probe_heads, telemetry).
    """
    device = train_bytes.device
    model_dim = model.embed.embedding_dim
    L = len(model.blocks)

    probe_heads: list[nn.Linear] = [
        nn.Linear(model_dim, 256).to(device) for _ in range(L)
    ]
    for h in probe_heads:
        nn.init.normal_(h.weight, std=0.33**0.5 / model_dim**0.5)
        nn.init.zeros_(h.bias)

    opts_block = [
        AdamW(b.parameters(), lr=body_lr, betas=(0.9, 0.95), weight_decay=0.0)
        for b in model.blocks
    ]
    opts_probe = [
        AdamW(h.parameters(), lr=probe_lr, betas=(0.9, 0.95), weight_decay=0.0)
        for h in probe_heads
    ]

    telemetry: dict = {
        "steps": 0,
        "duration_s": 0.0,
        "loss_last": [None] * L,
        "acc_last": [None] * L,
        "grad_norm_last": [None] * L,
    }

    n = train_bytes.numel()
    use_amp = device.type == "cuda"
    model.train()
    t_start = time.monotonic()
    print(
        f"[mono] Stage-1 begin  L={L}  body_lr={body_lr:.1e}  "
        f"probe_lr={probe_lr:.1e}  max_steps={max_steps}  "
        f"budget={max_seconds:.0f}s  batch={batch_size}  T={max_len}",
        flush=True,
    )

    step = 0
    while step < max_steps:
        if time.monotonic() - t_start > max_seconds:
            break

        idx = torch.randint(0, n - max_len - 1, (batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(max_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        # Each block ell gets its own forward+backward graph. For ell=0,
        # the inbound h is `norm1(embed(x))` which has a fresh graph. For
        # ell>0, the inbound h is `block_{ell-1}(...)` detached.
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _NullCtx()
        with ctx:
            h = model.norm1(model.embed(x))

        for ell in range(L):
            h_in = h.detach().requires_grad_(False)
            with ctx:
                h_out, _ = model.blocks[ell](h_in)
                # Probe head in fp32 for numeric stability of CE.
                logits = probe_heads[ell](h_out.float())
                loss = F.cross_entropy(
                    logits.reshape(-1, 256), y.reshape(-1)
                )

            opts_block[ell].zero_grad(set_to_none=True)
            opts_probe[ell].zero_grad(set_to_none=True)
            loss.backward()

            if log_every and (step % log_every == 0 or step == max_steps - 1):
                # Cheap grad norm on the block's first 2-D weight.
                gn = 0.0
                for p in model.blocks[ell].parameters():
                    if p.grad is not None:
                        gn += float(p.grad.detach().float().norm().item()) ** 2
                gn = gn ** 0.5
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    acc = (pred == y).float().mean().item()
                telemetry["loss_last"][ell] = float(loss.detach().item())
                telemetry["acc_last"][ell] = float(acc)
                telemetry["grad_norm_last"][ell] = gn

            opts_block[ell].step()
            opts_probe[ell].step()

            # Hand off h_out to next block, with gradient severed.
            h = h_out.detach()

        step += 1
        if log_every and (step % log_every == 0 or step == 1 or step == max_steps):
            elapsed = time.monotonic() - t_start
            loss_s = "  ".join(
                f"{(v if v is not None else 0):.3f}" for v in telemetry["loss_last"]
            )
            acc_s = "  ".join(
                f"{(v if v is not None else 0):.3f}" for v in telemetry["acc_last"]
            )
            gn_s = "  ".join(
                f"{(v if v is not None else 0):.2f}" for v in telemetry["grad_norm_last"]
            )
            print(
                f"[mono] step {step:4d}  t={elapsed:5.1f}s  "
                f"loss=[{loss_s}]  acc=[{acc_s}]  ||g||=[{gn_s}]",
                flush=True,
            )

    telemetry["steps"] = step
    telemetry["duration_s"] = time.monotonic() - t_start
    print(
        f"[mono] Stage-1 end    steps={step}  "
        f"duration={telemetry['duration_s']:.1f}s",
        flush=True,
    )
    return probe_heads, telemetry


@torch.no_grad()
def _stage1_probe_eval(
    model: GPT,
    probe_heads: list[nn.Linear],
    val_bytes: Tensor,
    *,
    max_len: int,
    batch_size: int = 16,
    n_val_chars: int = 200,
) -> float:
    """Quick eval: run a single batch through the body, ensemble the probe
    head logits across blocks (mean of softmaxes), argmax against true y.
    Returns char-acc on `n_val_chars` characters total.

    Uses random windows from val_bytes — purely for the diagnostic gate.
    """
    device = val_bytes.device
    n = val_bytes.numel()
    L = len(model.blocks)

    # Choose window length and batch so that batch_size * (max_len-1) >= n_val_chars.
    # Use a short window — we just want a quick estimate.
    win = min(max_len, 128)
    n_windows = max(1, (n_val_chars + (win - 1) - 1) // (win - 1))
    n_windows = min(n_windows, batch_size)
    # If a single batch isn't enough, take multiple batches.
    iters = max(1, (n_val_chars + n_windows * (win - 1) - 1) // (n_windows * (win - 1)))

    model.eval()
    use_amp = device.type == "cuda"
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _NullCtx()

    total_correct = 0
    total_chars = 0
    for _ in range(iters):
        if n < win + 1:
            break
        idx = torch.randint(0, n - win - 1, (n_windows,), device=device)
        offsets = idx[:, None] + torch.arange(win + 1, device=device)[None, :]
        flat = val_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        with ctx:
            h = model.norm1(model.embed(x))
            ensembled = None
            for ell in range(L):
                h, _ = model.blocks[ell](h)
                logits = probe_heads[ell](h.float())
                probs = F.softmax(logits, dim=-1)
                ensembled = probs if ensembled is None else ensembled + probs
            assert ensembled is not None
            pred = ensembled.argmax(dim=-1)
        total_correct += int((pred == y).sum().item())
        total_chars += int(y.numel())

    model.train()
    return total_correct / max(1, total_chars)


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
    print(
        f"[stage2] begin  n_steps={n_steps}  budget={max_seconds:.0f}s",
        flush=True,
    )
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

class MonoForwardCharModel(CharModel):
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
        print(f"[mono_forward_v2] SEED={seed}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Architecture / training hyperparameters --------------------------
    model_dim = 384
    num_layers = 6
    head_dim = 64
    max_len = 1024
    batch_size = 32

    # Stage 1 (Mono-Forward)
    # Budget ~30 s. At ~150-200 ms per multi-block step (6 blocks each with
    # their own fwd+bwd), expect ~150 steps. The spec says ~25 ms per
    # block-local step => 6 blocks/step * 25 ms = ~150 ms/step. Conservative.
    mono_max_seconds = 30.0
    mono_max_steps = 250
    mono_body_lr = 1e-3
    mono_probe_lr = 3e-3
    # Smaller T for Stage 1 — more steps per second; per-token CE doesn't
    # need full 1024-byte context to give signal.
    mono_max_len = 256
    mono_batch_size = 32
    mono_log_every = 25

    # Stage-1 gate threshold: spec says 0.10 (better than uniform 1/256).
    mono_gate_acc = 0.10

    # Stage 2 (CE) — matched to baseline 2150 steps.
    stage2_budget_seconds = 255.0
    stage2_n_steps = 2150
    cooldown_frac = 0.7
    embed_lr = 0.3
    head_lr = 1.0 / 320
    scalar_lr = 0.01
    muon_lr = 0.035
    muon_wd = 0.025
    log_every = 100

    # ---- Data -------------------------------------------------------------
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < max_len + 1:
        raise ValueError(f"need at least {max_len+1} bytes; got {n}")

    if valid_text is not None:
        raw_v = valid_text.encode("utf-8")
        val_bytes = torch.frombuffer(bytearray(raw_v), dtype=torch.uint8).to(device)
    else:
        # Use last 1% of train as a proxy val for the Stage-1 gate.
        cut = max(1, n // 100)
        val_bytes = train_bytes[-cut:]

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
    print(
        f"[mono_forward_v2] {n_params/1e6:.2f}M params  dim={model_dim} "
        f"L={num_layers} bs={batch_size} T={max_len}",
        flush=True,
    )

    # ---- Stage 1: Mono-Forward pretrain ----------------------------------
    probe_heads, mono_telemetry = _mono_forward_pretrain(
        model,
        train_bytes,
        batch_size=mono_batch_size,
        max_len=mono_max_len,
        body_lr=mono_body_lr,
        probe_lr=mono_probe_lr,
        max_steps=mono_max_steps,
        max_seconds=mono_max_seconds,
        log_every=mono_log_every,
    )

    # ---- Stage-1 diagnostic gate -----------------------------------------
    stage1_acc = _stage1_probe_eval(
        model, probe_heads, val_bytes,
        max_len=mono_max_len, batch_size=16, n_val_chars=200,
    )
    print(
        f"[stage1] probe-ensemble val_acc = {stage1_acc:.4f}  "
        f"(gate threshold = {mono_gate_acc:.2f}; uniform = {1/256:.4f})",
        flush=True,
    )

    if stage1_acc < mono_gate_acc:
        print(
            "[stage1] FAILED — probe heads produced no signal; "
            "re-init blocks (baseline-equivalent Stage 2)",
            flush=True,
        )
        _init_blocks_only(model)
    else:
        print(
            "[stage1] PASSED — proceeding to Stage 2 with Mono-Forward init",
            flush=True,
        )

    # Drop probe heads — they are not used in Stage 2 or eval.
    del probe_heads

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

    return MonoForwardCharModel(model)
