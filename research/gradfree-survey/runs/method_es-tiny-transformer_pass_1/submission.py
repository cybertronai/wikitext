"""ES on a tiny char-Transformer (pass 1) — gradient-free training via
antithetic OpenAI-ES rank-normalised updates on a flattened parameter vector.

Architecture and streaming-inference wrapper are reused from
``submissions/modded_nanogpt/submission.py``; only the training loop is
replaced with evolution strategies.

Spec: .survey/designs/method_es-tiny-transformer_pass_1.md
"""
from __future__ import annotations

__author__ = "@survey-es"

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Architecture (mirrors modded_nanogpt simple, ported verbatim; no caches in
# the ES training path — caches are only used by the streaming CharModel).
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


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 32):
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
        head_dim: int = 32,
        max_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        # NB: bf16 embedding (matches modded baseline).
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
# Init scheme (mirrors modded init: zero proj, normal embed, scaled others).
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
            w.fill_(1.0)
        else:
            raise RuntimeError(f"Uninitialized parameter: {name}")


# ---------------------------------------------------------------------------
# Evolution strategies trainer (gradient-free; no optimizer, no backprop).
# ---------------------------------------------------------------------------

@torch.no_grad()
def _flatten_params(model: nn.Module) -> tuple[Tensor, list[tuple[str, torch.Size, int]]]:
    """Concatenate every parameter into a single fp32 vector on CUDA.

    Returns the flat vector and a layout descriptor so we can splat
    perturbed copies back in via ``functional_call``.
    """
    parts: list[Tensor] = []
    layout: list[tuple[str, torch.Size, int]] = []
    offset = 0
    for name, p in model.named_parameters():
        flat = p.detach().to(dtype=torch.float32).reshape(-1)
        parts.append(flat)
        layout.append((name, p.shape, flat.numel()))
        offset += flat.numel()
    theta = torch.cat(parts).contiguous()
    return theta, layout


@torch.no_grad()
def _theta_to_param_dict(
    theta: Tensor,
    layout: list[tuple[str, torch.Size, int]],
    param_dtypes: dict[str, torch.dtype],
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    off = 0
    for name, shape, n in layout:
        chunk = theta[off:off + n].view(shape).to(dtype=param_dtypes[name])
        out[name] = chunk
        off += n
    return out


@torch.no_grad()
def _write_theta_into_model(
    theta: Tensor,
    model: nn.Module,
    layout: list[tuple[str, torch.Size, int]],
) -> None:
    off = 0
    name_to_param = dict(model.named_parameters())
    for name, shape, n in layout:
        p = name_to_param[name]
        p.data.copy_(theta[off:off + n].view(shape).to(dtype=p.dtype))
        off += n


@torch.no_grad()
def _eval_perturbed(
    model: GPT,
    theta_perturbed: Tensor,
    layout: list[tuple[str, torch.Size, int]],
    param_dtypes: dict[str, torch.dtype],
    x: Tensor,
    y: Tensor,
) -> float:
    """One forward of ``model`` with parameters loaded from ``theta_perturbed``.

    Returns mean cross-entropy NLL (lower = better; fitness = -NLL).
    Implementation: stateless ``functional_call`` so we never touch
    ``model.parameters()`` directly (and never create an autograd graph,
    since the whole function runs under ``torch.no_grad()``).
    """
    from torch.func import functional_call
    params = _theta_to_param_dict(theta_perturbed, layout, param_dtypes)
    # Also pass the rotary buffer through so functional_call doesn't try to
    # re-register it; named_buffers() includes it on the module.
    buffers = {n: b for n, b in model.named_buffers()}
    state = {**params, **buffers}
    logits, _ = functional_call(model, state, (x,))
    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
    return float(loss.item())


def _train_es(
    text: str,
    device: torch.device,
    *,
    n_layers: int = 4,
    d_model: int = 64,
    head_dim: int = 32,
    ctx_len: int = 64,
    population: int = 64,
    sigma: float = 0.02,
    alpha: float = 0.05,
    batch_size: int = 16,
    n_iters_target: int = 150,
    time_budget_s: float = 270.0,
    log_every: int = 10,
) -> GPT:
    assert population % 2 == 0, "antithetic sampling requires even population"

    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < ctx_len + 1:
        raise ValueError(f"need at least {ctx_len + 1} bytes; got {n}")

    model = GPT(
        vocab_size=256,
        num_layers=n_layers,
        model_dim=d_model,
        head_dim=head_dim,
        max_len=ctx_len,
    ).to(device)
    _init_modded(model)
    model.eval()  # no autograd; pure forward.

    # Capture per-parameter dtypes so we round-trip through fp32 theta
    # without silently up-casting (matters for embed.weight, which is bf16).
    param_dtypes = {name: p.dtype for name, p in model.named_parameters()}

    theta, layout = _flatten_params(model)
    D = theta.numel()
    P = population
    half = P // 2

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[es] model: L={n_layers} d={d_model} h={head_dim} ctx={ctx_len} "
        f"params={n_params:,}  D={D:,}",
        flush=True,
    )
    print(
        f"[es] population P={P} (antithetic half={half}) sigma={sigma} "
        f"alpha={alpha} batch={batch_size} budget={time_budget_s:.0f}s "
        f"iters_target={n_iters_target}",
        flush=True,
    )

    # Rank-norm template: ranks 0..P-1 mapped to [-0.5, +0.5], then
    # standardized to unit std. Constant per-iter — precompute once.
    ranks = torch.arange(P, device=device, dtype=torch.float32)
    rank_centered = ranks / (P - 1) - 0.5
    rank_centered = rank_centered / (rank_centered.std() + 1e-12)

    t0 = time.monotonic()
    iters_done = 0
    last_log_nll = float("nan")

    # Reserve some time for final flush / inference setup.
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= time_budget_s:
            break
        if iters_done >= n_iters_target:
            # Used our planned step budget; stop even if there's wall time
            # left (avoid runaway when iterations turn out fast).
            break

        # Sample one shared minibatch (common-random-numbers across P).
        idx = torch.randint(0, n - ctx_len - 1, (batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(ctx_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        # Antithetic noise [P, D].
        eps_half = torch.randn(half, D, device=device, dtype=torch.float32)
        eps = torch.cat([eps_half, -eps_half], dim=0)  # [P, D]

        # Evaluate each perturbed model sequentially (vmap+functional_call
        # with stacked attention kernels is fragile; the loop is fast enough
        # at D≈230k and matches the spec's documented fallback).
        nlls = torch.empty(P, device=device, dtype=torch.float32)
        for i in range(P):
            theta_i = theta + sigma * eps[i]
            nll = _eval_perturbed(model, theta_i, layout, param_dtypes, x, y)
            nlls[i] = nll

        # Fitness = -NLL → rank ascending by fitness means ascending by -NLL,
        # i.e. descending by NLL. Sort NLLs ascending; lowest NLL = best.
        order = torch.argsort(nlls, descending=False)  # best first
        # f_norm[i] in the order: best gets +0.5/std, worst gets -0.5/std.
        f_norm = torch.empty(P, device=device, dtype=torch.float32)
        # rank_centered is sorted ascending (worst→best mapping below).
        # We want best→+, worst→-, so reverse: idx 0 = best gets the +max.
        f_norm[order] = rank_centered.flip(0)

        # dtheta = (alpha / (P * sigma)) * sum_i f_norm_i * eps_i
        update = (alpha / (P * sigma)) * (f_norm[:, None] * eps).sum(dim=0)
        # Clamp per-coord update to 5*sigma to defuse runaway directions.
        update.clamp_(-5.0 * sigma, 5.0 * sigma)
        theta.add_(update)

        iters_done += 1
        last_log_nll = float(nlls.mean().item())
        if log_every and (iters_done % log_every == 0 or iters_done == 1):
            elapsed = time.monotonic() - t0
            best = float(nlls.min().item())
            worst = float(nlls.max().item())
            print(
                f"[es] iter {iters_done:4d}  "
                f"mean_NLL={last_log_nll:.4f}  best={best:.4f}  worst={worst:.4f}  "
                f"||theta||={float(theta.norm().item()):.2f}  "
                f"||update||_inf={float(update.abs().max().item()):.4f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.monotonic() - t0
    print(
        f"[es] training done — iters={iters_done} elapsed={elapsed:.1f}s "
        f"final_mean_NLL={last_log_nll:.4f}",
        flush=True,
    )

    # Write the trained theta back into the model for inference.
    _write_theta_into_model(theta, model, layout)
    return model


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper (ported from modded baseline, unchanged).
# ---------------------------------------------------------------------------

class ESTinyTransformerCharModel(CharModel):
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
# Entry point.
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[es] SEED={seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _train_es(train_text, device)
    return ESTinyTransformerCharModel(model)
