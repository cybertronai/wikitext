"""ES on a *micro* char-Transformer (pass 2) — gradient-free training via
antithetic OpenAI-ES with centered rank-shaping (Wierstra NES style) and
linear sigma annealing, on a much smaller model + full 270 s budget.

Architecture and streaming-inference wrapper are reused from pass 1; the
ES loop is upgraded per the pass-2 spec.

Spec: .survey/designs/method_es-tiny-transformer_pass_2.md
"""
from __future__ import annotations

__author__ = "@survey-es-p2"

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
    """Pass-2 MLP — narrower hidden dim (2x rather than 4x) per spec."""

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        hdim = hidden_mult * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, mlp_mult: int = 2):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim=head_dim)
        self.mlp = MLP(dim, hidden_mult=mlp_mult)
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
        max_len: int = 32,
        mlp_mult: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        # NB: bf16 embedding (matches modded baseline).
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(
            [Block(model_dim, head_dim=head_dim, mlp_mult=mlp_mult)
             for _ in range(num_layers)]
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
def _flatten_params(
    model: nn.Module,
) -> tuple[Tensor, list[tuple[str, torch.Size, int]]]:
    parts: list[Tensor] = []
    layout: list[tuple[str, torch.Size, int]] = []
    for name, p in model.named_parameters():
        flat = p.detach().to(dtype=torch.float32).reshape(-1)
        parts.append(flat)
        layout.append((name, p.shape, flat.numel()))
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
    """
    from torch.func import functional_call
    params = _theta_to_param_dict(theta_perturbed, layout, param_dtypes)
    buffers = {n: b for n, b in model.named_buffers()}
    state = {**params, **buffers}
    logits, _ = functional_call(model, state, (x,))
    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
    return float(loss.item())


@torch.no_grad()
def _centered_rank_weights(P: int, device: torch.device) -> Tensor:
    """Wierstra-NES centered rank weights.

    For rank i in 1..P (best=1, worst=P):
        u_i = max(0, log(P/2 + 1) - log(i))
    Then normalise: u_i /= sum(u_i), then center to zero mean.
    Returns a 1-D tensor of shape [P] indexed by rank (rank 0 == best).
    """
    ranks = torch.arange(1, P + 1, device=device, dtype=torch.float32)
    log_half = math.log(P / 2 + 1)
    u = torch.clamp(log_half - torch.log(ranks), min=0.0)
    u = u / u.sum()
    u = u - u.mean()  # zero-mean → no bias drift
    return u


def _train_es(
    text: str,
    device: torch.device,
    *,
    n_layers: int = 2,
    d_model: int = 32,
    head_dim: int = 32,
    mlp_mult: int = 2,
    ctx_len: int = 32,
    population: int = 128,
    sigma_start: float = 0.05,
    sigma_end: float = 0.01,
    alpha: float = 0.03,
    batch_size: int = 32,
    time_budget_s: float = 260.0,
    iters_estimate: int = 740,
    log_every: int = 25,
) -> GPT:
    """Antithetic OpenAI-ES with centered rank shaping + sigma anneal.

    Hyperparameters from pass-2 spec. The loop runs until time_budget_s
    of wall time elapses (no iter cap) — sigma is annealed linearly
    over an *expected* iters_estimate horizon (truncated to current iter
    for sigma but iters keep going).
    """
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
        mlp_mult=mlp_mult,
    ).to(device)
    _init_modded(model)
    model.eval()  # no autograd; pure forward.

    param_dtypes = {name: p.dtype for name, p in model.named_parameters()}

    theta, layout = _flatten_params(model)
    D = theta.numel()
    P = population
    half = P // 2

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[es-p2] model: L={n_layers} d={d_model} h={head_dim} "
        f"mlp_mult={mlp_mult} ctx={ctx_len}  params={n_params:,}  D={D:,}",
        flush=True,
    )
    print(
        f"[es-p2] population P={P} (antithetic half={half}) "
        f"sigma_start={sigma_start} sigma_end={sigma_end} alpha={alpha} "
        f"batch={batch_size} budget={time_budget_s:.0f}s "
        f"iters_estimate={iters_estimate}",
        flush=True,
    )

    # Centered rank weights — best gets the largest positive weight, worst
    # gets a (small) negative weight. Zero-mean, so antithetic noise +
    # symmetric weighting gives an unbiased rank-based gradient estimate.
    rank_w = _centered_rank_weights(P, device)  # [P], rank 0 = best
    print(
        f"[es-p2] rank weights: min={float(rank_w.min()):.4f} "
        f"max={float(rank_w.max()):.4f} "
        f"||w||_1={float(rank_w.abs().sum()):.4f}",
        flush=True,
    )

    t0 = time.monotonic()
    iters_done = 0
    last_log_nll = float("nan")
    last_sigma = sigma_start
    last_update_inf = 0.0

    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= time_budget_s:
            break

        # Sigma anneal: linear from sigma_start → sigma_end over
        # iters_estimate. After iters_estimate the floor sigma_end holds.
        frac = min(1.0, iters_done / max(1, iters_estimate))
        sigma_t = sigma_start + (sigma_end - sigma_start) * frac
        last_sigma = sigma_t

        # Sample a common minibatch (CRN across population).
        idx = torch.randint(0, n - ctx_len - 1, (batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(ctx_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        # Antithetic noise [P, D].
        eps_half = torch.randn(half, D, device=device, dtype=torch.float32)
        eps = torch.cat([eps_half, -eps_half], dim=0)

        # Sequential forward over the P perturbations (vmap+functional_call
        # collides with SDPA causal kernels; the loop is fast at D≈33k).
        nlls = torch.empty(P, device=device, dtype=torch.float32)
        for i in range(P):
            theta_i = theta + sigma_t * eps[i]
            nll = _eval_perturbed(
                model, theta_i, layout, param_dtypes, x, y
            )
            nlls[i] = nll

        # Rank: smallest NLL = best. Best gets rank 0 weight (largest +).
        order = torch.argsort(nlls, descending=False)
        f_weighted = torch.empty(P, device=device, dtype=torch.float32)
        f_weighted[order] = rank_w  # order[0] = best index → gets rank_w[0]

        # update = (alpha / sigma) * sum_i w_i * eps_i
        # (rank weights are already sum-normalised; divide by sigma_t for
        # the standard NES-style scaling.)
        update = (alpha / sigma_t) * (f_weighted[:, None] * eps).sum(dim=0)
        update.clamp_(-5.0 * sigma_t, 5.0 * sigma_t)
        theta.add_(update)
        last_update_inf = float(update.abs().max().item())

        iters_done += 1
        last_log_nll = float(nlls.mean().item())
        if log_every and (iters_done % log_every == 0 or iters_done == 1):
            elapsed = time.monotonic() - t0
            best = float(nlls.min().item())
            worst = float(nlls.max().item())
            print(
                f"[es-p2] iter {iters_done:4d}  "
                f"mean_NLL={last_log_nll:.4f}  best={best:.4f}  "
                f"worst={worst:.4f}  sigma={sigma_t:.4f}  "
                f"||theta||={float(theta.norm().item()):.2f}  "
                f"||upd||_inf={last_update_inf:.4f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.monotonic() - t0
    print(
        f"[es-p2] training done — iters={iters_done} elapsed={elapsed:.1f}s "
        f"final_sigma={last_sigma:.4f} final_mean_NLL={last_log_nll:.4f}",
        flush=True,
    )

    _write_theta_into_model(theta, model, layout)
    # Stash some diagnostics for inspection from outside.
    model._es_iters_done = iters_done  # type: ignore[attr-defined]
    model._es_final_nll = last_log_nll  # type: ignore[attr-defined]
    model._es_final_sigma = last_sigma  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper.
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
        print(f"[es-p2] SEED={seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _train_es(train_text, device)
    return ESTinyTransformerCharModel(model)
