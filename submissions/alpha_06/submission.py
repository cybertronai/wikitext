"""alpha_06 — Clean hybrid (E3 NN + W31 GPU order-12 KN n-gram) at α=0.60.

Finer α sweep below alpha_065 (0.7407 current best clean acc). The α
curve looks concave on [0.5, 0.8]; testing α=0.60 (more n-gram weight)
to bracket the peak.

Architecture identical to alpha_065; only ALPHA changes (0.65 → 0.60).
"""
from __future__ import annotations

__author__ = "@subagent-xorfix-2026-05-19"

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ===========================================================================
# Constants
# ===========================================================================

# Brief specifies order-12, matching W31's default.
MAX_ORDER = 12
MAX_CTX_LEN = MAX_ORDER - 1
KN_DISCOUNT = 0.5

# Hybrid mixer constant: NN weight. Finer sweep below α=0.65; testing α=0.60.
ALPHA: float = 0.60


# ===========================================================================
# Part 1 — W31 GPU KN build (verbatim from gpu_ngram_w3/submission.py).
# ===========================================================================


def _pack_window_chunk(
    arr_int64: Tensor,
    start: int,
    end: int,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Pack k-byte windows into (hi, lo) int64 pairs. k>8 splits the key
    across two int64s; k<=8 packs entirely into ``lo`` with ``hi=0``."""
    n = end - start
    m = n - k + 1
    if m <= 0:
        device = arr_int64.device
        return (torch.zeros(0, dtype=torch.int64, device=device),
                torch.zeros(0, dtype=torch.int64, device=device))
    chunk = arr_int64[start:end]
    device = chunk.device
    if k <= 8:
        lo = torch.zeros(m, dtype=torch.int64, device=device)
        for j in range(k):
            lo = (lo << 8) | chunk[j:j + m]
        hi = torch.zeros(m, dtype=torch.int64, device=device)
    else:
        hi = torch.zeros(m, dtype=torch.int64, device=device)
        for j in range(k - 8):
            hi = (hi << 8) | chunk[j:j + m]
        lo = torch.zeros(m, dtype=torch.int64, device=device)
        for j in range(k - 8, k):
            lo = (lo << 8) | chunk[j:j + m]
    return hi, lo


def _sort_and_dedupe(
    hi: Tensor, lo: Tensor, counts: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if hi.numel() == 0:
        return hi, lo, counts
    device = hi.device
    order_lo = torch.argsort(lo, stable=True)
    hi = hi[order_lo]
    lo = lo[order_lo]
    counts = counts[order_lo]
    order_hi = torch.argsort(hi, stable=True)
    hi = hi[order_hi]
    lo = lo[order_hi]
    counts = counts[order_hi]
    n = hi.numel()
    change = torch.ones(n, dtype=torch.bool, device=device)
    change[1:] = (hi[1:] != hi[:-1]) | (lo[1:] != lo[:-1])
    group_id = torch.cumsum(change.to(torch.int64), dim=0) - 1
    n_groups = int(group_id[-1].item()) + 1
    merged_hi = hi[change]
    merged_lo = lo[change]
    merged_counts = torch.zeros(n_groups, dtype=torch.float32, device=device)
    merged_counts.scatter_add_(0, group_id, counts)
    return merged_hi, merged_lo, merged_counts


def _build_top_order_gpu(
    train_bytes_u8: Tensor,
    k: int,
    chunk_bytes: int = 32 * 1024 * 1024,
) -> tuple[Tensor, Tensor, Tensor]:
    device = train_bytes_u8.device
    n = train_bytes_u8.numel()
    if n < k:
        empty_i = torch.zeros(0, dtype=torch.int64, device=device)
        empty_f = torch.zeros(0, dtype=torch.float32, device=device)
        return empty_i, empty_i.clone(), empty_f
    arr_int64 = train_bytes_u8.to(torch.int64)
    agg_hi = torch.zeros(0, dtype=torch.int64, device=device)
    agg_lo = torch.zeros(0, dtype=torch.int64, device=device)
    agg_counts = torch.zeros(0, dtype=torch.float32, device=device)
    start = 0
    while start < n:
        end = min(n, start + chunk_bytes)
        if end - start < k:
            if end >= n:
                break
            start = end - (k - 1)
            continue
        hi, lo = _pack_window_chunk(arr_int64, start, end, k)
        cnt = torch.ones(hi.numel(), dtype=torch.float32, device=device)
        hi, lo, cnt = _sort_and_dedupe(hi, lo, cnt)
        if agg_hi.numel() == 0:
            agg_hi, agg_lo, agg_counts = hi, lo, cnt
        else:
            all_hi = torch.cat([agg_hi, hi])
            all_lo = torch.cat([agg_lo, lo])
            all_cnt = torch.cat([agg_counts, cnt])
            agg_hi, agg_lo, agg_counts = _sort_and_dedupe(all_hi, all_lo, all_cnt)
        if end >= n:
            break
        start = end - (k - 1)
    return agg_hi, agg_lo, agg_counts


def _step_down_gpu(
    hi: Tensor, lo: Tensor, counts: Tensor, k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if hi.numel() == 0 or k <= 1:
        device = hi.device
        return (torch.zeros(0, dtype=torch.int64, device=device),
                torch.zeros(0, dtype=torch.int64, device=device),
                torch.zeros(0, dtype=torch.float32, device=device))
    new_k = k - 1
    if k > 8:
        if new_k > 8:
            new_hi = hi & ((1 << ((new_k - 8) * 8)) - 1)
            new_lo = lo
        else:
            new_hi = torch.zeros_like(hi)
            new_lo = lo
    else:
        new_hi = torch.zeros_like(hi)
        new_lo = lo & ((1 << (new_k * 8)) - 1)
    return _sort_and_dedupe(new_hi, new_lo, counts)


def _gpu_table_to_w3_layout(
    hi: Tensor, lo: Tensor, counts: Tensor, k: int,
) -> dict:
    """Convert GPU (hi, lo, counts) at order k into the W3 CPU layout
    (ctx_keys, ctx_view, ctx_offsets, next_bytes, counts,
    total_count_per_ctx, n_distinct_per_ctx) — ready to drop into the
    KN predict path.
    """
    ctx_len = k - 1
    n = hi.numel()
    hi_cpu = hi.cpu().numpy()
    lo_cpu = lo.cpu().numpy()
    counts_cpu = counts.cpu().numpy().astype(np.int64)
    bytes_arr = np.zeros((n, k), dtype=np.uint8)
    if n > 0:
        if k > 8:
            hi_bytes = k - 8
            for j in range(hi_bytes):
                shift = (hi_bytes - 1 - j) * 8
                bytes_arr[:, j] = (hi_cpu >> shift) & 0xFF
            for j in range(8):
                shift = (7 - j) * 8
                bytes_arr[:, hi_bytes + j] = (lo_cpu >> shift) & 0xFF
        else:
            for j in range(k):
                shift = (k - 1 - j) * 8
                bytes_arr[:, j] = (lo_cpu >> shift) & 0xFF
    next_arr = bytes_arr[:, ctx_len].copy()
    counts_arr = counts_cpu.astype(np.int32, copy=False)
    if ctx_len == 0:
        return {
            "ctx_len": 0,
            "ctx_keys": np.empty((1, 0), dtype=np.uint8),
            "ctx_view": None,
            "ctx_offsets": np.array([0, n], dtype=np.int64),
            "next_bytes": next_arr,
            "counts": counts_arr,
            "total_count_per_ctx": np.array([int(counts_cpu.sum())], dtype=np.int64),
            "n_distinct_per_ctx": np.array([n], dtype=np.int32),
        }
    ctx_arr = np.ascontiguousarray(bytes_arr[:, :ctx_len])
    ctx_view_full = ctx_arr.view(np.dtype((np.void, ctx_len)))[:, 0]
    if n == 0:
        starts = np.zeros(0, dtype=np.int64)
    else:
        change = np.ones(n, dtype=bool)
        change[1:] = ctx_view_full[1:] != ctx_view_full[:-1]
        starts = np.flatnonzero(change).astype(np.int64)
    n_ctx = starts.shape[0]
    ctx_keys = np.ascontiguousarray(ctx_arr[starts])
    ctx_view = ctx_keys.view(np.dtype((np.void, ctx_len)))[:, 0]
    ctx_offsets = np.empty(n_ctx + 1, dtype=np.int64)
    ctx_offsets[:n_ctx] = starts
    ctx_offsets[n_ctx] = n
    total_per_ctx = (
        np.add.reduceat(counts_cpu, starts) if n_ctx > 0
        else np.zeros(0, dtype=np.int64)
    )
    n_distinct = (ctx_offsets[1:] - ctx_offsets[:-1]).astype(np.int32)
    return {
        "ctx_len": ctx_len,
        "ctx_keys": ctx_keys,
        "ctx_view": ctx_view,
        "ctx_offsets": ctx_offsets,
        "next_bytes": next_arr,
        "counts": counts_arr,
        "total_count_per_ctx": total_per_ctx,
        "n_distinct_per_ctx": n_distinct,
    }


def _build_continuation_base(bigram_next_arr: np.ndarray) -> np.ndarray:
    counts = np.bincount(bigram_next_arr, minlength=256).astype(np.float64)
    s = counts.sum()
    if s > 0:
        counts /= s
    else:
        counts[:] = 1.0 / 256.0
    return counts


def build_w31_kn_tables(
    train_bytes_u8: Tensor, max_order: int = MAX_ORDER,
) -> tuple[list, np.ndarray]:
    """Build W31-style GPU KN tables and transfer to W3 CPU layout."""
    device = train_bytes_u8.device
    t_total = time.monotonic()
    print(f"[clean_w31] starting GPU KN build; max_order={max_order} "
          f"D={KN_DISCOUNT}", flush=True)
    t0 = time.monotonic()
    hi, lo, counts = _build_top_order_gpu(train_bytes_u8, max_order)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[clean_w31] top order={max_order} unique pairs: {hi.numel():,}  "
          f"{time.monotonic()-t0:.1f}s", flush=True)
    order_tables: list = [None] * max_order  # ctx_len 0..MAX_CTX_LEN
    t0 = time.monotonic()
    order_tables[max_order - 1] = _gpu_table_to_w3_layout(hi, lo, counts, max_order)
    print(f"[clean_w31] ctx_len={max_order-1} "
          f"ctxs={order_tables[max_order-1]['ctx_keys'].shape[0]:,} "
          f"{time.monotonic()-t0:.1f}s", flush=True)
    bigram_next_for_base = None
    for new_k in range(max_order - 1, 0, -1):
        t0 = time.monotonic()
        hi, lo, counts = _step_down_gpu(hi, lo, counts, new_k + 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        order_tables[new_k - 1] = _gpu_table_to_w3_layout(hi, lo, counts, new_k)
        tbl = order_tables[new_k - 1]
        print(f"[clean_w31] ctx_len={new_k-1} ctxs={tbl['ctx_keys'].shape[0]:,} "
              f"{time.monotonic()-t0:.1f}s", flush=True)
        if new_k == 2:
            bigram_next_for_base = tbl["next_bytes"].copy()
    if bigram_next_for_base is not None:
        continuation = _build_continuation_base(bigram_next_for_base)
    else:
        continuation = np.full(256, 1.0 / 256.0, dtype=np.float64)
    print(f"[clean_w31] KN build done: {time.monotonic()-t_total:.1f}s",
          flush=True)
    return order_tables, continuation


def kn_distribution(
    order_tables: list, continuation: np.ndarray,
    history: bytes, max_ctx_len: int, discount: float = KN_DISCOUNT,
) -> np.ndarray:
    """KN-interpolated next-byte distribution (same recurrence as W3)."""
    D = discount
    p = continuation.astype(np.float64).copy()
    hist_len = len(history)
    max_k = min(max_ctx_len, hist_len)
    if max_k == 0:
        return p
    for k in range(1, max_k + 1):
        tbl = order_tables[k]
        if tbl is None:
            continue
        ctx_view = tbl["ctx_view"]
        if ctx_view is None or ctx_view.shape[0] == 0:
            continue
        tail = bytes(history[-k:])
        q = np.frombuffer(tail, dtype=np.uint8).view(
            np.dtype((np.void, k)),
        )[0]
        idx = int(np.searchsorted(ctx_view, q))
        if idx >= ctx_view.shape[0] or ctx_view[idx] != q:
            continue
        lo = int(tbl["ctx_offsets"][idx])
        hi = int(tbl["ctx_offsets"][idx + 1])
        nb = tbl["next_bytes"][lo:hi]
        cn = tbl["counts"][lo:hi].astype(np.float64)
        total = float(tbl["total_count_per_ctx"][idx])
        n_distinct = int(tbl["n_distinct_per_ctx"][idx])
        if total <= 0.0:
            continue
        discounted = np.maximum(cn - D, 0.0) / total
        lam = D * n_distinct / total
        p_new = lam * p
        p_new[nb] = p_new[nb] + discounted
        p = p_new
    return p


# ===========================================================================
# Part 2 — modded-nanogpt NN (verbatim from nano_plus_ngram).
# ===========================================================================


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

    def forward(self, x, kv_cache=None, offset=0):
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

    def forward(self, x):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim=head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, kv_cache=None, offset=0):
        h, new_kv = self.attn(self.norm1(x), kv_cache, offset=offset)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, head_dim=64, max_len=1024):
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

    def forward(self, inputs, kv_caches=None, offset=0):
        x = self.norm1(self.embed(inputs))
        new_caches = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv, offset=offset)
            new_caches.append(new_kv)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits, new_caches


def zeropower_via_newtonschulz5(G):
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


def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.0, mu=0.95):
        params = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
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


def _init_modded(model):
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


class TrainConfig:
    def __init__(
        self,
        model_dim=256,
        num_layers=4,
        head_dim=64,
        max_len=1024,
        batch_size=32,
        n_steps=1200,
        cooldown_frac=0.7,
        embed_lr=0.3,
        head_lr=1.0 / 320,
        scalar_lr=0.01,
        muon_lr=0.035,
        muon_wd=0.025,
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
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"H={self.model_dim//self.head_dim} bs={self.batch_size} "
                f"T={self.max_len} steps={self.n_steps})")


def _train_modded(
    train_bytes_gpu: Tensor, cfg: TrainConfig, device: torch.device,
) -> GPT:
    n = train_bytes_gpu.numel()
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
    print(f"[clean_w31] NN {n_params/1e6:.2f}M params  cfg={cfg}")

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
        flat = train_bytes_gpu[offsets].long()
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
                f"[clean_w31] NN step {step:5d}/{cfg.n_steps}  "
                f"loss {loss.item():.4f}  elapsed {elapsed:.0f}s",
                flush=True,
            )
    return model


# ===========================================================================
# Part 3 — Streaming hybrid CharModel.
# ===========================================================================


class CleanHybridW31CharModel(CharModel):
    """E3-style NN + W31 GPU KN n-gram mixed at α=0.7."""

    def __init__(
        self,
        model: GPT,
        order_tables: list,
        continuation: np.ndarray,
        max_ctx_len: int = MAX_CTX_LEN,
        discount: float = KN_DISCOUNT,
        alpha: float = ALPHA,
        device: torch.device | None = None,
    ):
        self.model = model
        self.order_tables = order_tables
        self.continuation = continuation
        self.max_ctx_len = max_ctx_len
        self.discount = float(discount)
        self.alpha = float(alpha)
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self._kv: list[tuple[Tensor, Tensor]] | None = None
        self._next_logits: Tensor | None = None
        self._pos: int = 0
        self._history: bytearray = bytearray()

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = None
        self._pos = 0
        self._history = bytearray()
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        logits, self._kv = self.model(x, None, offset=self._pos)
        self._next_logits = logits[0, -1]
        self._pos = 1

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        if self._next_logits is None:
            raise RuntimeError("predict() called before reset()")
        p_nn = F.softmax(self._next_logits.float(), dim=-1).cpu().numpy()
        p_kn = kn_distribution(
            self.order_tables, self.continuation, bytes(self._history),
            max_ctx_len=self.max_ctx_len, discount=self.discount,
        ).astype(np.float32)
        p_mix = self.alpha * p_nn + (1.0 - self.alpha) * p_kn
        out: dict[str, float] = {}
        for byte_id in range(256):
            p = float(p_mix[byte_id])
            if p <= 0.0:
                continue
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
            self._history.append(byte)
            if len(self._history) > self.max_ctx_len:
                del self._history[: len(self._history) - self.max_ctx_len]

    def _maybe_trim_cache(self) -> None:
        if self._kv is None:
            return
        cur = self._kv[0][0].shape[2]
        if cur < self.model.max_len:
            return
        keep = self.model.max_len - 1
        self._kv = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in self._kv]


# ===========================================================================
# Entry point
# ===========================================================================

SMOKE_TRAIN_BYTES = 10_000


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[clean_w31] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = train_text.encode("utf-8")
    is_smoke = len(raw) < SMOKE_TRAIN_BYTES

    train_bytes_u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)

    if is_smoke:
        kn_max_order = max(2, min(MAX_ORDER, len(raw) // 32))
        seq = max(8, min(64, len(raw) // 4))
        cfg = TrainConfig(
            model_dim=64,
            num_layers=2,
            head_dim=32,
            max_len=seq,
            batch_size=2,
            n_steps=4,
            log_every=0,
        )
        print(f"[clean_w31] SMOKE mode (train={len(raw)} bytes)  "
              f"NN steps={cfg.n_steps}  kn_max_order={kn_max_order}")
    else:
        kn_max_order = MAX_ORDER
        cfg = TrainConfig()

    # Phase A: GPU KN build (W31 pattern).
    order_tables, continuation = build_w31_kn_tables(
        train_bytes_u8, max_order=kn_max_order,
    )

    # Phase B: GPU NN train (E3 pattern).
    model = _train_modded(train_bytes_u8, cfg, device)

    return CleanHybridW31CharModel(
        model, order_tables, continuation,
        max_ctx_len=kn_max_order - 1, discount=KN_DISCOUNT,
        alpha=ALPHA, device=device,
    )
