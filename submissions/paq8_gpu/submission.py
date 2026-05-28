"""paq8_gpu — KN base + diverse PAQ8-family mixer for byte-level LM.

Experiment N4 from research/gradfree_analysis.md.

Hypothesis:
  `paq_mixer_v3` reached 3.58 kJ / 0.7048 with 11 independent-WB byte
  orders + a 22-feature mixer. PAQ8 / CMIX hit ~0.12 bpc on enwik8 via
  two compounding moves: (a) KN-interpolated chained base (the path
  `gpu_ngram_o14_xorfix` took to 0.7184), (b) orthogonal context
  families on top — sparse skip contexts, case-folded contexts, and a
  match-model. We combine these here with a tiny logistic mixer.

Architecture:
  1. KN base distribution from standard byte n-grams orders 1..12.
     Computed once per predict step (CPU lookups on sorted ctx_view).
  2. PAQ8 model families (each contributes its own 256-vector):
       (a) skip-1 sparse context: previous 5 bytes with a 1-byte gap
           — captures word-stem regularities through whitespace.
       (b) case-folded byte order-7: lowercase byte stream, then
           order-7 n-gram. Captures stem regularities independent of
           case.
       (c) match model: scan last MATCH_HISTORY_BYTES of stream for
           the longest suffix match length >= MATCH_MIN_LEN; predict
           the byte that followed it in history with high mass.
  3. Mixer: 2-layer MLP, 14 features per family (so 14 * K = 56), with
     K = 4 (KN-base, skip1, case7, match). Hidden = 32, output = K
     softmax weights. ~2K params. Trained on 200K held-out positions.

Target:
  Sub-5-kJ / >= 0.72 displaces nothing on the leaderboard but extends
  the PAQ8 Pareto corner. Stretch: >= 0.74 to displace alpha_06.
"""
from __future__ import annotations

__author__ = "@worker-paq8-gpu"

import os
import subprocess
import sys
import time

# Workaround: as of 2026-05-25 the Modal image at ghcr.io/ab-10/wikitext-bench:latest
# is missing numpy. Install it inline if not present.
try:
    import numpy as np
except ImportError:
    print("[paq8] numpy not available; installing via pip", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--quiet", "numpy==2.1.3"])
    import numpy as np

import torch
from torch import Tensor

from wikitext import CharModel


# ===========================================================================
# Constants
# ===========================================================================

# KN base
MAX_ORDER = 11  # ctx_len = MAX_ORDER - 1 (matched to paq_mixer_v3 build cost)
MAX_CTX_LEN = MAX_ORDER - 1
KN_DISCOUNT = 0.5

# PAQ8 family configs
SKIP1_LEN = 5  # context length (5 bytes); with 1-byte gap → reaches 6 bytes back
CASE7_ORDER = 7  # ctx_len = 6 over lowercase byte stream

# Match model
MATCH_HISTORY_BYTES = 2048  # last 2 KB of stream
MATCH_MIN_LEN = 4  # need at least 4-byte suffix match
MATCH_MAX_LEN = 16  # cap suffix length checked

# Mixer
MIXER_HIDDEN = 32
MIXER_TRAIN_STEPS = 1500
MIXER_BATCH = 4096
MIXER_LR = 3e-3
MIXER_HELDOUT_BYTES = 2_000_000
MIXER_SAMPLE_POSITIONS = 80_000  # smaller — 14-dim features, 4 outputs

# Number of mixed model families (KN_base, skip1, case7, match).
N_FAMILIES = 4


# ===========================================================================
# Part 1 — GPU n-gram table builder (lifted from gpu_ngram_w31_k11).
# ===========================================================================


def _pack_window_chunk(
    arr_int64: Tensor, start: int, end: int, k: int,
) -> tuple[Tensor, Tensor]:
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
    del order_lo, order_hi
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
    train_bytes_u8: Tensor, k: int, chunk_bytes: int = 32 * 1024 * 1024,
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


# ===========================================================================
# Part 2 — Transfer to CPU-side W3 layout (for predict-time lookup).
# ===========================================================================


def _gpu_table_to_w3_layout(
    hi: Tensor, lo: Tensor, counts: Tensor, k: int,
) -> dict:
    """Convert (hi, lo, counts) at order k into W3 CPU layout."""
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


def build_kn_tables(
    train_bytes_u8: Tensor, max_order: int = MAX_ORDER,
) -> tuple[list, np.ndarray]:
    """Build chained-backoff KN tables on GPU, transferred to CPU."""
    device = train_bytes_u8.device
    t_total = time.monotonic()
    print(f"[paq8] KN build max_order={max_order} D={KN_DISCOUNT}", flush=True)
    t0 = time.monotonic()
    hi, lo, counts = _build_top_order_gpu(train_bytes_u8, max_order)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[paq8] KN top order={max_order} pairs={hi.numel():,} {time.monotonic()-t0:.1f}s",
          flush=True)
    order_tables: list = [None] * max_order
    t0 = time.monotonic()
    order_tables[max_order - 1] = _gpu_table_to_w3_layout(hi, lo, counts, max_order)
    print(f"[paq8] KN ctx_len={max_order-1} "
          f"ctxs={order_tables[max_order-1]['ctx_keys'].shape[0]:,} "
          f"{time.monotonic()-t0:.1f}s", flush=True)
    bigram_next = None
    for new_k in range(max_order - 1, 0, -1):
        t0 = time.monotonic()
        hi, lo, counts = _step_down_gpu(hi, lo, counts, new_k + 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        order_tables[new_k - 1] = _gpu_table_to_w3_layout(hi, lo, counts, new_k)
        tbl = order_tables[new_k - 1]
        print(f"[paq8] KN ctx_len={new_k-1} ctxs={tbl['ctx_keys'].shape[0]:,} "
              f"{time.monotonic()-t0:.1f}s", flush=True)
        if new_k == 2:
            bigram_next = tbl["next_bytes"].copy()
    if bigram_next is not None:
        continuation = _build_continuation_base(bigram_next)
    else:
        continuation = np.full(256, 1.0 / 256.0, dtype=np.float64)
    print(f"[paq8] KN build done {time.monotonic()-t_total:.1f}s", flush=True)
    return order_tables, continuation


def kn_distribution(
    order_tables: list, continuation: np.ndarray,
    history: bytes, max_ctx_len: int, discount: float = KN_DISCOUNT,
) -> np.ndarray:
    """KN-interpolated next-byte distribution."""
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


def kn_features_at(
    order_tables: list, history: bytes, max_ctx_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (depth, log_total): the longest order that has a hit, and
    log of the matching context's total count. Cheap diagnostic used as
    mixer features for the KN base."""
    depth = 0
    log_total = 0.0
    hist_len = len(history)
    max_k = min(max_ctx_len, hist_len)
    for k in range(1, max_k + 1):
        tbl = order_tables[k]
        if tbl is None:
            continue
        ctx_view = tbl["ctx_view"]
        if ctx_view is None or ctx_view.shape[0] == 0:
            continue
        tail = bytes(history[-k:])
        q = np.frombuffer(tail, dtype=np.uint8).view(np.dtype((np.void, k)))[0]
        idx = int(np.searchsorted(ctx_view, q))
        if idx >= ctx_view.shape[0] or ctx_view[idx] != q:
            continue
        depth = k
        log_total = float(np.log(float(tbl["total_count_per_ctx"][idx]) + 1.0))
    return np.float32(depth), np.float32(log_total)


# ===========================================================================
# Part 3 — Skip-1 sparse contexts (independent WB table).
#
# For context window [b[-6], b[-5], b[-4], b[-3], b[-2]] -> next byte b[0],
# i.e. skip one byte (position -1 omitted). Built on GPU by indexing into
# the byte stream with a custom offset pattern.
# ===========================================================================


def _build_skip1_gpu(
    train_bytes_u8: Tensor, ctx_len: int = SKIP1_LEN,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build (hi, lo, counts) for skip-1 sparse contexts.

    Context window positions (relative to next-byte at 0):
      [-(ctx_len+1), -ctx_len, ..., -2]   (skip -1, predict 0)

    We build the equivalent of a (ctx_len + 1)-byte key where the last
    byte is the target. Concretely: at position i with i >= ctx_len+1,
    the key is bytes[i - ctx_len - 1 : i - 1] + bytes[i].
    """
    device = train_bytes_u8.device
    n = train_bytes_u8.numel()
    k = ctx_len + 1  # total key bytes including target
    # We need positions i in [ctx_len+1, n-1].
    # Build using torch.stack along context positions.
    if n < ctx_len + 2:
        empty_i = torch.zeros(0, dtype=torch.int64, device=device)
        empty_f = torch.zeros(0, dtype=torch.float32, device=device)
        return empty_i, empty_i.clone(), empty_f

    arr_i64 = train_bytes_u8.to(torch.int64)
    # m valid positions
    m = n - (ctx_len + 1)  # i ranges over [ctx_len+1, n-1], len = n - ctx_len - 1 = m
    # Context bytes at relative offsets [-(ctx_len+1), ..., -2]:
    # For i, the context is arr[i - ctx_len - 1 + j] for j in 0..ctx_len-1.
    # The j-th context byte for sample at output index s is arr[s + j] where
    # s = i - ctx_len - 1 ranges over [0, m - 1].
    # Then target byte is arr[s + ctx_len + 1] = arr[i].
    # All slices have length m.
    # k <= 8 case (k = ctx_len + 1; ctx_len = 5 → k = 6 ≤ 8 → pack in lo).
    if k > 8:
        raise NotImplementedError("skip1 only implemented for k <= 8")
    lo = torch.zeros(m, dtype=torch.int64, device=device)
    # Context bytes
    for j in range(ctx_len):
        lo = (lo << 8) | arr_i64[j:j + m]
    # Target byte at position s + ctx_len + 1 (skip s + ctx_len which is position -1)
    lo = (lo << 8) | arr_i64[ctx_len + 1: ctx_len + 1 + m]
    hi = torch.zeros(m, dtype=torch.int64, device=device)
    counts = torch.ones(m, dtype=torch.float32, device=device)
    return _sort_and_dedupe(hi, lo, counts)


def build_skip1_table(
    train_bytes_u8: Tensor, ctx_len: int = SKIP1_LEN,
) -> dict:
    """Build skip-1 table in W3-like layout with WB smoothing."""
    print(f"[paq8] skip1 build ctx_len={ctx_len}", flush=True)
    t0 = time.monotonic()
    hi, lo, counts = _build_skip1_gpu(train_bytes_u8, ctx_len)
    if train_bytes_u8.device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[paq8] skip1 unique={hi.numel():,} {time.monotonic()-t0:.1f}s",
          flush=True)
    # k bytes = ctx_len + 1 (last byte is target)
    return _gpu_table_to_w3_layout(hi, lo, counts, ctx_len + 1)


def skip1_distribution(
    table: dict, history: bytes,
) -> tuple[np.ndarray, float, float]:
    """Witten-Bell smoothed distribution from skip-1 table.

    Context: history[-(ctx_len+1) : -1] (skip last byte).
    """
    ctx_len = table["ctx_len"]
    ctx_view = table["ctx_view"]
    if ctx_view is None or ctx_view.shape[0] == 0:
        return np.full(256, 1.0 / 256.0, dtype=np.float64), 0.0, 0.0
    needed = ctx_len + 1
    if len(history) < needed:
        return np.full(256, 1.0 / 256.0, dtype=np.float64), 0.0, 0.0
    tail = bytes(history[-needed:-1])  # length ctx_len, skip last
    q = np.frombuffer(tail, dtype=np.uint8).view(np.dtype((np.void, ctx_len)))[0]
    idx = int(np.searchsorted(ctx_view, q))
    if idx >= ctx_view.shape[0] or ctx_view[idx] != q:
        return np.full(256, 1.0 / 256.0, dtype=np.float64), 0.0, 0.0
    lo = int(table["ctx_offsets"][idx])
    hi = int(table["ctx_offsets"][idx + 1])
    nb = table["next_bytes"][lo:hi]
    cn = table["counts"][lo:hi].astype(np.float64)
    total = float(table["total_count_per_ctx"][idx])
    if total <= 0.0:
        return np.full(256, 1.0 / 256.0, dtype=np.float64), 0.0, 0.0
    # WB smoothing: discount mass = 1 / (total + 1), spread uniformly.
    out = np.zeros(256, dtype=np.float64)
    denom = total + 1.0
    out[nb] = cn / denom
    unseen = 1.0 / denom
    # Distribute unseen mass uniformly over zero entries.
    zero_mask = out == 0.0
    n_zero = int(zero_mask.sum())
    if n_zero > 0:
        out[zero_mask] = unseen / n_zero
    s = out.sum()
    if s > 0:
        out /= s
    return out, 1.0, float(np.log(total + 1.0))


# ===========================================================================
# Part 4 — Case-folded byte order-7 (independent WB table).
# ===========================================================================


def build_case_folded_table(
    train_bytes_u8: Tensor, order: int = CASE7_ORDER,
) -> dict:
    """Build order-N byte table where the byte stream is first lowercased
    (ASCII A-Z → a-z; other bytes unchanged)."""
    device = train_bytes_u8.device
    print(f"[paq8] case-folded order={order}", flush=True)
    t0 = time.monotonic()
    # Lowercase: bytes 65..90 → 97..122. Only do this on ASCII.
    is_upper = (train_bytes_u8 >= 65) & (train_bytes_u8 <= 90)
    lower_bytes = train_bytes_u8.clone()
    lower_bytes[is_upper] = lower_bytes[is_upper] + 32
    hi, lo, counts = _build_top_order_gpu(lower_bytes, order)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[paq8] case-folded unique={hi.numel():,} {time.monotonic()-t0:.1f}s",
          flush=True)
    return _gpu_table_to_w3_layout(hi, lo, counts, order)


def case_folded_distribution(
    table: dict, history: bytes,
) -> tuple[np.ndarray, float, float]:
    """WB smoothed distribution; ctx is case-folded last (order-1) bytes."""
    ctx_len = table["ctx_len"]
    ctx_view = table["ctx_view"]
    if ctx_view is None or ctx_view.shape[0] == 0:
        return np.full(256, 1.0 / 256.0, dtype=np.float64), 0.0, 0.0
    if len(history) < ctx_len:
        return np.full(256, 1.0 / 256.0, dtype=np.float64), 0.0, 0.0
    # Lowercase the tail (ASCII A-Z).
    tail = bytearray(history[-ctx_len:])
    for i in range(len(tail)):
        if 65 <= tail[i] <= 90:
            tail[i] += 32
    q = np.frombuffer(bytes(tail), dtype=np.uint8).view(
        np.dtype((np.void, ctx_len)),
    )[0]
    idx = int(np.searchsorted(ctx_view, q))
    if idx >= ctx_view.shape[0] or ctx_view[idx] != q:
        return np.full(256, 1.0 / 256.0, dtype=np.float64), 0.0, 0.0
    lo = int(table["ctx_offsets"][idx])
    hi = int(table["ctx_offsets"][idx + 1])
    nb = table["next_bytes"][lo:hi]
    cn = table["counts"][lo:hi].astype(np.float64)
    total = float(table["total_count_per_ctx"][idx])
    if total <= 0.0:
        return np.full(256, 1.0 / 256.0, dtype=np.float64), 0.0, 0.0
    out = np.zeros(256, dtype=np.float64)
    denom = total + 1.0
    out[nb] = cn / denom
    unseen = 1.0 / denom
    zero_mask = out == 0.0
    n_zero = int(zero_mask.sum())
    if n_zero > 0:
        out[zero_mask] = unseen / n_zero
    s = out.sum()
    if s > 0:
        out /= s
    return out, 1.0, float(np.log(total + 1.0))


# ===========================================================================
# Part 5 — Match model (online, predict-time).
#
# Maintain a sliding window of the last MATCH_HISTORY_BYTES bytes.
# At predict time, find the longest exact suffix of the window's tail
# (of length >= MATCH_MIN_LEN, <= MATCH_MAX_LEN) that appears earlier in
# the window, and predict the byte that followed it there.
#
# Implementation: brute-force search via numpy. For each match length
# from MATCH_MAX_LEN down to MATCH_MIN_LEN, look for the last occurrence
# of the suffix in the window (excluding the most recent suffix itself).
# Use numpy.frombuffer + a small loop.
# ===========================================================================


def match_model_predict(
    history_buf: bytes,
) -> tuple[np.ndarray, float, float]:
    """Return (256-vec distribution, found_flag, log_match_len).

    If no match >= MATCH_MIN_LEN found, returns uniform with found=0.
    """
    n = len(history_buf)
    if n < MATCH_MIN_LEN + 1:
        return np.full(256, 1.0 / 256.0, dtype=np.float64), 0.0, 0.0
    arr = np.frombuffer(history_buf, dtype=np.uint8)
    # Search for the longest suffix-match in the window.
    # Limit search to last MATCH_HISTORY_BYTES.
    max_suffix = min(MATCH_MAX_LEN, n - 1)
    # We need a suffix of length L that matches somewhere in arr[:-L]
    # (the matched window's last index < n - L). Then the predicted byte
    # is arr[match_idx + L].
    best_len = 0
    pred_byte = -1
    # Start from longest; first hit wins (longest match wins).
    for L in range(max_suffix, MATCH_MIN_LEN - 1, -1):
        if L >= n:
            continue
        suffix = arr[n - L:n]  # length L
        # Search in arr[: n - L]; need match_idx + L < n.
        # Implement via tobytes find() which is fast C-level.
        search_region = arr[: n - L].tobytes()
        suf_b = suffix.tobytes()
        pos = search_region.rfind(suf_b)
        if pos >= 0 and pos + L < n:
            best_len = L
            pred_byte = int(arr[pos + L])
            break
    if best_len == 0 or pred_byte < 0:
        return np.full(256, 1.0 / 256.0, dtype=np.float64), 0.0, 0.0
    # Build a peaked distribution: high mass on predicted byte, small
    # smoothing over other bytes. Mass concentration scales with match
    # length: longer match → more confident.
    confidence = min(0.95, 0.5 + 0.05 * (best_len - MATCH_MIN_LEN))
    out = np.full(256, (1.0 - confidence) / 255.0, dtype=np.float64)
    out[pred_byte] = confidence
    s = out.sum()
    if s > 0:
        out /= s
    return out, 1.0, float(np.log(best_len + 1.0))


# ===========================================================================
# Part 6 — Tiny mixer (numpy at predict-time).
# ===========================================================================


class TinyMixer:
    """in_dim → hidden → K_families softmax."""

    def __init__(self, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray):
        self.W1 = W1.astype(np.float32)
        self.b1 = b1.astype(np.float32)
        self.W2 = W2.astype(np.float32)
        self.b2 = b2.astype(np.float32)

    def forward_softmax(self, feat: np.ndarray) -> np.ndarray:
        if feat.ndim == 1:
            h = np.tanh(feat @ self.W1 + self.b1)
            z = h @ self.W2 + self.b2
            z -= z.max()
            e = np.exp(z)
            return (e / e.sum()).astype(np.float32)
        h = np.tanh(feat @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


# ===========================================================================
# Part 7 — CharModel.
# ===========================================================================


class PAQ8GPUCharModel(CharModel):
    def __init__(
        self,
        kn_tables: list,
        continuation: np.ndarray,
        skip1_table: dict,
        case7_table: dict,
        mixer: TinyMixer,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        max_kn_ctx_len: int,
    ):
        self._kn_tables = kn_tables
        self._cont = continuation
        self._skip1 = skip1_table
        self._case7 = case7_table
        self._mixer = mixer
        self._fm = feat_mean.astype(np.float32)
        self._fs = np.where(feat_std < 1e-6, 1.0, feat_std).astype(np.float32)
        self._max_kn = max_kn_ctx_len
        # History: keep enough for KN, case7, skip1, and match model.
        self._hist_keep = max(max_kn_ctx_len, CASE7_ORDER, SKIP1_LEN + 2,
                              MATCH_HISTORY_BYTES)
        self._history = bytearray()

    def reset(self) -> None:
        self._history.clear()

    def predict(self) -> dict[str, float]:
        p = self._mixed_dist()
        best = int(p.argmax())
        return {chr(best): 1.0}

    def observe(self, char: str) -> None:
        self._history.extend(char.encode("utf-8"))
        if len(self._history) > self._hist_keep:
            del self._history[:-self._hist_keep]

    def _mixed_dist(self) -> np.ndarray:
        hist = bytes(self._history)
        # KN base
        p_kn = kn_distribution(
            self._kn_tables, self._cont, hist, self._max_kn, KN_DISCOUNT,
        )
        depth_kn, log_total_kn = kn_features_at(
            self._kn_tables, hist, self._max_kn,
        )
        # Skip-1
        p_skip1, found_skip1, log_total_skip1 = skip1_distribution(
            self._skip1, hist,
        )
        # Case-folded
        p_case7, found_case7, log_total_case7 = case_folded_distribution(
            self._case7, hist,
        )
        # Match model (uses up to MATCH_HISTORY_BYTES tail)
        match_hist = hist[-MATCH_HISTORY_BYTES:]
        p_match, found_match, log_match_len = match_model_predict(match_hist)

        # Per-family entropies (cheap)
        def ent(p):
            return float(-(p * np.log(np.clip(p, 1e-30, 1.0))).sum())
        ent_kn = ent(p_kn)
        ent_skip1 = ent(p_skip1)
        ent_case7 = ent(p_case7)
        ent_match = ent(p_match)

        # Feature vector: per-family [found, log_total, entropy] + bias.
        feat = np.array([
            1.0, log_total_kn, ent_kn, depth_kn,
            found_skip1, log_total_skip1, ent_skip1,
            found_case7, log_total_case7, ent_case7,
            found_match, log_match_len, ent_match,
            1.0,
        ], dtype=np.float32)
        feat = (feat - self._fm) / self._fs
        weights = self._mixer.forward_softmax(feat)  # (4,)
        # Mixed.
        p = (weights[0] * p_kn
             + weights[1] * p_skip1
             + weights[2] * p_case7
             + weights[3] * p_match)
        s = p.sum()
        if s > 1e-30:
            p = p / s
        return p


# ===========================================================================
# Part 8 — Mixer training (GPU).
# ===========================================================================


def _collect_mixer_training_data(
    kn_tables: list,
    cont: np.ndarray,
    skip1_tbl: dict,
    case7_tbl: dict,
    heldout: bytes,
    n_positions: int,
    max_kn_ctx_len: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample positions in heldout; for each, compute per-family
    log-probs + features + target. Returns:
        feats: (N, in_dim)
        per_fam_logp_at_target: (N, 4) — log p_k(target) for each family
        targets: (N,)
    """
    rng = np.random.default_rng(seed)
    n = len(heldout)
    # Need enough history for all context families; match model can use
    # shorter history if we're below MATCH_HISTORY_BYTES.
    min_hist = max(max_kn_ctx_len, CASE7_ORDER, SKIP1_LEN + 2, MATCH_MIN_LEN + 1) + 1
    lo_idx = min_hist
    hi_idx = n - 1
    if hi_idx <= lo_idx:
        raise ValueError(
            f"heldout too small for mixer training "
            f"(n={n}, min_required={lo_idx + 2})"
        )
    n_positions = min(n_positions, hi_idx - lo_idx)
    pos = rng.integers(lo_idx, hi_idx, size=n_positions)

    in_dim = 14  # see feature vector below
    feats = np.zeros((n_positions, in_dim), dtype=np.float32)
    logp_at_target = np.zeros((n_positions, 4), dtype=np.float32)
    targets = np.zeros(n_positions, dtype=np.int64)

    # Per-position loop. Hot path; needs to be fast.
    # 200,000 positions × 4 families. Each KN call walks up to 11 orders.
    # Cost: ~3-5s in numpy; acceptable.
    arr_full = np.frombuffer(heldout, dtype=np.uint8)
    t0 = time.monotonic()
    for i, p_idx in enumerate(pos):
        target = int(arr_full[p_idx])
        targets[i] = target
        # History: bytes up to (not including) p_idx.
        # Slice to last 8192 for the match model; KN/skip1/case7 only
        # need shorter.
        hist_start = max(0, p_idx - MATCH_HISTORY_BYTES)
        hist = heldout[hist_start:p_idx]
        # KN
        p_kn = kn_distribution(kn_tables, cont, hist, max_kn_ctx_len, KN_DISCOUNT)
        depth_kn, log_total_kn = kn_features_at(kn_tables, hist, max_kn_ctx_len)
        # Skip-1
        p_skip1, found_skip1, log_total_skip1 = skip1_distribution(skip1_tbl, hist)
        # Case7
        p_case7, found_case7, log_total_case7 = case_folded_distribution(case7_tbl, hist)
        # Match
        p_match, found_match, log_match_len = match_model_predict(
            hist[-MATCH_HISTORY_BYTES:]
        )
        def ent(p):
            return float(-(p * np.log(np.clip(p, 1e-30, 1.0))).sum())
        ent_kn = ent(p_kn)
        ent_skip1 = ent(p_skip1)
        ent_case7 = ent(p_case7)
        ent_match = ent(p_match)
        feats[i] = [
            1.0, log_total_kn, ent_kn, depth_kn,
            found_skip1, log_total_skip1, ent_skip1,
            found_case7, log_total_case7, ent_case7,
            found_match, log_match_len, ent_match,
            1.0,
        ]
        logp_at_target[i, 0] = float(np.log(max(p_kn[target], 1e-30)))
        logp_at_target[i, 1] = float(np.log(max(p_skip1[target], 1e-30)))
        logp_at_target[i, 2] = float(np.log(max(p_case7[target], 1e-30)))
        logp_at_target[i, 3] = float(np.log(max(p_match[target], 1e-30)))
        if (i + 1) % 50_000 == 0:
            print(f"[paq8] mixer-data {i+1:,}/{n_positions:,} "
                  f"({(i+1)/(time.monotonic()-t0):.0f} pos/s)", flush=True)
    return feats, logp_at_target, targets


def _train_mixer_gpu(
    feats: np.ndarray, logp_at_target: np.ndarray, targets: np.ndarray,
    n_steps: int = MIXER_TRAIN_STEPS,
    batch: int = MIXER_BATCH,
    lr: float = MIXER_LR,
    hidden: int = MIXER_HIDDEN,
    device: torch.device = torch.device("cuda"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the mixer with log-sum-exp loss against true target log-prob."""
    in_dim = feats.shape[1]
    K = logp_at_target.shape[1]
    N = feats.shape[0]
    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    feats_t = torch.from_numpy(feats).to(device)
    logp_t = torch.from_numpy(logp_at_target).to(device)

    fm = feats_t.mean(dim=0)
    fs = feats_t.std(dim=0).clamp(min=1e-6)
    feats_norm = (feats_t - fm) / fs

    W1 = torch.zeros(in_dim, hidden, device=device, requires_grad=True)
    b1 = torch.zeros(hidden, device=device, requires_grad=True)
    W2 = torch.zeros(hidden, K, device=device, requires_grad=True)
    b2 = torch.zeros(K, device=device, requires_grad=True)
    with torch.no_grad():
        W1.normal_(mean=0.0, std=0.1, generator=rng)
        W2.normal_(mean=0.0, std=0.1, generator=rng)

    opt = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
    t0 = time.monotonic()
    last_loss = None
    log_every = max(100, n_steps // 8)
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch,), device=device, generator=rng)
        x = feats_norm[idx]
        yp = logp_t[idx]  # (B, K) — log p_k(target) for each family
        h = torch.tanh(x @ W1 + b1)
        z = h @ W2 + b2
        log_w = torch.log_softmax(z, dim=-1)  # (B, K)
        # Mixed log-prob: logsumexp(log_w + log_p_target)
        mixed = torch.logsumexp(log_w + yp, dim=-1)  # (B,)
        loss = -mixed.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu().item())
        if step % log_every == 0 or step == n_steps - 1:
            print(f"[paq8] mixer step={step:4d} loss={last_loss:.4f}", flush=True)
    return (
        W1.detach().cpu().numpy(),
        b1.detach().cpu().numpy(),
        W2.detach().cpu().numpy(),
        b2.detach().cpu().numpy(),
        fm.cpu().numpy(),
        fs.cpu().numpy(),
    )


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
        print(f"[paq8] SEED={seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = train_text.encode("utf-8")
    is_smoke = len(raw) < SMOKE_TRAIN_BYTES

    if is_smoke:
        max_order = max(3, min(MAX_ORDER, len(raw) // 64))
        n_steps = 30
        n_sample = min(500, max(50, len(raw) // 8))
        case7_order = min(CASE7_ORDER, max_order)
        skip1_len = min(SKIP1_LEN, max_order - 1)
        heldout_bytes = max(200, len(raw) // 5)
        print(f"[paq8] SMOKE mode train={len(raw)}B max_order={max_order}")
    else:
        max_order = MAX_ORDER
        n_steps = MIXER_TRAIN_STEPS
        n_sample = MIXER_SAMPLE_POSITIONS
        case7_order = CASE7_ORDER
        skip1_len = SKIP1_LEN
        heldout_bytes = MIXER_HELDOUT_BYTES

    print(f"[paq8] device={device} max_order={max_order} "
          f"skip1_len={skip1_len} case7_order={case7_order}", flush=True)

    t_total = time.monotonic()
    if heldout_bytes > 0 and len(raw) - heldout_bytes >= 1024:
        table_bytes = raw[:-heldout_bytes]
        heldout = raw[-heldout_bytes:]
    else:
        table_bytes = raw
        heldout = raw[-max(200, len(raw) // 5):]

    train_bytes_u8 = torch.frombuffer(bytearray(table_bytes), dtype=torch.uint8).to(device)
    print(f"[paq8] encoded {train_bytes_u8.numel():,} train bytes "
          f"({time.monotonic()-t_total:.1f}s); heldout={len(heldout):,} bytes",
          flush=True)

    # Build KN tables.
    kn_tables, continuation = build_kn_tables(train_bytes_u8, max_order=max_order)

    # Build skip-1 table.
    skip1_tbl = build_skip1_table(train_bytes_u8, ctx_len=skip1_len)

    # Build case-folded order-7 table.
    case7_tbl = build_case_folded_table(train_bytes_u8, order=case7_order)

    # Free GPU memory.
    del train_bytes_u8
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t_tables = time.monotonic() - t_total
    print(f"[paq8] all tables built in {t_tables:.1f}s", flush=True)

    # Collect mixer training data on heldout.
    t0 = time.monotonic()
    feats, logp_t, targets = _collect_mixer_training_data(
        kn_tables, continuation, skip1_tbl, case7_tbl,
        heldout, n_sample, max_kn_ctx_len=max_order - 1, seed=42,
    )
    print(f"[paq8] mixer data collected {feats.shape[0]:,} samples "
          f"feat_dim={feats.shape[1]} ({time.monotonic()-t0:.1f}s)", flush=True)

    # Train mixer.
    t0 = time.monotonic()
    W1, b1, W2, b2, fm, fs = _train_mixer_gpu(
        feats, logp_t, targets,
        n_steps=n_steps, batch=min(MIXER_BATCH, feats.shape[0]),
        lr=MIXER_LR, hidden=MIXER_HIDDEN, device=device,
    )
    print(f"[paq8] mixer fit done {time.monotonic()-t0:.1f}s", flush=True)

    mixer = TinyMixer(W1, b1, W2, b2)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[paq8] total build: {time.monotonic()-t_total:.1f}s", flush=True)

    return PAQ8GPUCharModel(
        kn_tables=kn_tables,
        continuation=continuation,
        skip1_table=skip1_tbl,
        case7_table=case7_tbl,
        mixer=mixer,
        feat_mean=fm,
        feat_std=fs,
        max_kn_ctx_len=max_order - 1,
    )
