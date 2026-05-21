"""chunker_phase1_v2 — Schmidhuber chunker Phase 1, run 2 (tune NN capacity).

Same architecture as v1; only knob change: upper-tier H capacity is
increased to match alpha_06's NN (d=256, L=4, 1200 steps). Run 2 of the
adaptive-3-run budget — tests whether v1's d=192 NN was the limiter.

Original v1 docstring follows.

chunker_phase1_v1 — Schmidhuber chunker Phase 1 (1991/1993).

Architecture:
- Lower tier L: GPU KN n-gram (W31-style, order-12). Provides the surprise
  signal p_L(true_byte | context). Cheap, no GPU forward at inference time
  per byte (single searchsorted on prebuilt tables).
- Upper tier H: 4-layer d=256 modded-nanogpt transformer. Trained ONLY on
  surprise positions (positions where p_L(true_byte) < tau). Sees full
  context but loss is masked to surprise positions only.
- Output combiner: at predict(), always blend NN + KN via
  p_final = alpha * p_nn + (1-alpha) * p_kn  with alpha=0.5.

This is the spec_16_chunker.md Phase 1 architecture, with two deviations
from a literal Schmidhuber chunker for practical reasons:
1. L = n-gram, not a transformer. The D1 diagnostic used a 2L/d=128
   transformer for L; here we use the KN tables we'd already build for the
   hybrid baseline. Same surprise-signal role.
2. H runs on every predict() rather than just at surprise positions. The
   KV-cache state continuity over surprise-only positions is delicate; we
   instead train H to specialize on surprise positions via masked loss
   and blend uniformly at inference. This is the cleanest mechanistic
   isolation of "H gets training signal only from hard bytes."

Why this could beat alpha_06 (14kJ / 0.7437):
- Standard hybrid (alpha_06) trains NN on ALL bytes uniformly. NN burns
  capacity on easy bytes (~73% of corpus) that KN already solves.
- Chunker: dedicates NN capacity to hard bytes (~27% of corpus). NN learns
  the harder conditional distribution. KN handles easy bytes.

Run 1 hyperparameters (best-guess literature config):
- tau = 0.1 (D1 PASS threshold; p_s(0.1)=0.267)
- H: d=192, L=4, 800 Muon steps, max_len=512
- alpha = 0.5 (NN and KN equal at inference; NN slightly less because
  it's trained on a hard subset and may be noisier on easy bytes).

Adaptive 3-run budget per the iterative-research skill rule.
"""
from __future__ import annotations

__author__ = "@explore-chunker-2026-05-19"

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

# KN n-gram (lower tier L).
MAX_ORDER = 12
MAX_CTX_LEN = MAX_ORDER - 1
KN_DISCOUNT = 0.5

# Surprise threshold. v1 measured p_s(0.30) = 0.4351 on real WikiText
# — higher than D1's 0.267 target. v2 lowers tau to 0.15 to get a
# smaller, harder subset (~25% of positions).
TAU = 0.15

# Upper tier H (NN). v2: match alpha_06 (d=256/L=4/1200 steps) — v1's NN
# was undertrained (loss 2.25 vs alpha_06's likely 1.5). Same capacity
# AND more steps should let H learn the hard-byte distribution.
H_MODEL_DIM = 256
H_NUM_LAYERS = 4
H_HEAD_DIM = 64
H_MAX_LEN = 1024
H_BATCH_SIZE = 32
H_N_STEPS = 1200

# Inference mix. v2: match alpha_06's α=0.60 to test if surprise-trained
# NN at the same mix as full-trained NN performs better or worse.
ALPHA = 0.60

SMOKE_TRAIN_BYTES = 10_000

# Sign-bit constant for unsigned-lex sort via XOR. 1<<63 overflows int64
# literal; -(1<<63) = INT64_MIN is the same bit pattern in two's complement.
SIGN_BIT_AS_INT64 = -(1 << 63)


# ===========================================================================
# Part 1 — GPU KN build (W31-style, lifted from alpha_06/submission.py).
# ===========================================================================


def _pack_window_chunk(
    arr_int64: Tensor,
    start: int,
    end: int,
    k: int,
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
    # XOR-bit fix for sign-bit aliasing (per gpu_ngram_o14_xorfix).
    sign_bit = torch.tensor(SIGN_BIT_AS_INT64, dtype=torch.int64, device=device)
    sort_lo = lo.bitwise_xor(sign_bit)
    sort_hi = hi.bitwise_xor(sign_bit)
    order_lo = torch.argsort(sort_lo, stable=True)
    sort_hi = sort_hi[order_lo]
    hi = hi[order_lo]
    lo = lo[order_lo]
    counts = counts[order_lo]
    order_hi = torch.argsort(sort_hi, stable=True)
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
    device = train_bytes_u8.device
    t_total = time.monotonic()
    print(f"[chunker] starting GPU KN build; max_order={max_order} "
          f"D={KN_DISCOUNT}", flush=True)
    t0 = time.monotonic()
    hi, lo, counts = _build_top_order_gpu(train_bytes_u8, max_order)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[chunker] top order={max_order} unique pairs: {hi.numel():,}  "
          f"{time.monotonic()-t0:.1f}s", flush=True)
    order_tables: list = [None] * max_order
    t0 = time.monotonic()
    order_tables[max_order - 1] = _gpu_table_to_w3_layout(hi, lo, counts, max_order)
    print(f"[chunker] ctx_len={max_order-1} "
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
        print(f"[chunker] ctx_len={new_k-1} ctxs={tbl['ctx_keys'].shape[0]:,} "
              f"{time.monotonic()-t0:.1f}s", flush=True)
        if new_k == 2:
            bigram_next_for_base = tbl["next_bytes"].copy()
    if bigram_next_for_base is not None:
        continuation = _build_continuation_base(bigram_next_for_base)
    else:
        continuation = np.full(256, 1.0 / 256.0, dtype=np.float64)
    print(f"[chunker] KN build done: {time.monotonic()-t_total:.1f}s",
          flush=True)
    return order_tables, continuation


def kn_distribution(
    order_tables: list, continuation: np.ndarray,
    history: bytes, max_ctx_len: int, discount: float = KN_DISCOUNT,
) -> np.ndarray:
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
# Part 2 — Surprise-mask precomputation on GPU.
# ===========================================================================
# Goal: for each training position i, compute p_kn(byte_i | byte_{i-11:i})
# and decide if it's a surprise (p_kn(true) < tau).
#
# Doing the full KN recursion per-position would be slow. Approximation:
# use raw MLE of the LONGEST matching context (i.e., the W31 "longest hit"
# path), which is the dominant KN term for non-sparse contexts.
#
# Concretely: hash the last-k bytes for each k=1..max_ctx_len, look up in
# the n-gram count table for that k. If the (ctx, next_byte) pair exists,
# its MLE prob is count(ctx, next_byte) / count(ctx). Take the longest k
# that has a positive count; use that as p_L(true).
#
# This is exactly the path the dynamic-patching literature (BLT, SpaceByte)
# uses for boundary detection — fast, vectorized, "use what the highest-
# order n-gram says."

def _build_surprise_mask_gpu(
    train_bytes_u8: Tensor,
    order_tables: list,
    tau: float,
    max_order: int = MAX_ORDER,
    chunk_size: int = 16_000_000,
    surprise_orders: tuple = (4,),
    min_ctx_count: int = 8,
) -> Tensor:
    """Compute a boolean tensor S[i] indicating whether position i is a
    surprise (p_L(true_byte_i | last k bytes) < tau).

    Heuristic: p_L(byte_i) ~= MLE_k(byte_i | ctx_{i-k..i}) where k is in
    `surprise_orders`. We use order-4 alone because:
    - Order-4 contexts are densely observed on WikiText (~256k+ unique).
    - MLE on order-4 is reliable (most contexts have count >> 1).
    - Higher orders (e.g., 7) often have count=1 contexts → MLE=1.0 (the
      lookup APPEARS confident but is actually sparse/unreliable).

    Positions where the order-4 context lookup misses (rare) fall back to
    p_true=0 → automatic surprise (treats them as "hard" — they will
    rely on H to predict).
    """
    device = train_bytes_u8.device
    n = train_bytes_u8.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)
    # We'll fill p_true[i] = best MLE estimate of P(byte_i | context),
    # tracking the longest order that hit.
    p_true_gpu = torch.zeros(n, dtype=torch.float32, device=device)
    hit_order = torch.zeros(n, dtype=torch.int8, device=device)

    arr_int64 = train_bytes_u8.to(torch.int64)

    # Precompute order-by-order for orders in `surprise_orders`.
    for k_ctx in surprise_orders:
        if k_ctx >= max_order:
            continue
        # k_ctx = context length; need byte at position i, conditioned on
        # bytes [i-k_ctx, i-1]. So we look at full window of size k_ctx+1.
        K_full = k_ctx + 1
        tbl = order_tables[k_ctx]  # tables indexed by ctx_len
        if tbl is None:
            continue
        ctx_view = tbl["ctx_view"]
        if ctx_view is None or ctx_view.shape[0] == 0:
            continue

        # Bring the table CPU arrays onto the GPU once.
        # ctx_view is a numpy void-byte view; reconstruct the keys array.
        ctx_keys_np = tbl["ctx_keys"]  # (n_ctx, ctx_len) uint8
        next_bytes_np = tbl["next_bytes"]  # (n_rows,) uint8
        counts_np = tbl["counts"]  # (n_rows,) int32
        ctx_offsets_np = tbl["ctx_offsets"]  # (n_ctx+1,) int64
        total_per_ctx_np = tbl["total_count_per_ctx"]  # (n_ctx,) int64

        n_ctx = ctx_keys_np.shape[0]
        if n_ctx == 0:
            continue

        # Pack ctx_keys into int64 per-row for vectorized searchsorted.
        # k_ctx <= 11 in our setup (max_order=12), so fits in int64 only if
        # k_ctx <= 8. For k_ctx in 9..11, need two int64s.
        if k_ctx <= 8:
            # Pack ctx_keys[k_ctx columns of uint8] -> int64 lo
            ctx_keys_t = torch.from_numpy(ctx_keys_np.astype(np.int64)).to(device)
            ctx_lo_table = torch.zeros(n_ctx, dtype=torch.int64, device=device)
            for j in range(k_ctx):
                ctx_lo_table = (ctx_lo_table << 8) | ctx_keys_t[:, j]
            ctx_hi_table = torch.zeros(n_ctx, dtype=torch.int64, device=device)
        else:
            ctx_keys_t = torch.from_numpy(ctx_keys_np.astype(np.int64)).to(device)
            hi_bytes = k_ctx - 8
            ctx_hi_table = torch.zeros(n_ctx, dtype=torch.int64, device=device)
            for j in range(hi_bytes):
                ctx_hi_table = (ctx_hi_table << 8) | ctx_keys_t[:, j]
            ctx_lo_table = torch.zeros(n_ctx, dtype=torch.int64, device=device)
            for j in range(hi_bytes, k_ctx):
                ctx_lo_table = (ctx_lo_table << 8) | ctx_keys_t[:, j]
        # Apply XOR sign-bit fix to match sort order (table was sorted
        # under XOR transformation).
        sign_bit_t = torch.tensor(SIGN_BIT_AS_INT64, dtype=torch.int64, device=device)
        ctx_lo_table_xor = ctx_lo_table.bitwise_xor(sign_bit_t)
        ctx_hi_table_xor = ctx_hi_table.bitwise_xor(sign_bit_t)

        # Build a composite key as a single int128 — but torch lacks int128.
        # Instead, sort table by (hi, lo) and do hierarchical searchsorted:
        # First narrow by hi, then lo within.
        # The table was already sorted via the build's _sort_and_dedupe
        # (sort by xor'd lo, then stable sort by xor'd hi -> final order
        # is xor'd-hi major, xor'd-lo minor). So we can:
        #   - searchsorted by hi: find candidate range
        #   - within range, searchsorted by lo

        # Actually simpler: we encode the combined key as a tensor of
        # shape (n_ctx, 2): [hi_xor, lo_xor]. For searchsorted we use the
        # fact that pairs are sortable lex when we order along hi first.
        # We'll do block searchsorted: find lower/upper indices for hi
        # match, then within block do searchsorted for lo.

        # Build query keys: from train_bytes, for each position i, the
        # window [i-k_ctx..i-1] is the context, byte[i] is the target.
        # We need to query positions i = k_ctx..n-1.
        m = n - k_ctx
        if m <= 0:
            continue

        # We process in chunks to avoid OOM.
        for cstart in range(0, m, chunk_size):
            cend = min(m, cstart + chunk_size)
            # query positions [cstart .. cend)  correspond to absolute
            # positions [cstart + k_ctx .. cend + k_ctx).
            # context window: bytes[(cstart) .. (cend + k_ctx - 1)] sliding.
            # build hi/lo for each window of size k_ctx.
            # Inline the pack: we have train_bytes_u8 on GPU.
            ctx_view_start = cstart
            ctx_view_end = cend + k_ctx  # exclusive
            if k_ctx <= 8:
                q_lo = torch.zeros(cend - cstart, dtype=torch.int64, device=device)
                for j in range(k_ctx):
                    q_lo = (q_lo << 8) | arr_int64[ctx_view_start + j: ctx_view_start + j + (cend - cstart)]
                q_hi = torch.zeros(cend - cstart, dtype=torch.int64, device=device)
            else:
                q_hi = torch.zeros(cend - cstart, dtype=torch.int64, device=device)
                for j in range(hi_bytes):
                    q_hi = (q_hi << 8) | arr_int64[ctx_view_start + j: ctx_view_start + j + (cend - cstart)]
                q_lo = torch.zeros(cend - cstart, dtype=torch.int64, device=device)
                for j in range(hi_bytes, k_ctx):
                    q_lo = (q_lo << 8) | arr_int64[ctx_view_start + j: ctx_view_start + j + (cend - cstart)]
            q_lo_xor = q_lo.bitwise_xor(sign_bit_t)
            q_hi_xor = q_hi.bitwise_xor(sign_bit_t)

            # Step 1: find range of ctx_hi_table_xor == q_hi_xor.
            lo_hi = torch.searchsorted(ctx_hi_table_xor, q_hi_xor, right=False)
            hi_hi = torch.searchsorted(ctx_hi_table_xor, q_hi_xor, right=True)
            # Step 2: within [lo_hi, hi_hi), find ctx_lo_table_xor == q_lo_xor.
            # Use single global searchsorted to find candidate; then verify
            # both hi and lo match.
            # Implementation: vectorized binary-search inside per-row slice
            # is awkward; instead, do a global lo-searchsorted, then check
            # that result lies in [lo_hi, hi_hi).
            # Note: the table is hi-major, lo-minor. So lo_xor is NOT
            # globally sorted (only within an hi group). But within
            # [lo_hi, hi_hi), it IS sorted. So we can use torch.searchsorted
            # with sorter not natively... let's do per-row binary search
            # manually using the (lo_hi, hi_hi) bracket.

            # Per-row binary search: we manually iterate log2 steps,
            # narrowing [lo, hi) toward where ctx_lo_table_xor == q_lo_xor.
            lo = lo_hi.clone()
            hi = hi_hi.clone()
            # max iterations = ceil(log2(n_ctx))
            max_iter = max(1, int(np.ceil(np.log2(max(2, n_ctx)))))
            for _ in range(max_iter):
                mid = (lo + hi) // 2
                # bound mid by max index
                mid_clamped = torch.clamp(mid, 0, n_ctx - 1)
                m_val = ctx_lo_table_xor[mid_clamped]
                # Narrow: if m_val < q_lo_xor → search right; else left.
                go_right = m_val < q_lo_xor
                lo = torch.where(go_right, mid + 1, lo)
                hi = torch.where(go_right, hi, mid)
                # exit if lo >= hi for all (we just keep iterating; safe)

            # Now lo points to first index where lo_table >= q. Check
            # lo < hi_hi and ctx_lo_table_xor[lo] == q_lo_xor and
            # ctx_hi_table_xor[lo] == q_hi_xor.
            lo_clamped = torch.clamp(lo, 0, n_ctx - 1)
            in_range = (lo < hi_hi) & (lo >= lo_hi)
            lo_eq = ctx_lo_table_xor[lo_clamped] == q_lo_xor
            hi_eq = ctx_hi_table_xor[lo_clamped] == q_hi_xor
            ctx_hit = in_range & lo_eq & hi_eq  # bool, (chunk,)
            # Now we have a candidate ctx index per query (lo_clamped).
            # For matched rows, look at (ctx_offsets[lo_clamped],
            # ctx_offsets[lo_clamped+1]) range in next_bytes, find where
            # next_bytes == target.
            # target bytes = train_bytes_u8[cstart + k_ctx .. cend + k_ctx)
            target = train_bytes_u8[cstart + k_ctx: cend + k_ctx].to(torch.int64)
            # We need: for each query row in this chunk, search the slice
            # next_bytes[lo:hi] for value==target.
            # Vectorize via flattened next_bytes_t + offsets.

            # Pre-move tables to GPU once outside the chunk loop. Lift this
            # out of the loop:
            if not hasattr(_build_surprise_mask_gpu, "_cache"):
                _build_surprise_mask_gpu._cache = {}
            cache = _build_surprise_mask_gpu._cache
            cache_key = (k_ctx, id(tbl))
            if cache_key not in cache:
                next_bytes_t = torch.from_numpy(next_bytes_np.astype(np.int64)).to(device)
                counts_t = torch.from_numpy(counts_np.astype(np.int64)).to(device)
                ctx_offsets_t = torch.from_numpy(ctx_offsets_np.astype(np.int64)).to(device)
                total_per_ctx_t = torch.from_numpy(total_per_ctx_np.astype(np.int64)).to(device)
                cache[cache_key] = (next_bytes_t, counts_t, ctx_offsets_t, total_per_ctx_t)
            next_bytes_t, counts_t, ctx_offsets_t, total_per_ctx_t = cache[cache_key]

            # For each candidate row: get its (offset_start, offset_end).
            off_lo = ctx_offsets_t[lo_clamped]  # int64, shape (chunk,)
            off_hi = ctx_offsets_t[lo_clamped + 1]  # int64
            # Now find where in next_bytes[off_lo:off_hi] equals target.
            # We do this with per-row binary search (since next_bytes are
            # sorted within a ctx group due to construction order).
            #
            # Actually next_bytes within a ctx group are NOT guaranteed
            # sorted (they're whatever bytes followed that ctx in train
            # order, then dedupe groups them).
            #
            # In _sort_and_dedupe, the FULL (k+1)-byte key was sorted; the
            # ctx_len bytes were the prefix and the (k+1)-th byte (next_byte)
            # was the suffix. So after sort, within a ctx group, next_bytes
            # ARE in ascending order. So we CAN do binary search.

            # Per-row binary search for next_byte == target.
            lo2 = off_lo.clone()
            hi2 = off_hi.clone()
            n_rows = next_bytes_t.numel()
            for _ in range(max(1, int(np.ceil(np.log2(max(2, int(off_hi.max().item()) - int(off_lo.min().item()) + 1)))))):
                mid2 = (lo2 + hi2) // 2
                mid2_clamped = torch.clamp(mid2, 0, n_rows - 1)
                m2 = next_bytes_t[mid2_clamped]
                go_right2 = m2 < target
                lo2 = torch.where(go_right2, mid2 + 1, lo2)
                hi2 = torch.where(go_right2, hi2, mid2)

            lo2_clamped = torch.clamp(lo2, 0, n_rows - 1)
            in_range2 = (lo2 < off_hi) & (lo2 >= off_lo)
            byte_eq = next_bytes_t[lo2_clamped] == target
            pair_hit = ctx_hit & in_range2 & byte_eq

            # Compute MLE = count / total, only for pair_hit rows.
            pair_count = counts_t[lo2_clamped].to(torch.float32)
            ctx_total = total_per_ctx_t[lo_clamped].to(torch.float32)
            ctx_total_safe = torch.where(ctx_total > 0, ctx_total, torch.ones_like(ctx_total))
            p_mle = pair_count / ctx_total_safe  # (chunk,)
            p_mle = torch.where(pair_hit, p_mle, torch.zeros_like(p_mle))

            # Update p_true_gpu for absolute positions [cstart+k_ctx .. cend+k_ctx)
            # but only where pair_hit AND this k is higher than what's been recorded.
            abs_lo = cstart + k_ctx
            abs_hi = cend + k_ctx
            cur_order = hit_order[abs_lo:abs_hi]
            should_update = pair_hit & (cur_order < k_ctx)
            # We want to assign p_true_gpu where should_update.
            p_true_slice = p_true_gpu[abs_lo:abs_hi]
            p_true_slice_new = torch.where(should_update, p_mle, p_true_slice)
            p_true_gpu[abs_lo:abs_hi] = p_true_slice_new
            order_slice_new = torch.where(should_update,
                                          torch.full_like(cur_order, k_ctx),
                                          cur_order)
            hit_order[abs_lo:abs_hi] = order_slice_new

        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"[chunker] surprise pass k_ctx={k_ctx} done", flush=True)

    # Bytes with no hit at any order use the continuation (uniform-ish)
    # fallback; they'll be treated as "surprise" (low p_true → surprise).
    # We just compare p_true_gpu < tau.
    surprise = p_true_gpu < tau
    return surprise


# ===========================================================================
# Part 3 — Upper-tier H transformer (modded-nanogpt arch, smaller).
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


# ===========================================================================
# Part 4 — Surprise-masked NN training loop.
# ===========================================================================


def _train_h_with_surprise_mask(
    train_bytes_gpu: Tensor,
    surprise_mask: Tensor,  # (n,) bool — TRUE = surprise position
    cfg: dict,
    device: torch.device,
) -> GPT:
    """Train H model with cross-entropy MASKED to surprise positions only."""
    n = train_bytes_gpu.numel()
    max_len = cfg["max_len"]
    batch_size = cfg["batch_size"]
    n_steps = cfg["n_steps"]

    if n < max_len + 1:
        raise ValueError(f"need at least {max_len+1} bytes; got {n}")

    model = GPT(
        vocab_size=256,
        num_layers=cfg["num_layers"],
        model_dim=cfg["model_dim"],
        head_dim=cfg["head_dim"],
        max_len=max_len,
    ).to(device)
    _init_modded(model)
    block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2]
    scalars = [p for p in model.parameters() if p.ndim < 2]
    optimizer1 = AdamW(
        [
            dict(params=[model.embed.weight], lr=cfg["embed_lr"]),
            dict(params=[model.proj.weight], lr=cfg["head_lr"]),
            dict(params=scalars, lr=cfg["scalar_lr"]),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    optimizer2 = Muon(block_2d, lr=cfg["muon_lr"], weight_decay=cfg["muon_wd"])
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    n_surprise = int(surprise_mask.sum().item())
    print(f"[chunker] H model: {n_params/1e6:.2f}M params, "
          f"surprise positions: {n_surprise:,}/{n:,} "
          f"({100.0*n_surprise/n:.1f}%)", flush=True)

    def set_lr(step: int) -> None:
        progress = step / n_steps
        cooldown_frac = cfg.get("cooldown_frac", 0.7)
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

    # Mask shape: (n,) bool. surprise_mask[i] = is position i a surprise.
    # When training, target is bytes[start+1: start+max_len+1]. The mask
    # for those targets is surprise_mask[start+1: start+max_len+1].

    for step in range(n_steps):
        set_lr(step)
        idx = torch.randint(0, n - max_len - 1, (batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(max_len + 1, device=device)[None, :]
        flat = train_bytes_gpu[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]
        # mask shape: (batch, max_len). Take surprise mask for the target positions.
        target_offsets = idx[:, None] + torch.arange(1, max_len + 1, device=device)[None, :]
        target_mask = surprise_mask[target_offsets]  # (B, T) bool
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                # Per-position cross-entropy.
                logp = F.log_softmax(logits.float(), dim=-1)
                # gather log p(true_byte)
                nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
                # Apply mask: only count surprise positions.
                mask_f = target_mask.float()
                # Avoid divide-by-zero in degenerate batches.
                denom = mask_f.sum().clamp(min=1.0)
                loss = (nll * mask_f).sum() / denom
        else:
            logits, _ = model(x)
            logp = F.log_softmax(logits.float(), dim=-1)
            nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            mask_f = target_mask.float()
            denom = mask_f.sum().clamp(min=1.0)
            loss = (nll * mask_f).sum() / denom
        loss.backward()
        for opt in optimizers:
            opt.step()
        if step % 100 == 0 or step == n_steps - 1:
            elapsed = time.monotonic() - t0
            print(
                f"[chunker] H step {step:5d}/{n_steps}  "
                f"loss {loss.item():.4f}  elapsed {elapsed:.0f}s",
                flush=True,
            )
    return model


# ===========================================================================
# Part 5 — Streaming hybrid CharModel.
# ===========================================================================


class ChunkerPhase1CharModel(CharModel):
    """Schmidhuber chunker Phase 1: KN (L) + surprise-trained NN (H), blended."""

    def __init__(
        self,
        model: GPT,
        order_tables: list,
        continuation: np.ndarray,
        max_ctx_len: int = MAX_CTX_LEN,
        discount: float = KN_DISCOUNT,
        alpha: float = ALPHA,
        tau: float = TAU,
        device: torch.device | None = None,
    ):
        self.model = model
        self.order_tables = order_tables
        self.continuation = continuation
        self.max_ctx_len = max_ctx_len
        self.discount = float(discount)
        self.alpha = float(alpha)
        self.tau = float(tau)
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
        # v2: simple fixed-alpha mix (no surprise gating), to isolate
        # whether v1's below-floor result was from inference gating vs
        # training-on-subset.
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


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[chunker] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = train_text.encode("utf-8")
    is_smoke = len(raw) < SMOKE_TRAIN_BYTES

    train_bytes_u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)

    if is_smoke:
        kn_max_order = max(2, min(MAX_ORDER, len(raw) // 32))
        seq = max(8, min(64, len(raw) // 4))
        h_cfg = dict(
            model_dim=64,
            num_layers=2,
            head_dim=32,
            max_len=seq,
            batch_size=2,
            n_steps=4,
            embed_lr=0.3,
            head_lr=1.0 / 320,
            scalar_lr=0.01,
            muon_lr=0.035,
            muon_wd=0.025,
            cooldown_frac=0.7,
        )
        print(f"[chunker] SMOKE mode (train={len(raw)} bytes)  "
              f"NN steps={h_cfg['n_steps']}  kn_max_order={kn_max_order}")
    else:
        kn_max_order = MAX_ORDER
        h_cfg = dict(
            model_dim=H_MODEL_DIM,
            num_layers=H_NUM_LAYERS,
            head_dim=H_HEAD_DIM,
            max_len=H_MAX_LEN,
            batch_size=H_BATCH_SIZE,
            n_steps=H_N_STEPS,
            embed_lr=0.3,
            head_lr=1.0 / 320,
            scalar_lr=0.01,
            muon_lr=0.035,
            muon_wd=0.025,
            cooldown_frac=0.7,
        )

    # Phase A: build KN n-gram tables (lower tier L).
    order_tables, continuation = build_w31_kn_tables(
        train_bytes_u8, max_order=kn_max_order,
    )

    # Phase B: precompute surprise mask via vectorized KN-MLE lookups.
    print(f"[chunker] computing surprise mask (tau={TAU}) ...", flush=True)
    t_surprise = time.monotonic()
    surprise_mask = _build_surprise_mask_gpu(
        train_bytes_u8, order_tables, tau=TAU,
        max_order=kn_max_order,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    n_total = surprise_mask.numel()
    n_surprise = int(surprise_mask.sum().item())
    p_s = n_surprise / max(1, n_total)
    print(f"[chunker] surprise computed in {time.monotonic()-t_surprise:.1f}s: "
          f"p_s = {p_s:.4f} ({n_surprise:,}/{n_total:,})", flush=True)

    # Phase C: train H on surprise positions (masked CE).
    model = _train_h_with_surprise_mask(
        train_bytes_u8, surprise_mask, h_cfg, device,
    )

    return ChunkerPhase1CharModel(
        model, order_tables, continuation,
        max_ctx_len=kn_max_order - 1, discount=KN_DISCOUNT,
        alpha=ALPHA, tau=TAU, device=device,
    )
