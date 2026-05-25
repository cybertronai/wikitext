"""Modified Kneser-Ney at K=11 with per-count discounts (Chen-Goodman MKN).

Paradigm A6 (Pitman-Yor / MKN family). Iter-4 exp 7/10.

Hypothesis: KN uses single D=0.5 for all count values. Modified KN
(Chen & Goodman 1996) uses D1 for c=1, D2 for c=2, D3 for c>=3,
computed per-order from count-of-counts statistics. Slight acc lift
possible at iso-K with no J cost change.

Mechanism:
  * Encode train_text as uint8 tensor on GPU.
  * For order k = MAX_ORDER (= 12 here, slightly less than W3's 14 so
    the dual-int64 key encoding stays simple), build sliding k-byte
    windows packed into two int64s per window: hi = leftmost max(0, k-8)
    bytes, lo = rightmost min(k, 8) bytes.
  * torch.unique-via-sort on (hi, lo) lex: do stable sort by lo then by
    hi, then RLE to find unique (hi, lo) pairs with summed counts.
  * Chained step-down to lower orders: drop leftmost byte from the key
    (hi <<= 8 conceptually, masking and shifting between hi/lo), re-sort
    and sum counts.
  * KN-smoothed predict: at each context, walk from longest order down
    accumulating discounted mass + interpolating with lower-order
    estimate. Same recurrence as W3.

Cap at order 12 (vs W3's 14) for build-time safety. Expected accuracy
~0.7150 (between E1's 0.7086 and W3's 0.7184).
"""
from __future__ import annotations

__author__ = "@gabrielnan"

import os
import time

import numpy as np
import torch
from torch import Tensor

from wikitext import CharModel


MAX_ORDER = 11  # context window includes next byte; ctx_len = MAX_ORDER - 1
MAX_CTX_LEN = MAX_ORDER - 1
KN_DISCOUNT = 0.5
NGRAM_EPS = 1e-3


# ---------------------------------------------------------------------------
# Dual-int64 key encoding helpers.
#
# A k-byte window [b0, b1, ..., b_{k-1}] (b0 leftmost) is packed as:
#   if k <= 8: hi = 0; lo = b0 * 256^(k-1) + ... + b_{k-1}
#   if k >  8: hi = b0 * 256^(k-9) + ... + b_{k-9}
#              lo = b_{k-8} * 256^7 + ... + b_{k-1}
# Lex order on the original byte tuple corresponds to lex on (hi, lo).
# ---------------------------------------------------------------------------

def _pack_window_chunk(
    arr_int64: Tensor,  # full byte stream as int64 on GPU
    start: int,
    end: int,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Return (hi, lo) int64 tensors of shape (n_windows,) packing all
    k-byte windows fully contained in arr_int64[start:end].

    n_windows = (end - start) - k + 1 (assumes end - start >= k).
    """
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
        # hi packs first k-8 bytes; lo packs last 8 bytes.
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
    """Sort (hi, lo) lex (asc) and sum counts per unique (hi, lo).

    counts is float32. Returns (uniq_hi, uniq_lo, uniq_counts).
    """
    if hi.numel() == 0:
        return hi, lo, counts
    device = hi.device
    # Stable sort by lo, then stable sort by hi → lex sort.
    order_lo = torch.argsort(lo, stable=True)
    hi = hi[order_lo]
    lo = lo[order_lo]
    counts = counts[order_lo]
    order_hi = torch.argsort(hi, stable=True)
    hi = hi[order_hi]
    lo = lo[order_hi]
    counts = counts[order_hi]
    del order_lo, order_hi
    # RLE on (hi, lo) pairs.
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
    """Build unique (hi, lo, count) for order-k windows on GPU.

    Returns three 1-D int64/float32 tensors, lex-sorted by (hi, lo).
    Processes in chunks with (k-1)-byte overlap; pairwise merges.
    """
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
        # Dedupe within chunk first.
        hi, lo, cnt = _sort_and_dedupe(hi, lo, cnt)
        # Merge with accumulator.
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
    """Drop leftmost byte from each k-byte key, re-sort, sum counts.

    Returns the new (hi, lo, counts) at order k-1.
    """
    if hi.numel() == 0 or k <= 1:
        device = hi.device
        return (torch.zeros(0, dtype=torch.int64, device=device),
                torch.zeros(0, dtype=torch.int64, device=device),
                torch.zeros(0, dtype=torch.float32, device=device))

    new_k = k - 1
    # New encoding: pack new_k bytes which are the original b1..b_{k-1}.
    if k > 8:
        if new_k > 8:
            # Both old and new have hi+lo. Drop b0:
            # old hi had b0..b_{k-9} packed; new hi has b1..b_{k-9} = old hi without b0.
            # new hi = old hi & ((1 << ((new_k - 8)*8)) - 1)
            new_hi = hi & ((1 << ((new_k - 8) * 8)) - 1)
            new_lo = lo
        else:  # new_k <= 8 (i.e. k == 9, new_k == 8)
            # All bytes b1..b8 are in old lo. New hi = 0, new lo = old lo.
            new_hi = torch.zeros_like(hi)
            new_lo = lo
    else:
        # k <= 8: all in lo. Drop b0 from lo.
        new_hi = torch.zeros_like(hi)
        new_lo = lo & ((1 << (new_k * 8)) - 1)

    # Re-sort and dedupe (multiple old keys may collapse to same new key).
    return _sort_and_dedupe(new_hi, new_lo, counts)


# ---------------------------------------------------------------------------
# Build per-order KN tables (CPU-side numpy arrays for predict).
#
# After all builds finish on GPU, transfer to CPU. We use the same numpy
# layout as W3 (DeepBackoffKNModel) so the KN predict code path can be
# reused verbatim.
# ---------------------------------------------------------------------------

def _gpu_table_to_w3_layout(
    hi: Tensor, lo: Tensor, counts: Tensor, k: int,
) -> dict:
    """Build the W3-format order dict from sorted (hi, lo, counts) at order k.

    Output dict keys (mirror W3's _build_order_tables):
      ctx_len, ctx_keys (M, ctx_len) uint8, ctx_view (void view),
      ctx_offsets (M+1) int64, next_bytes uint8, counts int32,
      total_count_per_ctx int64, n_distinct_per_ctx int32.
    """
    ctx_len = k - 1
    n = hi.numel()

    # Decode each (hi, lo) into a length-k uint8 array of bytes (b0..b_{k-1}).
    hi_cpu = hi.cpu().numpy()
    lo_cpu = lo.cpu().numpy()
    counts_cpu = counts.cpu().numpy().astype(np.int64)

    bytes_arr = np.zeros((n, k), dtype=np.uint8)
    if n > 0:
        # k bytes: leftmost max(0, k-8) come from hi, rest from lo.
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
        # Unigram: single empty ctx; all bytes are "next".
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
    # Find start positions of distinct ctxs (rows where ctx changes).
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
    total_per_ctx = np.add.reduceat(counts_cpu, starts) if n_ctx > 0 else np.zeros(0, dtype=np.int64)
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
    """Unigram continuation distribution: p_cont(c) ∝ |{h : N(h,c) > 0}|.

    bigram_next_arr is the order-2 `next_bytes` (one row per distinct
    (h, c) pair where h is a single byte). bincount over next gives
    the count of distinct preceding bytes per c.
    """
    counts = np.bincount(bigram_next_arr, minlength=256).astype(np.float64)
    s = counts.sum()
    if s > 0:
        counts /= s
    else:
        counts[:] = 1.0 / 256.0
    return counts


# ---------------------------------------------------------------------------
# CharModel — KN-smoothed predict (reuses W3's logic, predict on CPU).
# ---------------------------------------------------------------------------

class DeepBackoffKNModel(CharModel):
    def __init__(
        self,
        order_tables: list,
        continuation: np.ndarray,
        max_ctx_len: int,
        discount: float,
        mkn_discounts: list = None,  # list of (D1, D2, D3) per order, indexed by k
    ):
        self._tables = order_tables
        self._max_ctx_len = max_ctx_len
        self._D = float(discount)
        # mkn_discounts[k] = (D1, D2, D3) for that order; if None use scalar D
        self._mkn = mkn_discounts
        self._p_base = continuation.astype(np.float64)
        self._history = bytearray()

    def reset(self) -> None:
        self._history.clear()

    def predict(self) -> dict[str, float]:
        p = self._kn_dist()
        best = int(p.argmax())
        return {chr(best): 1.0}

    def observe(self, char: str) -> None:
        self._history.extend(char.encode("utf-8"))
        if len(self._history) > self._max_ctx_len:
            del self._history[:-self._max_ctx_len]

    def _kn_dist(self) -> np.ndarray:
        D = self._D
        p = self._p_base.copy()
        history = self._history
        hist_len = len(history)
        max_k = min(self._max_ctx_len, hist_len)
        if max_k == 0:
            return p

        for k in range(1, max_k + 1):
            tbl = self._tables[k]
            ctx_view = tbl["ctx_view"]
            if ctx_view is None or ctx_view.shape[0] == 0:
                continue
            tail = bytes(history[-k:])
            q = np.frombuffer(tail, dtype=np.uint8).view(
                np.dtype((np.void, k))
            )[0]
            idx = int(np.searchsorted(ctx_view, q))
            if idx >= ctx_view.shape[0] or ctx_view[idx] != q:
                continue
            lo = int(tbl["ctx_offsets"][idx])
            hi = int(tbl["ctx_offsets"][idx + 1])
            nb = tbl["next_bytes"][lo:hi]
            cn = tbl["counts"][lo:hi].astype(np.float64)
            total = float(tbl["total_count_per_ctx"][idx])
            if total <= 0.0:
                continue
            if self._mkn is not None and self._mkn[k] is not None:
                D1, D2, D3 = self._mkn[k]
                # Discount each count by its bucket.
                d_arr = np.where(cn == 1, D1, np.where(cn == 2, D2, D3))
                discounted = np.maximum(cn - d_arr, 0.0) / total
                # Lambda for backoff: sum of discount mass / total
                # MKN lambda = (D1 * N1 + D2 * N2 + D3 * N3+) / total
                # where Nk = number of distinct next-bytes with count == k
                N1 = np.sum(cn == 1)
                N2 = np.sum(cn == 2)
                N3 = np.sum(cn >= 3)
                lam = (D1 * N1 + D2 * N2 + D3 * N3) / total
            else:
                n_distinct = int(tbl["n_distinct_per_ctx"][idx])
                discounted = np.maximum(cn - D, 0.0) / total
                lam = D * n_distinct / total
            p_new = lam * p
            p_new[nb] = p_new[nb] + discounted
            p = p_new
        return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SMOKE_TRAIN_BYTES = 10_000


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[gpu_ngram_w3] SEED={seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = train_text.encode("utf-8")
    is_smoke = len(raw) < SMOKE_TRAIN_BYTES

    max_order = MAX_ORDER
    if is_smoke:
        # Clamp to fit tiny corpus.
        max_order = min(MAX_ORDER, max(2, len(raw) // 32))
        print(f"[gpu_ngram_w3] SMOKE mode (train={len(raw)} bytes) max_order={max_order}")

    discount = KN_DISCOUNT
    print(f"[gpu_ngram_w3] starting build; max_order={max_order} D={discount}",
          flush=True)

    t_total = time.monotonic()
    SUBSET_FRAC = float(os.environ.get("SUBSET_FRAC", "0.7"))
    if not is_smoke and SUBSET_FRAC < 1.0:
        raw = raw[:int(len(raw) * SUBSET_FRAC)]
        print(f"[gpu_ngram_w3] SUBSET {SUBSET_FRAC} -> {len(raw):,} train bytes", flush=True)
    train_bytes_u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n_bytes = train_bytes_u8.numel()
    print(f"[gpu_ngram_w3] encoded train: {n_bytes:,} bytes ({time.monotonic()-t_total:.1f}s)",
          flush=True)

    # Build top-order on GPU.
    t0 = time.monotonic()
    top_k = max_order
    hi, lo, counts = _build_top_order_gpu(train_bytes_u8, top_k)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[gpu_ngram_w3] top order={top_k} unique pairs: {hi.numel():,}  "
          f"{time.monotonic()-t0:.1f}s", flush=True)

    # Order_tables[k] for k in 0..max_ctx_len.
    order_tables = [None] * max_order  # indices 0..max_order-1 = ctx_len 0..MAX_CTX_LEN

    # Top order: transfer to W3 layout.
    t0 = time.monotonic()
    order_tables[top_k - 1] = _gpu_table_to_w3_layout(hi, lo, counts, top_k)
    print(f"[gpu_ngram_w3] ctx_len={top_k-1} ctxs={order_tables[top_k-1]['ctx_keys'].shape[0]:,} "
          f"rows={order_tables[top_k-1]['next_bytes'].shape[0]:,}  "
          f"{time.monotonic()-t0:.1f}s", flush=True)

    # Chained step-down.
    bigram_next_for_base = None
    for new_k in range(top_k - 1, 0, -1):
        t0 = time.monotonic()
        hi, lo, counts = _step_down_gpu(hi, lo, counts, new_k + 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        order_tables[new_k - 1] = _gpu_table_to_w3_layout(hi, lo, counts, new_k)
        tbl = order_tables[new_k - 1]
        print(f"[gpu_ngram_w3] ctx_len={new_k-1} ctxs={tbl['ctx_keys'].shape[0]:,} "
              f"rows={tbl['next_bytes'].shape[0]:,}  "
              f"{time.monotonic()-t0:.1f}s", flush=True)
        # Capture bigram (ctx_len=1, k=2) next_bytes for continuation base.
        if new_k == 2:
            bigram_next_for_base = tbl["next_bytes"].copy()

    # Continuation base from bigram (or unigram if max_order < 2).
    if bigram_next_for_base is not None:
        continuation = _build_continuation_base(bigram_next_for_base)
    else:
        continuation = np.full(256, 1.0 / 256.0, dtype=np.float64)

    # ---- MKN per-order discount computation ----
    t0 = time.monotonic()
    mkn_discounts = [None] * max_order
    # Skip MKN on tiny corpus where count statistics are unreliable.
    use_mkn = (n_bytes > 1_000_000) and (not is_smoke)
    if use_mkn:
        for k in range(1, max_order):
            tbl = order_tables[k]
            if tbl is None or tbl["counts"].shape[0] == 0:
                continue
            cn = tbl["counts"]  # count of each (ctx, next) pair
            n1 = int(np.sum(cn == 1))
            n2 = int(np.sum(cn == 2))
            n3 = int(np.sum(cn == 3))
            n4 = int(np.sum(cn == 4))
            # Chen-Goodman formulas — require n1 > n2 > n3 > n4 (the
            # typical n-gram regime). If reversed (dense small corpus),
            # the formula produces negative D values. Skip MKN if so.
            if n1 + 2 * n2 == 0:
                mkn_discounts[k] = (0.5, 0.5, 0.5)
                continue
            if n1 < n2 or n2 < n3:
                # Reversed regime — formula invalid; use scalar.
                mkn_discounts[k] = (0.5, 0.5, 0.5)
                continue
            Y = n1 / (n1 + 2 * n2)
            D1 = 1.0 - 2.0 * Y * (n2 / max(n1, 1))
            D2 = 2.0 - 3.0 * Y * (n3 / max(n2, 1))
            D3 = 3.0 - 4.0 * Y * (n4 / max(n3, 1))
            # Clamp to sensible ranges (literature: D1 ~ 0.5, D2 ~ 1, D3+ ~ 1.5)
            D1 = max(0.1, min(1.0, D1))
            D2 = max(0.1, min(2.0, D2))
            D3 = max(0.1, min(3.0, D3))
            mkn_discounts[k] = (D1, D2, D3)
            print(f"[mkn] k={k} D1={D1:.3f} D2={D2:.3f} D3={D3:.3f} (n1={n1}, n2={n2}, n3={n3})", flush=True)
    else:
        print(f"[mkn] skipping MKN (tiny corpus or smoke); fallback to scalar D=0.5", flush=True)
    print(f"[mkn] discounts computed: {time.monotonic()-t0:.1f}s", flush=True)

    print(f"[gpu_ngram_w3] total build: {time.monotonic()-t_total:.1f}s",
          flush=True)

    return DeepBackoffKNModel(
        order_tables=order_tables,
        continuation=continuation,
        max_ctx_len=max_order - 1,
        discount=discount,
        mkn_discounts=mkn_discounts,
    )
