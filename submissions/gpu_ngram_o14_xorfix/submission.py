"""GPU port at order-14 with XOR-bit-flip sign-fix sort (W31 + 2 orders + GPU fix).

Variant of gpu_ngram_o14 that eliminates the ~150s CPU re-sort overhead
by performing a sign-bit-corrected sort directly on GPU.

The bug
=======
When packing k>=9 byte windows into two int64 (hi, lo), any byte in slot
[k-8..k-1] with high bit set (>= 0x80) causes the encoded int64 to land
in the negative half of signed int64. torch.sort is a SIGNED sort →
groups same-byte rows correctly (the bit pattern within an equivalence
class is identical) but the GLOBAL order of distinct (hi, lo) keys is
scrambled because negative values sort BEFORE positive values.

This breaks np.searchsorted in the KN predict path, which assumes
unsigned lex order.

The fix
=======
Flip the sign bit before sorting:
  sort_lo = lo XOR (1 << 63)
  sort_hi = hi XOR (1 << 63)

This re-maps the signed sort range so that the original unsigned
ordering is preserved: bytes 0x00..0x7F map to 0x8000_0000_0000_0000..
0xFFFF_FFFF_FFFF_FFFF (large positive), and bytes 0x80..0xFF map to
0x0000_0000_0000_0000..0x7FFF_FFFF_FFFF_FFFF (small positive). Both
halves sort in the right order, and signed sort now produces unsigned
lex order.

We argsort by sort_lo (stable), then argsort by sort_hi (stable) — same
two-pass pattern as before, just on the XOR-shifted keys. After sort,
hi/lo/counts are reindexed; no XOR-back is needed because the byte
decoding in _gpu_table_to_w3_layout reads the original (un-XORed)
values, which we keep around.

Expected outcome
================
gpu_ngram_o14: 5,143 J / 0.7184 acc, dominated by the 150s CPU re-sort.
Eliminating that → ~25s build → 1.5-2.5 kJ at the same 0.7184 acc,
matching or beating W3 CPU on energy AND matching it on acc, all
L2-clean (GPU-active throughout build).
"""
from __future__ import annotations

__author__ = "@gabrielnan"

import os
import time

import numpy as np
import torch
from torch import Tensor

from wikitext import CharModel


MAX_ORDER = 14  # context window includes next byte; ctx_len = MAX_ORDER - 1
MAX_CTX_LEN = MAX_ORDER - 1
KN_DISCOUNT = 0.5
NGRAM_EPS = 1e-3

# Constant 1 << 63 as Python int — overflows int64 if you write it as a
# tensor literal, so we keep it as a Python int and XOR via a wrap-aware
# torch.bitwise_xor against a precomputed int64 tensor in _sort_and_dedupe.
SIGN_BIT = 1 << 63
# Two's-complement signed-int64 representation of 1<<63 is -(1<<63) =
# -9223372036854775808.
SIGN_BIT_AS_INT64 = -SIGN_BIT


# ---------------------------------------------------------------------------
# Dual-int64 key encoding helpers.
# ---------------------------------------------------------------------------

def _pack_window_chunk(
    arr_int64: Tensor,
    start: int,
    end: int,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Return (hi, lo) int64 tensors packing all k-byte windows in
    arr_int64[start:end]. Identical to W31 packing — the XOR fix is
    applied only at sort time, not at storage."""
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
    """Sort (hi, lo) by UNSIGNED lex order on GPU via XOR sign-bit fix,
    then sum counts per unique (hi, lo).

    The signed-int64 GPU sort is corrected by XOR-flipping the sign bit
    on a SORT KEY copy (sort_lo = lo ^ (1<<63), sort_hi = hi ^ (1<<63)).
    The original (un-XORed) hi, lo values are kept and reindexed by the
    sort permutation. This avoids the 150s CPU re-sort that the previous
    gpu_ngram_o14 needed.
    """
    if hi.numel() == 0:
        return hi, lo, counts
    device = hi.device
    sign_bit = torch.tensor(SIGN_BIT_AS_INT64, dtype=torch.int64, device=device)
    # Sort keys with sign bit flipped → signed-sort produces unsigned lex.
    sort_lo = lo.bitwise_xor(sign_bit)
    sort_hi = hi.bitwise_xor(sign_bit)
    # Stable sort by sort_lo, then stable sort by sort_hi → lex sort.
    order_lo = torch.argsort(sort_lo, stable=True)
    sort_hi = sort_hi[order_lo]
    hi = hi[order_lo]
    lo = lo[order_lo]
    counts = counts[order_lo]
    order_hi = torch.argsort(sort_hi, stable=True)
    hi = hi[order_hi]
    lo = lo[order_hi]
    counts = counts[order_hi]
    del order_lo, order_hi, sort_hi, sort_lo
    # RLE on (hi, lo) pairs (original encoded values; equality is bit-identity).
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
    """Build unique (hi, lo, count) for order-k windows on GPU."""
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
    """Drop leftmost byte from each k-byte key, re-sort, sum counts."""
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
        else:  # new_k == 8
            new_hi = torch.zeros_like(hi)
            new_lo = lo
    else:
        new_hi = torch.zeros_like(hi)
        new_lo = lo & ((1 << (new_k * 8)) - 1)

    return _sort_and_dedupe(new_hi, new_lo, counts)


# ---------------------------------------------------------------------------
# Build per-order KN tables (CPU-side numpy arrays for predict).
#
# NOTE: unlike gpu_ngram_o14, we DO NOT re-sort on CPU. The XOR-corrected
# GPU sort already produces unsigned lex order. We just need to decode
# the (hi, lo) into raw bytes and then run the same RLE-on-ctx logic.
# ---------------------------------------------------------------------------

def _gpu_table_to_w3_layout(
    hi: Tensor, lo: Tensor, counts: Tensor, k: int,
) -> dict:
    """Build the W3-format order dict from (hi, lo, counts) at order k.

    The GPU sort with XOR fix already produces unsigned lex order, so
    no CPU re-sort is needed. Bytes are decoded from (hi, lo) into a
    contiguous uint8 array; ctx group boundaries found by row-equality.
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

    # NOTE: NO CPU re-sort. The GPU XOR-fixed sort already gave us
    # unsigned lex order. Verify only in smoke if needed.

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
    counts = np.bincount(bigram_next_arr, minlength=256).astype(np.float64)
    s = counts.sum()
    if s > 0:
        counts /= s
    else:
        counts[:] = 1.0 / 256.0
    return counts


# ---------------------------------------------------------------------------
# CharModel — KN-smoothed predict (same as W3/W31/O14).
# ---------------------------------------------------------------------------

class DeepBackoffKNModel(CharModel):
    def __init__(
        self,
        order_tables: list,
        continuation: np.ndarray,
        max_ctx_len: int,
        discount: float,
    ):
        self._tables = order_tables
        self._max_ctx_len = max_ctx_len
        self._D = float(discount)
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
            n_distinct = int(tbl["n_distinct_per_ctx"][idx])
            if total <= 0.0:
                continue
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
        print(f"[gpu_ngram_o14_xorfix] SEED={seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = train_text.encode("utf-8")
    is_smoke = len(raw) < SMOKE_TRAIN_BYTES

    max_order = MAX_ORDER
    if is_smoke:
        max_order = min(MAX_ORDER, max(2, len(raw) // 32))
        print(f"[gpu_ngram_o14_xorfix] SMOKE mode (train={len(raw)} bytes) max_order={max_order}")

    discount = KN_DISCOUNT
    print(f"[gpu_ngram_o14_xorfix] starting build; max_order={max_order} D={discount}",
          flush=True)

    t_total = time.monotonic()
    train_bytes_u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n_bytes = train_bytes_u8.numel()
    print(f"[gpu_ngram_o14_xorfix] encoded train: {n_bytes:,} bytes ({time.monotonic()-t_total:.1f}s)",
          flush=True)

    t0 = time.monotonic()
    top_k = max_order
    hi, lo, counts = _build_top_order_gpu(train_bytes_u8, top_k)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[gpu_ngram_o14_xorfix] top order={top_k} unique pairs: {hi.numel():,}  "
          f"{time.monotonic()-t0:.1f}s", flush=True)

    order_tables = [None] * max_order

    t0 = time.monotonic()
    order_tables[top_k - 1] = _gpu_table_to_w3_layout(hi, lo, counts, top_k)
    print(f"[gpu_ngram_o14_xorfix] ctx_len={top_k-1} ctxs={order_tables[top_k-1]['ctx_keys'].shape[0]:,} "
          f"rows={order_tables[top_k-1]['next_bytes'].shape[0]:,}  "
          f"{time.monotonic()-t0:.1f}s", flush=True)

    bigram_next_for_base = None
    for new_k in range(top_k - 1, 0, -1):
        t0 = time.monotonic()
        hi, lo, counts = _step_down_gpu(hi, lo, counts, new_k + 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        order_tables[new_k - 1] = _gpu_table_to_w3_layout(hi, lo, counts, new_k)
        tbl = order_tables[new_k - 1]
        print(f"[gpu_ngram_o14_xorfix] ctx_len={new_k-1} ctxs={tbl['ctx_keys'].shape[0]:,} "
              f"rows={tbl['next_bytes'].shape[0]:,}  "
              f"{time.monotonic()-t0:.1f}s", flush=True)
        if new_k == 2:
            bigram_next_for_base = tbl["next_bytes"].copy()

    if bigram_next_for_base is not None:
        continuation = _build_continuation_base(bigram_next_for_base)
    else:
        continuation = np.full(256, 1.0 / 256.0, dtype=np.float64)

    print(f"[gpu_ngram_o14_xorfix] total build: {time.monotonic()-t_total:.1f}s",
          flush=True)

    return DeepBackoffKNModel(
        order_tables=order_tables,
        continuation=continuation,
        max_ctx_len=max_order - 1,
        discount=discount,
    )
