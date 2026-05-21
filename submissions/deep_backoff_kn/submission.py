"""Order-15 chained-backoff byte-level n-gram predictor with Kneser-Ney smoothing.

Paradigm: CLA-001 (extension of E1 — deeper order + smoothing).

Mechanism:
  * Training: like E1 v2, encode train_text as UTF-8 bytes, run parallel
    chunk-wise np.unique over 15-byte sliding windows (14-byte ctx + 1
    next byte), merge into a single global lex-sorted (ctx, next, count)
    table. Then chained step-down: for each order from 14..1, drop the
    leftmost ctx byte, re-sort, sum counts. At every order, retain the
    FULL sorted (ctx, next, count) table (not just argmax) so we can
    query a context's full distribution at predict-time. Also precompute
    per-order ctx prefix offsets so searchsorted gives O(log M) ctx
    lookup and the row range [lo:hi) gives that ctx's distribution.
  * Predict: walk from the longest matched context down to unigram,
    incrementally folding the Kneser-Ney smoothed distribution:
      p_kn(c|h) = max(N(h,c) - D, 0) / N(h) + (D * N+(h,*) / N(h)) * p_kn(c|h')
    where N+(h,*) is the number of distinct continuations of h. Greedy
    argmax over the final mixed 256-byte distribution.
  * Observe: append the encoded char to a rolling 14-byte history.

Memory: ~12-18 GB across all 15 sorted tables; fits comfortably on Modal
A100 host RAM (80+ GB).

L2 caveat: training is CPU/numpy only — same posture as E1. The W3 brief
acknowledges this: if the GPU-port (W1) lands, this paradigm gets ported
there. Until then, accepts the L2-spirit flag and runs the algorithm.
"""
from __future__ import annotations

__author__ = "@nakajimagabriel"

import multiprocessing
import os
import sys
import time
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from wikitext import CharModel


_FORK_TRAIN_BYTES: Optional[bytes] = None


# Maximum context length to build (order = MAX_CTX_LEN + 1).
#
# The W3 brief targets order-15 (ctx_len=14). A local full-data trial
# with MAX_CTX_LEN=14 measured ~182 s on Apple M-series (np.unique=64 s,
# step-down chain summed to ~110 s); scaled to Modal's ~1.5-1.9× slower
# per-thread CPU, that projects to 270-345 s — at or above the 300 s
# wall-clock cap.
#
# We ship MAX_CTX_LEN=13 (order-14) instead: a clean 2 orders deeper
# than E1's order-12, still tests the "deeper context + KN smoothing"
# hypothesis, and has comfortable timing margin (~140 s local → ~210-265 s
# Modal). Overridable via the DEEP_BACKOFF_MAX_CTX env var for follow-up
# experiments (e.g. submitting an order-15 retry once we know the host
# performance).
MAX_CTX_LEN: int = int(os.environ.get("DEEP_BACKOFF_MAX_CTX", "13"))

# Kneser-Ney absolute discount. Standard fixed-D in [0.5, 0.9].
KN_DISCOUNT: float = 0.5


# ---------------------------------------------------------------------------
# Build phase — parallel chunked np.unique, then chained step-down.
# ---------------------------------------------------------------------------

def _group_starts(view: np.ndarray) -> np.ndarray:
    """Starting indices of contiguous equal-value runs in a 1-D ndarray."""
    M = len(view)
    if M <= 1:
        return np.zeros(1, dtype=np.int64)
    changes = view[1:] != view[:-1]
    return np.concatenate([[0], np.flatnonzero(changes) + 1])


def _unique_windows(arr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """np.unique over k-byte sliding windows of `arr`. Returns
    (uniq_bytes_uint8(M,k), counts_int64(M,)), lex-sorted by full window.
    """
    if len(arr) < k:
        return np.empty((0, k), dtype=np.uint8), np.empty(0, dtype=np.int64)
    windows = sliding_window_view(arr, k)
    windows_c = np.ascontiguousarray(windows)
    row_view = windows_c.view(np.dtype((np.void, k)))[:, 0]
    uniq, counts = np.unique(row_view, return_counts=True)
    uniq_bytes = uniq.view(np.uint8).reshape(-1, k)
    return uniq_bytes, counts.astype(np.int64, copy=False)


def _chunk_unique_worker(args: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    start, end, k = args
    assert _FORK_TRAIN_BYTES is not None
    arr = np.frombuffer(_FORK_TRAIN_BYTES, dtype=np.uint8, offset=start,
                        count=end - start)
    return _unique_windows(arr, k)


def _merge_sorted_uniques(
    parts: list[tuple[np.ndarray, np.ndarray]], k: int
) -> tuple[np.ndarray, np.ndarray]:
    if not parts:
        return np.empty((0, k), dtype=np.uint8), np.empty(0, dtype=np.int64)
    if len(parts) == 1:
        return parts[0]

    all_rows = np.concatenate([p[0] for p in parts], axis=0)
    all_counts = np.concatenate([p[1] for p in parts], axis=0)
    rows_view = all_rows.view(np.dtype((np.void, k)))[:, 0]
    order = np.argsort(rows_view, kind="stable")
    sorted_rows = all_rows[order]
    sorted_counts = all_counts[order]
    sorted_view = sorted_rows.view(np.dtype((np.void, k)))[:, 0]
    starts = _group_starts(sorted_view)
    merged_counts = np.add.reduceat(sorted_counts, starts)
    merged_rows = sorted_rows[starts]
    return merged_rows, merged_counts


def _build_top_order(
    train_bytes: bytes, max_ctx_len: int = MAX_CTX_LEN,
    *, n_workers: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the order-(max_ctx_len+1) unique table from `train_bytes`."""
    global _FORK_TRAIN_BYTES
    k = max_ctx_len + 1
    n_bytes = len(train_bytes)

    arr_full = np.frombuffer(train_bytes, dtype=np.uint8)
    if n_bytes < 2_000_000:
        return _unique_windows(arr_full, k)

    if n_workers is None:
        n_workers = min(8, max(1, multiprocessing.cpu_count()))

    body = n_bytes - (k - 1)
    if body <= 0:
        return _unique_windows(arr_full, k)

    starts = np.linspace(0, body, n_workers + 1, dtype=np.int64)
    chunks: list[tuple[int, int, int]] = []
    for i in range(n_workers):
        s = int(starts[i])
        e_window = int(starts[i + 1])
        if e_window <= s:
            continue
        chunk_end = e_window + (k - 1)
        if chunk_end > n_bytes:
            chunk_end = n_bytes
        chunks.append((s, chunk_end, k))

    if not chunks:
        return _unique_windows(arr_full, k)

    del arr_full

    _FORK_TRAIN_BYTES = train_bytes
    try:
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            print("[deep-backoff-kn] WARNING: fork unavailable, "
                  "falling back to serial unique", flush=True)
            return _unique_windows(np.frombuffer(train_bytes, dtype=np.uint8), k)
        with ctx.Pool(processes=len(chunks)) as pool:
            parts = pool.map(_chunk_unique_worker, chunks)
    finally:
        _FORK_TRAIN_BYTES = None

    return _merge_sorted_uniques(parts, k)


def _step_down(
    table_bytes: np.ndarray, table_counts: np.ndarray, new_ctx_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the leftmost ctx byte; sum counts over the dropped byte;
    return a new lex-sorted (ctx, next, count) table for order new_ctx_len+1.
    """
    new_row_len = new_ctx_len + 1
    projected = table_bytes[:, 1:]
    projected_c = np.ascontiguousarray(projected)
    pv = projected_c.view(np.dtype((np.void, new_row_len)))[:, 0]
    order = pv.argsort(kind="stable")
    sorted_rows = projected_c[order]
    sorted_counts = table_counts[order]
    sorted_view = sorted_rows.view(np.dtype((np.void, new_row_len)))[:, 0]
    pair_starts = _group_starts(sorted_view)
    agg_counts = np.add.reduceat(sorted_counts, pair_starts)
    agg_rows = sorted_rows[pair_starts]
    return agg_rows, agg_counts


def _build_order_tables(
    table_bytes: np.ndarray, table_counts: np.ndarray, ctx_len: int,
) -> dict:
    """For a lex-sorted (ctx, next, count) table at order ctx_len+1,
    derive the data structures needed by KN at predict-time:

      * ctx_keys: shape (M, ctx_len) — unique contexts at this order
      * ctx_view: void-typed 1-D view of ctx_keys (for searchsorted)
      * ctx_offsets: int64 array of shape (M+1,) — row ranges in next/count
      * next_bytes: uint8 array of shape (total_rows,)
      * counts: int32 array of shape (total_rows,)
      * total_count_per_ctx: int64 array shape (M,) — N(h) (sum of counts)
      * n_distinct_per_ctx: int32 array shape (M,) — N+(h, *)

    For ctx_len == 0 (unigram), `ctx_keys` is empty and there's exactly
    one "ctx" (the empty one).
    """
    M = table_bytes.shape[0]
    next_arr = table_bytes[:, ctx_len].copy()
    counts_arr = table_counts.astype(np.int32, copy=False)

    if ctx_len == 0:
        total = int(table_counts.sum())
        return {
            "ctx_len": 0,
            "ctx_keys": np.empty((1, 0), dtype=np.uint8),
            "ctx_view": None,
            "ctx_offsets": np.array([0, M], dtype=np.int64),
            "next_bytes": next_arr,
            "counts": counts_arr,
            "total_count_per_ctx": np.array([total], dtype=np.int64),
            "n_distinct_per_ctx": np.array([M], dtype=np.int32),
        }

    ctx_arr = np.ascontiguousarray(table_bytes[:, :ctx_len])
    ctx_view_full = ctx_arr.view(np.dtype((np.void, ctx_len)))[:, 0]
    starts = _group_starts(ctx_view_full)
    n_ctx = starts.shape[0]
    ctx_keys = ctx_arr[starts]
    ctx_keys_c = np.ascontiguousarray(ctx_keys)
    ctx_view = ctx_keys_c.view(np.dtype((np.void, ctx_len)))[:, 0]

    ctx_offsets = np.empty(n_ctx + 1, dtype=np.int64)
    ctx_offsets[:n_ctx] = starts
    ctx_offsets[n_ctx] = M

    # Per-ctx total count and distinct-next count.
    total_per_ctx = np.add.reduceat(counts_arr.astype(np.int64), starts)
    n_distinct = (ctx_offsets[1:] - ctx_offsets[:-1]).astype(np.int32)

    return {
        "ctx_len": ctx_len,
        "ctx_keys": ctx_keys_c,
        "ctx_view": ctx_view,
        "ctx_offsets": ctx_offsets,
        "next_bytes": next_arr,
        "counts": counts_arr,
        "total_count_per_ctx": total_per_ctx,
        "n_distinct_per_ctx": n_distinct,
    }


# ---------------------------------------------------------------------------
# Continuation distribution for KN base (unigram → continuation form).
# ---------------------------------------------------------------------------

def _build_continuation_base(
    bigram_table_bytes: np.ndarray,
) -> np.ndarray:
    """Compute the continuation distribution for the unigram base:
        p_cont(c) ∝ |{h : N(h, c) > 0}|
    i.e. for each byte c, how many distinct order-1 contexts h precede it.

    Uses the order-2 (ctx_len=1) sorted (ctx, next) unique table — each
    distinct row contributes 1 to its `next` byte. Returns shape (256,).
    """
    next_arr = bigram_table_bytes[:, 1]
    counts = np.bincount(next_arr, minlength=256).astype(np.float64)
    s = counts.sum()
    if s > 0:
        counts /= s
    else:
        counts[:] = 1.0 / 256.0
    return counts


# ---------------------------------------------------------------------------
# CharModel implementation
# ---------------------------------------------------------------------------

class DeepBackoffKNModel(CharModel):
    """Order-15 byte-level n-gram with Kneser-Ney interpolated backoff.

    Predict-time per char: O(MAX_CTX_LEN * log M) plus 256-vector ops.
    """

    def __init__(
        self,
        order_tables: list[dict],
        continuation: np.ndarray,
        max_ctx_len: int = MAX_CTX_LEN,
        discount: float = KN_DISCOUNT,
    ):
        self._tables = order_tables
        self._max_ctx_len = max_ctx_len
        self._D = float(discount)
        # Base distribution: order-1 continuation prior.
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
        """Compute the KN-interpolated distribution over the 256 byte
        alphabet for the current history.

        Walks from order 1 up to the maximum matched order, blending the
        continuation distribution with each successively longer context's
        evidence using the standard interpolated KN recurrence:
            p_kn(c|h) = max(N(h,c) - D, 0) / N(h)
                       + (D * N+(h,*) / N(h)) * p_kn(c|h')
        """
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
                # Context not seen at this order: KN says fall back fully
                # to the lower-order distribution (i.e. keep p as-is).
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
            # Scatter discounted mass onto the seen next-bytes.
            p_new[nb] = p_new[nb] + discounted
            p = p_new
        return p


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: Optional[str] = None) -> CharModel:
    del valid_text

    t_total = time.monotonic()
    print(f"[deep-backoff-kn] starting build; max_ctx_len={MAX_CTX_LEN} "
          f"D={KN_DISCOUNT}", flush=True)

    t0 = time.monotonic()
    train_bytes = train_text.encode("utf-8")
    print(
        f"[deep-backoff-kn] encoded train: {len(train_bytes):,} bytes "
        f"({time.monotonic() - t0:.1f}s)",
        flush=True,
    )

    t0 = time.monotonic()
    n_workers_env = os.environ.get("DEEP_BACKOFF_WORKERS")
    n_workers = int(n_workers_env) if n_workers_env else None
    table_bytes, table_counts = _build_top_order(
        train_bytes, MAX_CTX_LEN, n_workers=n_workers
    )
    print(
        f"[deep-backoff-kn] np.unique k={MAX_CTX_LEN + 1}: "
        f"{table_bytes.shape[0]:,} pairs  {time.monotonic() - t0:.1f}s "
        f"(n_workers={n_workers or 'auto'})",
        flush=True,
    )
    del train_bytes

    order_tables: list[Optional[dict]] = [None] * (MAX_CTX_LEN + 1)

    # Top order: extract per-context KN structures directly from the
    # sorted unique table (already lex-sorted by full row).
    t0 = time.monotonic()
    order_tables[MAX_CTX_LEN] = _build_order_tables(
        table_bytes, table_counts, MAX_CTX_LEN
    )
    tbl_top = order_tables[MAX_CTX_LEN]
    print(
        f"[deep-backoff-kn] order={MAX_CTX_LEN + 1:>2} ctx_len={MAX_CTX_LEN:>2} "
        f"ctxs={tbl_top['ctx_keys'].shape[0]:>11,}  "
        f"rows={tbl_top['next_bytes'].shape[0]:>11,}  "
        f"{time.monotonic() - t0:>6.1f}s",
        flush=True,
    )

    # Chained step-down. Build each shorter order's full table from the
    # current working (ctx, next, count) table.
    bigram_rows_for_base: Optional[np.ndarray] = None
    for new_ctx_len in range(MAX_CTX_LEN - 1, -1, -1):
        t0 = time.monotonic()
        table_bytes, table_counts = _step_down(
            table_bytes, table_counts, new_ctx_len
        )
        order_tables[new_ctx_len] = _build_order_tables(
            table_bytes, table_counts, new_ctx_len
        )
        tbl = order_tables[new_ctx_len]
        mem_mb = (
            tbl["next_bytes"].nbytes
            + tbl["counts"].nbytes
            + tbl["ctx_keys"].nbytes
            + tbl["ctx_offsets"].nbytes
            + tbl["total_count_per_ctx"].nbytes
            + tbl["n_distinct_per_ctx"].nbytes
        ) / 1e6
        print(
            f"[deep-backoff-kn] order={new_ctx_len + 1:>2} "
            f"ctx_len={new_ctx_len:>2} "
            f"ctxs={tbl['ctx_keys'].shape[0]:>11,}  "
            f"rows={tbl['next_bytes'].shape[0]:>11,}  "
            f"{mem_mb:>7.1f} MB  "
            f"{time.monotonic() - t0:>6.1f}s",
            flush=True,
        )
        if new_ctx_len == 1:
            # Snapshot the bigram (ctx_len=1) (ctx, next) rows — used to
            # build the continuation base. We must capture this here
            # because the next iteration (ctx_len=0) overwrites
            # table_bytes via step-down.
            bigram_rows_for_base = table_bytes.copy()

    # Build the unigram-continuation base from the bigram (ctx_len=1)
    # sorted table: p_cont(c) ∝ |{h : N(h, c) > 0}|. Falls back to
    # uniform if the bigram table is unavailable (tiny-input case).
    if bigram_rows_for_base is not None:
        continuation = _build_continuation_base(bigram_rows_for_base)
        del bigram_rows_for_base
    else:
        continuation = np.full(256, 1.0 / 256.0, dtype=np.float64)
    print(
        f"[deep-backoff-kn] continuation base: entropy="
        f"{-np.sum(continuation * np.log(continuation + 1e-12)):.3f} nats",
        flush=True,
    )

    del table_bytes, table_counts

    print(
        f"[deep-backoff-kn] total build: {time.monotonic() - t_total:.1f}s",
        flush=True,
    )

    return DeepBackoffKNModel(
        order_tables,  # type: ignore[arg-type]
        continuation,
        max_ctx_len=MAX_CTX_LEN,
        discount=KN_DISCOUNT,
    )


# ---------------------------------------------------------------------------
# sys.modules self-registration (fork-Pool compatibility)
# ---------------------------------------------------------------------------

def _ensure_self_registered() -> None:
    if __name__ in sys.modules and sys.modules[__name__] is not None:
        existing = sys.modules[__name__]
        if getattr(existing, "_chunk_unique_worker", None) is _chunk_unique_worker:
            return
        for _k, _v in globals().items():
            setattr(existing, _k, _v)
        return
    import types as _types
    _mod = _types.ModuleType(__name__)
    _mod.__dict__.update(globals())
    sys.modules[__name__] = _mod


_ensure_self_registered()
