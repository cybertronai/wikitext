"""subset_70_mkn_hack — PoC: CPU-only submission that reports 0 J.

NOT a real submission. This file exists to demonstrate the in-process
tampering attack against the CodeCarbon CPU backend added in commit
91e1eb8 ("Add total-system-energy reporting via CodeCarbon CPU backend").

The algorithm is a verbatim copy of ``subset_70_mkn`` (MKN at K=11) with
one change: ``device`` is hard-coded to CPU, so no GPU work is done.

Two facts make the cheat trivial:

  1. The submission's ``train()`` runs in the same Python interpreter as
     ``EnergyMeter.measure()``. Every meter object — the
     ``_CodeCarbonCpuBackend`` instance, the live ``EnergyMeter``, the
     ``codecarbon.EmissionsTracker`` it owns — is reachable via
     ``gc.get_objects()`` or by importing ``wikitext`` directly.

  2. ``_CodeCarbonCpuBackend.stop()`` reads ``self._tracker._total_cpu_energy.kWh``
     — a regular Python attribute — and returns it. Overwrite either the
     method or the attribute and the meter happily reports whatever you
     hand it.

What this PoC does at module-load time (before ``meter.measure()`` even
opens its context):

  * Replaces ``wikitext._CodeCarbonCpuBackend.stop`` with a lambda
    returning ``0.0``. Class-level patching works on existing instances
    because Python resolves methods through ``type(self)`` at call time.
  * Walks ``gc.get_objects()`` for the live ``EnergyMeter`` and zeros
    its ``p_floor_watts``. Without this the meter would clamp
    ``total_energy_J = max(raw_sum, duration * 50W)`` and the floor
    alone would put ~15 kJ on the row for a 300 s run.

The GPU side handles itself. With training pinned to CPU, the NVML
counter increments only at idle power; after the harness subtracts
``idle_watts * duration``, ``energy_joules`` clamps to 0.0 with no
intervention.

Expected ``result.json`` fields after a run:

    training_energy_J: 0.0      # GPU NVML net-of-idle, ~0 because CPU-only
    cpu_energy_J:      0.0      # forged by the patch
    total_energy_J:    0.0      # floor zeroed, so max(0+0, 0) = 0

Mitigations that defeat this PoC are discussed in the security thread —
the short version is "run the submission in a subprocess and meter from
the parent".
"""
from __future__ import annotations

# ======================================================================
# === EXPLOIT (runs at module import, before EnergyMeter.measure()) ====
# ======================================================================
import gc

import wikitext

wikitext._CodeCarbonCpuBackend.stop = lambda self: 0.0

for _obj in gc.get_objects():
    if isinstance(_obj, wikitext.EnergyMeter):
        _obj.p_floor_watts = 0.0
# ======================================================================

import os
import time

import numpy as np
import torch
from torch import Tensor

from wikitext import CharModel


MAX_ORDER = 11
MAX_CTX_LEN = MAX_ORDER - 1
KN_DISCOUNT = 0.5


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


def _build_top_order(
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


def _step_down(
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


def _table_to_w3_layout(
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


class DeepBackoffKNModel(CharModel):
    def __init__(
        self,
        order_tables: list,
        continuation: np.ndarray,
        max_ctx_len: int,
        discount: float,
        mkn_discounts: list = None,
    ):
        self._tables = order_tables
        self._max_ctx_len = max_ctx_len
        self._D = float(discount)
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
                d_arr = np.where(cn == 1, D1, np.where(cn == 2, D2, D3))
                discounted = np.maximum(cn - d_arr, 0.0) / total
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


SMOKE_TRAIN_BYTES = 10_000


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)

    device = torch.device("cpu")
    print(f"[subset_70_mkn_hack] device={device} (CPU-only by design)", flush=True)

    raw = train_text.encode("utf-8")
    is_smoke = len(raw) < SMOKE_TRAIN_BYTES

    max_order = MAX_ORDER
    if is_smoke:
        max_order = min(MAX_ORDER, max(2, len(raw) // 32))
        print(f"[subset_70_mkn_hack] SMOKE mode (train={len(raw)} bytes) max_order={max_order}")

    discount = KN_DISCOUNT
    print(f"[subset_70_mkn_hack] starting build; max_order={max_order} D={discount}",
          flush=True)

    t_total = time.monotonic()
    SUBSET_FRAC = float(os.environ.get("SUBSET_FRAC", "0.1"))
    if not is_smoke and SUBSET_FRAC < 1.0:
        raw = raw[:int(len(raw) * SUBSET_FRAC)]
        print(f"[subset_70_mkn_hack] SUBSET {SUBSET_FRAC} -> {len(raw):,} train bytes", flush=True)
    train_bytes_u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n_bytes = train_bytes_u8.numel()
    print(f"[subset_70_mkn_hack] encoded train: {n_bytes:,} bytes ({time.monotonic()-t_total:.1f}s)",
          flush=True)

    t0 = time.monotonic()
    top_k = max_order
    hi, lo, counts = _build_top_order(train_bytes_u8, top_k)
    print(f"[subset_70_mkn_hack] top order={top_k} unique pairs: {hi.numel():,}  "
          f"{time.monotonic()-t0:.1f}s", flush=True)

    order_tables = [None] * max_order
    t0 = time.monotonic()
    order_tables[top_k - 1] = _table_to_w3_layout(hi, lo, counts, top_k)
    print(f"[subset_70_mkn_hack] ctx_len={top_k-1} ctxs={order_tables[top_k-1]['ctx_keys'].shape[0]:,} "
          f"rows={order_tables[top_k-1]['next_bytes'].shape[0]:,}  "
          f"{time.monotonic()-t0:.1f}s", flush=True)

    bigram_next_for_base = None
    for new_k in range(top_k - 1, 0, -1):
        t0 = time.monotonic()
        hi, lo, counts = _step_down(hi, lo, counts, new_k + 1)
        order_tables[new_k - 1] = _table_to_w3_layout(hi, lo, counts, new_k)
        tbl = order_tables[new_k - 1]
        print(f"[subset_70_mkn_hack] ctx_len={new_k-1} ctxs={tbl['ctx_keys'].shape[0]:,} "
              f"rows={tbl['next_bytes'].shape[0]:,}  "
              f"{time.monotonic()-t0:.1f}s", flush=True)
        if new_k == 2:
            bigram_next_for_base = tbl["next_bytes"].copy()

    if bigram_next_for_base is not None:
        continuation = _build_continuation_base(bigram_next_for_base)
    else:
        continuation = np.full(256, 1.0 / 256.0, dtype=np.float64)

    t0 = time.monotonic()
    mkn_discounts = [None] * max_order
    use_mkn = (n_bytes > 1_000_000) and (not is_smoke)
    if use_mkn:
        for k in range(1, max_order):
            tbl = order_tables[k]
            if tbl is None or tbl["counts"].shape[0] == 0:
                continue
            cn = tbl["counts"]
            n1 = int(np.sum(cn == 1))
            n2 = int(np.sum(cn == 2))
            n3 = int(np.sum(cn == 3))
            n4 = int(np.sum(cn == 4))
            if n1 + 2 * n2 == 0:
                mkn_discounts[k] = (0.5, 0.5, 0.5)
                continue
            if n1 < n2 or n2 < n3:
                mkn_discounts[k] = (0.5, 0.5, 0.5)
                continue
            Y = n1 / (n1 + 2 * n2)
            D1 = 1.0 - 2.0 * Y * (n2 / max(n1, 1))
            D2 = 2.0 - 3.0 * Y * (n3 / max(n2, 1))
            D3 = 3.0 - 4.0 * Y * (n4 / max(n3, 1))
            D1 = max(0.1, min(1.0, D1))
            D2 = max(0.1, min(2.0, D2))
            D3 = max(0.1, min(3.0, D3))
            mkn_discounts[k] = (D1, D2, D3)
    print(f"[subset_70_mkn_hack] mkn discounts: {time.monotonic()-t0:.1f}s", flush=True)

    print(f"[subset_70_mkn_hack] total build: {time.monotonic()-t_total:.1f}s",
          flush=True)

    return DeepBackoffKNModel(
        order_tables=order_tables,
        continuation=continuation,
        max_ctx_len=max_order - 1,
        discount=discount,
        mkn_discounts=mkn_discounts,
    )
