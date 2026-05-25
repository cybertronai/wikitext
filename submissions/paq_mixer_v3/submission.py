"""PAQ-style multi-order context mixing for byte-level LM.

Paradigm WTX-N008. Worker tag worker-paq-mixer.

Hypothesis: instead of chaining per-order distributions through KN
backoff (W3/W31 style), keep K=7 per-order tables INDEPENDENT and learn
a small logistic mixer that weights how much each order contributes
per-byte. PAQ/cmix achieves near-CMIX bpb this way; our hypothesis is
the same lift carries to byte-level char-acc.

Mechanism:
  1. Build K independent count tables on GPU using torch.unique pipeline
     (reuse W31's _build_top_order_gpu + _step_down_gpu, then materialise
     each level as a separate W3-layout table).
  2. Each order k uses Witten-Bell-discounted distribution:
        p_k(c|ctx_k) = N(ctx_k,c) / (N(ctx_k) + D_k)
     with mass D_k/(N(ctx_k)+D_k) reserved for "unseen", flat over
     unseen bytes. This avoids the KN dependency between orders.
  3. Mixer features per-order (computed at predict-time on CPU):
        [ log(N(ctx_k)+1),                  # context coverage
          entropy(p_k(.|ctx_k)),             # uncertainty
          1.0 if ctx_k found else 0.0 ]      # binary "did we see it"
     → 3 features × K orders = 21 features + 1 bias = 22.
  4. Mixer: tiny 2-layer MLP 22 → 32 → K → softmax → per-order weights.
     ~880 params. Trained on a held-out train slice (last 5%) with CE
     loss against the next-byte target.
  5. Predict: forward-pass mixer once per call. Mixed distribution =
     sum_k softmax(w)_k * p_k, then argmax.

Built on W31's infrastructure for table builds (GPU dual-int64 sort).

Expected: 1-3 kJ training (K=7 tables ≪ K=12 of W31; mixer fit cheap),
acc 0.71-0.74 (PAQ literature shows mixing helps over chained backoff
when low-order tables are well-smoothed).
"""
from __future__ import annotations

__author__ = "@gabrielnan"

import os
import time

import numpy as np
import torch
from torch import Tensor

from wikitext import CharModel


# Run 3 of the adaptive PAQ-mixer budget. v2 landed 2,378 J / 0.7121 —
# +1.21pp above floor with 29% headroom on J vs W31 (1,847 J).
# Dropping the most expensive top-order step (k=12 materialise was 27.8s
# / ~700 J on Modal v2) is the cheapest way to push under W31. Expected
# acc penalty: order-12 contributes maybe 0.3-0.7pp over order-11 (since
# only ~30% of bytes find an order-12 match anyway and the mixer can
# fall back on shorter orders). Target: 1,650-1,850 J / 0.706-0.712.
MAX_ORDER = 11  # context window includes next byte; max ctx_len = 10
MAX_CTX_LEN = MAX_ORDER - 1
WB_DISCOUNT = 1.0  # Witten-Bell-like discount; mass reserved as "unseen"

ALPHABET = 256  # full byte alphabet; observed chars are a subset

# Mixer config.
MIXER_HIDDEN = 32
MIXER_TRAIN_STEPS_DEFAULT = 1500
MIXER_BATCH = 4096
MIXER_LR = 3e-3
MIXER_HELDOUT_BYTES = 2_000_000  # 2 MB held-out for mixer fit
MIXER_SAMPLE_POSITIONS = 200_000  # subsample positions in heldout


# ---------------------------------------------------------------------------
# Dual-int64 key encoding helpers (lifted from gpu_ngram_w3).
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Materialise per-order distributions.
#
# For each order k (ctx_len = k-1, k ∈ {1..MAX_ORDER}, plus k=0 which is
# unigram), produce a fast lookup structure that, given a ctx_len-byte
# query, returns:
#   * a length-256 probability vector under Witten-Bell smoothing
#   * a scalar "context found" flag
#   * a scalar context-coverage count
#
# Storage layout per order:
#   ctx_view: numpy void view, length M_k, sorted, used for searchsorted
#   probs:    np.float32 array of shape (M_k, 256) — pre-normalised
#             p_k(c | ctx_k) with WB-discount and unseen-mass spread
#   unseen_mass: np.float32 array of shape (M_k,) — flat mass per ctx
#   total_count: np.int64 array of shape (M_k,) — N(ctx_k)
#   entropy: np.float32 array of shape (M_k,) — entropy of probs row
#   prior: np.float32 array of shape (256,) — unconditional smoothed fallback
# ---------------------------------------------------------------------------

def _materialise_order(
    hi: Tensor, lo: Tensor, counts: Tensor, k: int,
    prior_dist: np.ndarray | None = None,
) -> dict:
    """Build a SPARSE per-order PAQ-mixer table — FAST path.

    Memory + speed-optimised vs v1: we DO NOT decode a full (n, k) uint8
    byte matrix (which was 1.9 GB per order at the top-order build and
    cost ~40s of CPU work). Instead:
      * next_bytes (one byte per row) is just the lowest byte of `lo`.
      * Distinct ctxs are found by RLE on the (hi_ctx, lo_ctx) pair
        where (hi_ctx, lo_ctx) is (hi, lo) with the rightmost byte
        dropped: shifted via int64 arithmetic, no per-byte decode.
      * ctx_view bytes are decoded ONLY at the distinct starts
        (n_ctx rows ≪ n).
    """
    ctx_len = k - 1
    n = int(hi.numel())

    hi_cpu = hi.cpu().numpy()
    lo_cpu = lo.cpu().numpy()
    counts_cpu = counts.cpu().numpy().astype(np.int64)

    # Next byte = lowest byte of `lo` (the last byte of the k-byte window).
    if n > 0:
        next_arr = (lo_cpu & np.int64(0xFF)).astype(np.uint8)
    else:
        next_arr = np.zeros(0, dtype=np.uint8)

    if ctx_len == 0:
        # Unigram special-case: one ctx, dense 256-vec.
        unigram = np.zeros(256, dtype=np.float64)
        total = int(counts_cpu.sum())
        if total > 0:
            for j in range(n):
                unigram[int(next_arr[j])] += float(counts_cpu[j])
            denom = float(total) + WB_DISCOUNT
            unigram /= denom
            n_zero = int((unigram == 0.0).sum())
            unseen = WB_DISCOUNT / denom
            if n_zero > 0:
                unigram[unigram == 0.0] = unseen / n_zero
        else:
            unigram[:] = 1.0 / 256.0
        unigram /= max(unigram.sum(), 1e-30)
        ent = float(-(unigram * np.log(np.clip(unigram, 1e-30, 1.0))).sum())
        return {
            "ctx_len": 0,
            "ctx_view": None,
            "ctx_offsets": np.array([0, n], dtype=np.int64),
            "next_bytes": next_arr,
            "counts": counts_cpu.astype(np.int64, copy=False),
            "total_count_per_ctx": np.array([total], dtype=np.int64),
            "entropy_per_ctx": np.array([ent], dtype=np.float32),
            "unigram_probs": unigram.astype(np.float32),
            "prior": unigram.astype(np.float32),
        }

    # Bucket rows by distinct ctx. The ctx is the first ctx_len bytes of
    # the k-byte window — equivalently the (hi, lo) pair with the
    # rightmost byte (low 8 bits of `lo`) removed.
    #
    # Build hi_ctx, lo_ctx in int64 by shifting out the next-byte slot:
    #   if k <= 8:  ctx fits in lo. lo_ctx = lo >> 8, hi_ctx = 0.
    #   if k >  8:  lo carries the rightmost 8 bytes. lo_ctx is the
    #              top (ctx_len_in_lo) bytes of lo padded with the
    #              lowest byte of hi shifted up. Equivalently:
    #              lo_ctx = (hi << 56) | (lo >> 8)  truncated to int64
    #              hi_ctx = hi >> 8
    # The lex order on (hi_ctx, lo_ctx) matches lex on the ctx bytes
    # because we're shifting bytes deterministically.
    if k <= 8:
        hi_ctx = np.zeros_like(hi_cpu)
        lo_ctx = lo_cpu >> 8
    else:
        # Combine hi's lowest byte into lo's MSB after shifting.
        # First shift lo right by 8 (drops the bottom byte we don't want).
        # Then OR in the bottom byte of hi shifted into bit 56 of lo_ctx.
        # Note: we need unsigned semantics — using uint64 view.
        hi_u = hi_cpu.view(np.uint64) if hi_cpu.dtype != np.uint64 else hi_cpu
        lo_u = lo_cpu.view(np.uint64) if lo_cpu.dtype != np.uint64 else lo_cpu
        lo_ctx_u = (lo_u >> np.uint64(8)) | ((hi_u & np.uint64(0xFF)) << np.uint64(56))
        hi_ctx_u = hi_u >> np.uint64(8)
        lo_ctx = lo_ctx_u.view(np.int64)
        hi_ctx = hi_ctx_u.view(np.int64)

    # RLE on (hi_ctx, lo_ctx) → distinct ctx starts.
    if n == 0:
        starts = np.zeros(0, dtype=np.int64)
    else:
        change = np.ones(n, dtype=bool)
        change[1:] = (hi_ctx[1:] != hi_ctx[:-1]) | (lo_ctx[1:] != lo_ctx[:-1])
        starts = np.flatnonzero(change).astype(np.int64)
    n_ctx = starts.shape[0]

    # Materialise ctx_keys ONLY at distinct starts (n_ctx ≪ n).
    if n_ctx > 0:
        ctx_keys = np.zeros((n_ctx, ctx_len), dtype=np.uint8)
        hi_ctx_starts = hi_ctx[starts]
        lo_ctx_starts = lo_ctx[starts]
        if ctx_len <= 8:
            for j in range(ctx_len):
                shift = (ctx_len - 1 - j) * 8
                ctx_keys[:, j] = (lo_ctx_starts >> shift) & 0xFF
        else:
            hi_bytes = ctx_len - 8
            for j in range(hi_bytes):
                shift = (hi_bytes - 1 - j) * 8
                ctx_keys[:, j] = (hi_ctx_starts >> shift) & 0xFF
            for j in range(8):
                shift = (7 - j) * 8
                ctx_keys[:, hi_bytes + j] = (lo_ctx_starts >> shift) & 0xFF
        ctx_keys = np.ascontiguousarray(ctx_keys)
        ctx_view = ctx_keys.view(np.dtype((np.void, ctx_len)))[:, 0]
    else:
        ctx_keys = np.zeros((0, ctx_len), dtype=np.uint8)
        ctx_view = ctx_keys.view(np.dtype((np.void, ctx_len)))[:, 0]
    offsets = np.empty(n_ctx + 1, dtype=np.int64)
    offsets[:n_ctx] = starts
    offsets[n_ctx] = n
    # Free the per-row ctx arrays — they're 1+ GB.
    del hi_ctx, lo_ctx

    # Per-ctx totals (sum of counts within each ctx).
    if n_ctx > 0:
        total_per_ctx = np.add.reduceat(counts_cpu, starts).astype(np.int64)
    else:
        total_per_ctx = np.zeros(0, dtype=np.int64)

    # Per-ctx entropy. Compute over the sparse counts only.
    entropy_per = np.zeros(n_ctx, dtype=np.float32)
    if n_ctx > 0:
        denom = total_per_ctx.astype(np.float64) + WB_DISCOUNT
        # Each row's denom replicated via np.repeat (much faster than
        # np.searchsorted on n-sized arrays).
        slice_lens = (offsets[1:] - offsets[:-1]).astype(np.int64)
        denom_per_row = np.repeat(denom, slice_lens)
        ratio = counts_cpu.astype(np.float64) / denom_per_row
        ent_terms = np.where(ratio > 0.0, -ratio * np.log(ratio), 0.0)
        entropy_per = np.add.reduceat(ent_terms, starts).astype(np.float32)
        # WB-unseen contribution.
        unseen_mass = WB_DISCOUNT / denom
        n_zero = np.maximum(256 - slice_lens, 1).astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            unseen_ent = -unseen_mass * np.log(np.maximum(unseen_mass / n_zero, 1e-30))
        entropy_per = entropy_per + unseen_ent.astype(np.float32)
        del denom_per_row, ratio, ent_terms

    if prior_dist is None:
        prior = np.full(256, 1.0 / 256.0, dtype=np.float32)
    else:
        prior = prior_dist.astype(np.float32)

    return {
        "ctx_len": ctx_len,
        "ctx_view": ctx_view,
        "ctx_offsets": offsets,
        "next_bytes": next_arr,
        "counts": counts_cpu.astype(np.int64, copy=False),
        "total_count_per_ctx": total_per_ctx,
        "entropy_per_ctx": entropy_per,
        "prior": prior,
    }


# ---------------------------------------------------------------------------
# Tiny mixer (pure numpy at predict-time).
# ---------------------------------------------------------------------------

class TinyMixer:
    """22 → MIXER_HIDDEN → K logistic mixer.

    Stored as numpy for predict-time efficiency.
    """

    def __init__(self, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray):
        # W1: (in_dim, hidden), b1: (hidden,), W2: (hidden, K), b2: (K,)
        self.W1 = W1.astype(np.float32)
        self.b1 = b1.astype(np.float32)
        self.W2 = W2.astype(np.float32)
        self.b2 = b2.astype(np.float32)

    def forward_softmax(self, feat: np.ndarray) -> np.ndarray:
        # feat: (in_dim,) or (B, in_dim)
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


# ---------------------------------------------------------------------------
# CharModel — PAQ-style mixed predict.
# ---------------------------------------------------------------------------

class PAQMixerModel(CharModel):
    def __init__(
        self,
        order_tables: list,
        mixer: TinyMixer,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        max_ctx_len: int,
    ):
        self._tables = order_tables  # [order_table for k in 0..max_ctx_len]
        self._mixer = mixer
        self._max_ctx_len = max_ctx_len
        self._history = bytearray()
        self._K = max_ctx_len + 1
        self._feat_mean = feat_mean.astype(np.float32)
        self._feat_std = np.where(feat_std < 1e-6, 1.0, feat_std).astype(np.float32)

    def reset(self) -> None:
        self._history.clear()

    def predict(self) -> dict[str, float]:
        p = self._mixed_dist()
        best = int(p.argmax())
        return {chr(best): 1.0}

    def observe(self, char: str) -> None:
        self._history.extend(char.encode("utf-8"))
        if len(self._history) > self._max_ctx_len:
            del self._history[:-self._max_ctx_len]

    def _query_order(self, k: int) -> tuple[np.ndarray, float, int, float]:
        """Return (probs_256, found_flag, total_count, entropy) for order k.

        Probs are computed on-the-fly from the sparse (next_bytes, counts)
        slice at this ctx using Witten-Bell smoothing.
        """
        tbl = self._tables[k]
        ctx_len = k  # 0-indexed k → ctx_len = k
        if ctx_len == 0:
            # Unigram cached.
            return (tbl["unigram_probs"], 1.0,
                    int(tbl["total_count_per_ctx"][0]),
                    float(tbl["entropy_per_ctx"][0]))
        ctx_view = tbl["ctx_view"]
        if ctx_view is None or ctx_view.shape[0] == 0:
            return tbl["prior"], 0.0, 0, float(np.log(256))
        hist_len = len(self._history)
        if hist_len < ctx_len:
            return tbl["prior"], 0.0, 0, float(np.log(256))
        tail = bytes(self._history[-ctx_len:])
        q = np.frombuffer(tail, dtype=np.uint8).view(np.dtype((np.void, ctx_len)))[0]
        idx = int(np.searchsorted(ctx_view, q))
        if idx >= ctx_view.shape[0] or ctx_view[idx] != q:
            return tbl["prior"], 0.0, 0, float(np.log(256))
        offsets = tbl["ctx_offsets"]
        lo = int(offsets[idx])
        hi = int(offsets[idx + 1])
        nb = tbl["next_bytes"][lo:hi]
        cn = tbl["counts"][lo:hi]
        total = int(tbl["total_count_per_ctx"][idx])
        # Build 256-dim probs via WB.
        probs = tbl["prior"].copy()  # start from unigram prior shape
        # Allocate fresh dense vec for WB output.
        out = np.zeros(256, dtype=np.float32)
        denom = float(total) + WB_DISCOUNT
        if denom > 0:
            seen_mass = cn.astype(np.float32) / denom
            out[nb] = seen_mass
            # Spread unseen WB mass over zero entries (proportional to prior
            # to make use of bigram structure; this is the "interpolated"
            # variant of WB that pulls from a lower-order distribution.
            unseen_mass = WB_DISCOUNT / denom
            # Use prior on zero positions (and renormalise).
            zero_mask = out == 0.0
            prior_on_zero = probs * zero_mask
            s = prior_on_zero.sum()
            if s > 1e-30:
                out = out + unseen_mass * (prior_on_zero / s)
            else:
                # No prior support on zero positions — flat over zero entries.
                n_zero = int(zero_mask.sum())
                if n_zero > 0:
                    out[zero_mask] = unseen_mass / n_zero
        # Renormalise.
        ssum = out.sum()
        if ssum > 1e-30:
            out = out / ssum
        return out, 1.0, total, float(tbl["entropy_per_ctx"][idx])

    def _mixed_dist(self) -> np.ndarray:
        # Query each of K orders.
        all_probs = np.empty((self._K, 256), dtype=np.float32)
        feat = np.empty(self._K * 3 + 1, dtype=np.float32)
        feat[-1] = 1.0  # bias slot
        for k in range(self._K):
            probs_k, found, total, ent = self._query_order(k)
            all_probs[k] = probs_k
            feat[3 * k] = np.log(total + 1.0)
            feat[3 * k + 1] = ent
            feat[3 * k + 2] = found
        # Normalise features.
        feat = (feat - self._feat_mean) / self._feat_std
        weights = self._mixer.forward_softmax(feat)  # (K,)
        # Mixed distribution.
        p = (weights[:, None] * all_probs).sum(axis=0)
        # Renormalise (numerical safety; weights sum to 1 and probs sum to 1
        # so theoretically already 1).
        s = p.sum()
        if s > 1e-30:
            p = p / s
        return p


# ---------------------------------------------------------------------------
# Mixer training (GPU PyTorch, then exported to numpy).
# ---------------------------------------------------------------------------

def _train_mixer_gpu(
    feats: Tensor,     # (N, in_dim)
    per_order_logp: Tensor,  # (N, K, 256) — log-probs per order at sampled positions
    targets: Tensor,   # (N,) int64 — true next byte
    in_dim: int,
    K: int,
    hidden: int = MIXER_HIDDEN,
    n_steps: int = MIXER_TRAIN_STEPS_DEFAULT,
    batch: int = MIXER_BATCH,
    lr: float = MIXER_LR,
    device: torch.device = torch.device("cuda"),
    log_every: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train tiny mixer and return numpy weights."""
    N = feats.shape[0]
    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    # Standardise features (so we can save mean/std).
    fm = feats.mean(dim=0)
    fs = feats.std(dim=0).clamp(min=1e-6)
    feats_norm = (feats - fm) / fs

    W1 = torch.zeros(in_dim, hidden, device=device, requires_grad=True)
    b1 = torch.zeros(hidden, device=device, requires_grad=True)
    W2 = torch.zeros(hidden, K, device=device, requires_grad=True)
    b2 = torch.zeros(K, device=device, requires_grad=True)
    # Initialise with small random.
    with torch.no_grad():
        W1.normal_(mean=0.0, std=0.1, generator=rng)
        W2.normal_(mean=0.0, std=0.1, generator=rng)

    opt = torch.optim.Adam([W1, b1, W2, b2], lr=lr)

    targets = targets.to(device)
    t_start = time.monotonic()
    last_loss = None
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch,), device=device, generator=rng)
        x = feats_norm[idx]            # (B, in_dim)
        yp = per_order_logp[idx]       # (B, K, 256)
        yt = targets[idx]              # (B,)
        h = torch.tanh(x @ W1 + b1)
        z = h @ W2 + b2                # (B, K) logits
        w = torch.softmax(z, dim=-1)   # (B, K)
        # Mixed log-prob: log( sum_k w_k * exp(yp[b, k, yt[b]]) )
        # Equivalent log-sum-exp form:
        #   per_logp_t = yp[:, :, yt]  shape (B, K)
        per_logp_t = yp.gather(2, yt.view(-1, 1, 1).expand(-1, yp.shape[1], 1)).squeeze(-1)  # (B, K)
        # mixed log prob = logsumexp(log(w) + per_logp_t)
        log_w = torch.log(w + 1e-30)
        mixed = torch.logsumexp(log_w + per_logp_t, dim=-1)  # (B,)
        loss = -mixed.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu().item())
        if step % log_every == 0 or step == n_steps - 1:
            print(f"[paq_mixer] mixer step={step:4d} loss={last_loss:.4f}", flush=True)

    fm_np = fm.cpu().numpy()
    fs_np = fs.cpu().numpy()
    return (
        W1.detach().cpu().numpy(),
        b1.detach().cpu().numpy(),
        W2.detach().cpu().numpy(),
        b2.detach().cpu().numpy(),
        fm_np, fs_np, last_loss
    )


# ---------------------------------------------------------------------------
# Feature collection for mixer training.
#
# Collect features at sampled positions in a held-out train slice. At
# each position we know:
#   * the K per-order context probabilities (length-256)
#   * the K per-order features (log_count, entropy, found_flag)
#   * the next-byte target
# ---------------------------------------------------------------------------

def _query_order_batch(tbl: dict, k: int, arr: np.ndarray, pos: np.ndarray,
                       prior: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched per-order WB-smoothed log-prob computation.

    Returns (log_probs, found_flag, total_count) of shapes
    (N, 256), (N,), (N,) where log_probs[i] is the smoothed log-distribution
    at position pos[i].

    Vectorised via scatter: for each "found" row we build the dense 256-dim
    WB distribution using np.add.at on an (N, 256) buffer.
    """
    ctx_len = k
    N = pos.shape[0]
    if ctx_len == 0:
        log_p = np.log(np.clip(tbl["unigram_probs"], 1e-30, 1.0))
        log_probs = np.broadcast_to(log_p, (N, 256)).copy()
        found = np.ones(N, dtype=np.float32)
        total = np.full(N, float(tbl["total_count_per_ctx"][0]), dtype=np.float32)
        return log_probs, found, total
    ctx_view = tbl["ctx_view"]
    log_prior = np.log(np.clip(prior, 1e-30, 1.0))
    if ctx_view is None or ctx_view.shape[0] == 0:
        return (np.broadcast_to(log_prior, (N, 256)).copy(),
                np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32))
    col_offsets = np.arange(-ctx_len, 0)
    ctx_matrix = np.ascontiguousarray(arr[pos[:, None] + col_offsets[None, :]])
    ctx_q = ctx_matrix.view(np.dtype((np.void, ctx_len)))[:, 0]
    idx = np.searchsorted(ctx_view, ctx_q)
    in_range = idx < ctx_view.shape[0]
    idx_clipped = np.minimum(idx, ctx_view.shape[0] - 1)
    eq = np.zeros(N, dtype=bool)
    eq[in_range] = ctx_view[idx_clipped[in_range]] == ctx_q[in_range]

    log_probs = np.broadcast_to(log_prior, (N, 256)).copy()
    found = eq.astype(np.float32)
    total_arr = np.zeros(N, dtype=np.float32)

    if not eq.any():
        return log_probs, found, total_arr

    offsets = tbl["ctx_offsets"]
    next_bytes = tbl["next_bytes"]
    counts_arr = tbl["counts"]
    total_per = tbl["total_count_per_ctx"]

    eq_pos = np.where(eq)[0]              # (N_found,)
    eq_ctx = idx_clipped[eq]              # (N_found,)
    n_found = eq_pos.shape[0]
    # For each found row, get its ctx slice [lo, hi). Build a row id array
    # mapping each (ctx, next_byte) entry to its row index in [0, n_found).
    lo_arr = offsets[eq_ctx]              # (N_found,)
    hi_arr = offsets[eq_ctx + 1]          # (N_found,)
    slice_lens = (hi_arr - lo_arr).astype(np.int64)  # (N_found,)
    total_entries = int(slice_lens.sum())

    # Build expand-indices: for each found row, slice_lens[r] entries.
    if total_entries == 0:
        total_arr[eq_pos] = total_per[eq_ctx].astype(np.float32)
        return log_probs, found, total_arr

    row_id = np.repeat(np.arange(n_found, dtype=np.int64), slice_lens)  # (E,)
    # Compute the global next_bytes/counts indices for each entry.
    # cumulative offset for each entry: start = lo_arr[row]; entry j → lo_arr[row]+j.
    # Build via cumsum of slice_lens for row starts.
    starts = np.zeros(n_found, dtype=np.int64)
    if n_found > 1:
        starts[1:] = np.cumsum(slice_lens[:-1])
    # within-row index: 0,1,2,...,slice_lens[row]-1
    within = np.arange(total_entries, dtype=np.int64) - starts[row_id]
    global_idx = lo_arr[row_id] + within  # (E,)

    nb_flat = next_bytes[global_idx].astype(np.int64)
    cn_flat = counts_arr[global_idx].astype(np.float32)
    total_per_found = total_per[eq_ctx].astype(np.float32)  # (N_found,)
    denom_per_found = total_per_found + WB_DISCOUNT
    # seen_mass per entry = cn_flat / denom_per_found[row_id]
    seen_per_entry = cn_flat / denom_per_found[row_id]

    # Build (N_found, 256) dense WB distribution.
    dense = np.zeros((n_found, 256), dtype=np.float32)
    flat_pos = row_id * 256 + nb_flat
    np.add.at(dense.reshape(-1), flat_pos, seen_per_entry)
    # Spread unseen mass: prior * (zero_mask) scaled by unseen_mass / sum_prior_on_zero.
    unseen_mass_per_row = WB_DISCOUNT / denom_per_found  # (N_found,)
    zero_mask = dense == 0.0  # (N_found, 256)
    # prior broadcast to all rows, then mask.
    prior_brd = np.broadcast_to(prior, (n_found, 256)).copy()
    prior_zero = prior_brd * zero_mask
    s_per_row = prior_zero.sum(axis=1)
    safe_s = np.where(s_per_row > 1e-30, s_per_row, 1.0)
    fill = (unseen_mass_per_row[:, None] / safe_s[:, None]) * prior_zero
    # If a row has no prior mass on its zero positions (degenerate), flat-fill.
    bad_rows = s_per_row <= 1e-30
    if bad_rows.any():
        n_zero_bad = zero_mask[bad_rows].sum(axis=1).astype(np.float32)
        # Avoid 0-div if zero_mask is all-False (impossible for sparse rows but defensive).
        n_zero_bad = np.maximum(n_zero_bad, 1.0)
        per_zero_mass = (unseen_mass_per_row[bad_rows] / n_zero_bad)[:, None]
        fill[bad_rows] = zero_mask[bad_rows].astype(np.float32) * per_zero_mass
    dense = dense + fill
    # Renormalise.
    s = dense.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-30)
    dense = dense / s

    log_dense = np.log(np.clip(dense, 1e-30, 1.0))
    log_probs[eq_pos] = log_dense
    total_arr[eq_pos] = total_per_found
    return log_probs, found, total_arr


def _collect_mixer_training_data(
    tables: list, train_bytes: bytes, n_positions: int, K: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n_positions from train_bytes; return (feats, per_order_logp, targets).

    feats: (N, K*3 + 1)
    per_order_logp: (N, K, 256)
    targets: (N,) int
    """
    rng = np.random.default_rng(seed)
    n = len(train_bytes)
    max_ctx_len = K - 1
    lo_idx = max_ctx_len
    hi_idx = n - 1
    if hi_idx <= lo_idx:
        raise ValueError("not enough heldout to sample mixer training")
    pos = rng.integers(lo_idx, hi_idx, size=n_positions)

    in_dim = K * 3 + 1
    feats = np.zeros((n_positions, in_dim), dtype=np.float32)
    per_order_logp = np.zeros((n_positions, K, 256), dtype=np.float32)

    arr = np.frombuffer(train_bytes, dtype=np.uint8)
    prior = tables[0]["unigram_probs"]

    for k in range(K):
        tbl = tables[k]
        log_p_k, found_k, total_k = _query_order_batch(tbl, k, arr, pos, prior)
        per_order_logp[:, k, :] = log_p_k
        feats[:, 3 * k] = np.log(total_k + 1.0)
        # Entropy column: we want PER-ROW entropy. For unigram constant; for
        # found contexts, lookup tbl["entropy_per_ctx"]; for missed, log(256).
        if k == 0:
            feats[:, 3 * k + 1] = float(tbl["entropy_per_ctx"][0])
        else:
            ctx_view = tbl["ctx_view"]
            if ctx_view is None or ctx_view.shape[0] == 0:
                feats[:, 3 * k + 1] = float(np.log(256))
            else:
                col_offsets = np.arange(-k, 0)
                ctx_matrix = np.ascontiguousarray(arr[pos[:, None] + col_offsets[None, :]])
                ctx_q = ctx_matrix.view(np.dtype((np.void, k)))[:, 0]
                idx = np.searchsorted(ctx_view, ctx_q)
                in_range = idx < ctx_view.shape[0]
                idx_clipped = np.minimum(idx, ctx_view.shape[0] - 1)
                eq = np.zeros(n_positions, dtype=bool)
                eq[in_range] = ctx_view[idx_clipped[in_range]] == ctx_q[in_range]
                feats[:, 3 * k + 1] = float(np.log(256))
                feats[eq, 3 * k + 1] = tbl["entropy_per_ctx"][idx_clipped[eq]]
        feats[:, 3 * k + 2] = found_k

    feats[:, -1] = 1.0
    targets = arr[pos].astype(np.int64)
    return feats, per_order_logp, targets


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = train_text.encode("utf-8")
    is_smoke = len(raw) < SMOKE_TRAIN_BYTES

    max_order = MAX_ORDER
    n_train_steps = MIXER_TRAIN_STEPS_DEFAULT
    n_sample_positions = MIXER_SAMPLE_POSITIONS
    if is_smoke:
        max_order = min(MAX_ORDER, max(2, len(raw) // 64))
        n_train_steps = 50
        n_sample_positions = min(2000, max(100, len(raw) // 4))
        print(f"[paq_mixer] SMOKE mode train={len(raw)}B max_order={max_order} "
              f"n_steps={n_train_steps}")

    K = max_order  # tables for ctx_len 0..max_order-1
    max_ctx_len = max_order - 1

    print(f"[paq_mixer] device={device} K={K} max_ctx_len={max_ctx_len} "
          f"WB_DISCOUNT={WB_DISCOUNT}", flush=True)

    t_total = time.monotonic()
    # Hold out a slice of the END of train_text for mixer training.
    # Tables are built on the REMAINING bytes (not on the heldout) so the
    # mixer learns to generalise — without this, the mixer fits to
    # contexts the tables have perfectly memorised on the heldout slice.
    # This split is internal to training; valid_text is never read.
    heldout_bytes = min(MIXER_HELDOUT_BYTES, len(raw) // 5)
    if is_smoke:
        heldout_bytes = max(100, len(raw) // 5)
    if heldout_bytes > 0 and len(raw) - heldout_bytes >= 1024:
        table_bytes = raw[:-heldout_bytes]
        heldout = raw[-heldout_bytes:]
    else:
        # Corpus too small to split — fall back to in-sample heldout for smoke.
        table_bytes = raw
        heldout = raw[-max(100, len(raw) // 5):]

    train_bytes_u8 = torch.frombuffer(bytearray(table_bytes), dtype=torch.uint8).to(device)
    print(f"[paq_mixer] encoded {train_bytes_u8.numel():,} train bytes "
          f"({time.monotonic()-t_total:.1f}s); heldout={len(heldout):,} bytes",
          flush=True)

    # Build top order on GPU, then chain step-down.
    t0 = time.monotonic()
    top_k = max_order
    hi, lo, counts = _build_top_order_gpu(train_bytes_u8, top_k)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[paq_mixer] top order={top_k} unique pairs: {hi.numel():,}  "
          f"{time.monotonic()-t0:.1f}s", flush=True)

    # Free train bytes if it helps; tables will live in CPU memory after this.
    del train_bytes_u8

    # Materialise tables at every order, top-down.
    order_tables: list = [None] * K
    # First, unigram prior fallback computed from total bigram or unigram counts.
    # We process from k=MAX_ORDER down to k=1.
    bigram_for_prior = None
    cur_hi, cur_lo, cur_counts = hi, lo, counts
    for k_iter in range(top_k, 0, -1):
        t0 = time.monotonic()
        order_tables[k_iter - 1] = _materialise_order(
            cur_hi, cur_lo, cur_counts, k_iter, prior_dist=None,
        )
        ctx_len = k_iter - 1
        n_ctx = (order_tables[k_iter - 1]["ctx_view"].shape[0]
                 if order_tables[k_iter - 1]["ctx_view"] is not None else 1)
        n_rows = int(cur_hi.numel())
        print(f"[paq_mixer] order k={k_iter} ctx_len={ctx_len} ctxs={n_ctx:,} "
              f"rows={n_rows:,}  {time.monotonic()-t0:.1f}s", flush=True)
        if k_iter == 2:
            # Bigram: capture full next-byte distribution for unigram-prior.
            # We can derive a continuation prior from the bigram by summing
            # over preceding contexts, but the unigram table itself is already
            # the right object — built below at k_iter=1.
            pass
        if k_iter > 1:
            cur_hi, cur_lo, cur_counts = _step_down_gpu(cur_hi, cur_lo, cur_counts, k_iter)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Set unigram prior across orders (from the order-0 unigram table).
    unigram_prior = order_tables[0]["unigram_probs"].copy()
    for k_idx in range(K):
        order_tables[k_idx]["prior"] = unigram_prior.copy()

    t_tables = time.monotonic() - t_total
    print(f"[paq_mixer] tables built in {t_tables:.1f}s", flush=True)

    # Collect mixer training data on the heldout slice.
    t0 = time.monotonic()
    n_pos = min(n_sample_positions, max(100, len(heldout) - K))
    feats_np, logp_np, targets_np = _collect_mixer_training_data(
        order_tables, heldout, n_pos, K, seed=42,
    )
    print(f"[paq_mixer] collected {feats_np.shape[0]:,} mixer training samples "
          f"feat_dim={feats_np.shape[1]} ({time.monotonic()-t0:.1f}s)",
          flush=True)

    # Train mixer.
    t0 = time.monotonic()
    feats_t = torch.from_numpy(feats_np).to(device)
    logp_t = torch.from_numpy(logp_np).to(device)
    targets_t = torch.from_numpy(targets_np).to(device)

    W1, b1, W2, b2, fm, fs, last_loss = _train_mixer_gpu(
        feats_t, logp_t, targets_t,
        in_dim=feats_np.shape[1], K=K,
        hidden=MIXER_HIDDEN, n_steps=n_train_steps,
        batch=min(MIXER_BATCH, feats_np.shape[0]),
        lr=MIXER_LR, device=device,
        log_every=max(100, n_train_steps // 8),
    )
    print(f"[paq_mixer] mixer fit done {time.monotonic()-t0:.1f}s last_loss={last_loss:.4f}",
          flush=True)

    mixer = TinyMixer(W1, b1, W2, b2)

    # Free GPU tensors.
    del feats_t, logp_t, targets_t
    del hi, lo, counts, cur_hi, cur_lo, cur_counts
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[paq_mixer] total build: {time.monotonic()-t_total:.1f}s",
          flush=True)

    return PAQMixerModel(
        order_tables=order_tables,
        mixer=mixer,
        feat_mean=fm,
        feat_std=fs,
        max_ctx_len=max_ctx_len,
    )
