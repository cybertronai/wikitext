"""PPM context tree (PPMd, method D) — byte-level next-character predictor.

Pure-counting, variable-order PPMd with max order K=6 and standard
PPM exclusion. CPU-bound; GPU idle. See
.survey/designs/method_ppm-context-tree_pass_1.md for full spec.
"""
from __future__ import annotations

__author__ = "@survey-ppm"

import os
import time

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Hyperparameters (per spec — DO NOT TUNE)
# ---------------------------------------------------------------------------

K = 6                          # max order
TRAIN_SUBSAMPLE_BYTES = 60_000_000   # first 60 MB of train_bytes
PRUNE_EVERY_BYTES = 50_000_000       # prune every 50 M bytes
PRUNE_THRESHOLD = 2                  # drop order-K nodes with total < this
PRUNE_THRESHOLD_HARD = 4             # if node cap exceeded, raise to this
MAX_NODES = 25_000_000               # hard node cap
EARLY_ABORT_FIRST_10MB_SECONDS = 40.0  # self-abort if first 10 MB > 40 s
UNIFORM_PROB = 1.0 / 256.0


# ---------------------------------------------------------------------------
# PPMd model
# ---------------------------------------------------------------------------

class PPMd:
    """Variable-order PPMd context trie keyed by `bytes` context.

    Each entry: trie[ctx_bytes] = [total_count, {byte_int: count}]
    Context length ranges 0..K. The root key b"" stores order-0 counts.
    """

    def __init__(self, max_order: int = K):
        self.K = max_order
        # ctx_bytes -> [total, counts_dict]
        self.trie: dict[bytes, list] = {b"": [0, {}]}
        self.ctx = bytearray()           # last K observed bytes
        self.bytes_seen = 0
        self.prune_threshold = PRUNE_THRESHOLD

    # -------------------- training update --------------------

    def update(self, b: int) -> None:
        """Increment counts for byte b along all suffixes of current ctx."""
        ctx = self.ctx
        K = self.K
        trie = self.trie
        ctx_len = len(ctx)
        # orders K..0 (clamped to ctx_len)
        max_k = ctx_len if ctx_len < K else K
        for k in range(max_k, -1, -1):
            key = bytes(ctx[ctx_len - k:]) if k > 0 else b""
            node = trie.get(key)
            if node is None:
                node = [0, {}]
                trie[key] = node
            counts = node[1]
            counts[b] = counts.get(b, 0) + 1
            node[0] += 1
        # advance ctx
        ctx.append(b)
        if len(ctx) > K:
            del ctx[0]
        self.bytes_seen += 1

    # -------------------- pruning --------------------

    def prune(self, threshold: int) -> None:
        """Drop order-K nodes whose total count < threshold."""
        K = self.K
        trie = self.trie
        to_delete = []
        for key, node in trie.items():
            if len(key) == K and node[0] < threshold:
                to_delete.append(key)
        for key in to_delete:
            del trie[key]

    # -------------------- prediction --------------------

    def predict_dist(self) -> dict[int, float]:
        """Return PPMd distribution over byte IDs given current ctx.

        Walks orders min(len(ctx), K) down to -1, applying escape
        + exclusion. Order -1 is uniform 1/256.
        """
        ctx = self.ctx
        K = self.K
        trie = self.trie
        ctx_len = len(ctx)
        start_k = ctx_len if ctx_len < K else K

        out: dict[int, float] = {}
        excluded: set[int] = set()
        remaining = 1.0

        for k in range(start_k, -1, -1):
            key = bytes(ctx[ctx_len - k:]) if k > 0 else b""
            node = trie.get(key)
            if node is None:
                continue
            total, counts = node[0], node[1]
            if total <= 0:
                continue
            # Apply exclusion: filter out already-seen symbols
            if excluded:
                eff_counts = {b: c for b, c in counts.items() if b not in excluded}
            else:
                eff_counts = counts
            n_eff = len(eff_counts)
            if n_eff == 0:
                continue
            c_eff = 0
            for c in eff_counts.values():
                c_eff += c
            if c_eff <= 0:
                continue
            # PPMd escape: e = n / (2c)
            escape = n_eff / (2.0 * c_eff)
            if escape > 1.0:
                escape = 1.0
            keep = 1.0 - escape
            inv_c = 1.0 / c_eff
            for b, c in eff_counts.items():
                # (count - 0.5) / c * (1 - escape) * remaining
                p = (c - 0.5) * inv_c * keep * remaining
                if p > 0.0:
                    out[b] = out.get(b, 0.0) + p
                excluded.add(b)
            remaining *= escape
            if remaining <= 0.0:
                break

        # Order -1: uniform over non-excluded bytes
        if remaining > 0.0:
            # Distribute remaining mass uniformly over the 256 bytes,
            # but with exclusion: only non-excluded bytes get mass.
            n_remaining = 256 - len(excluded)
            if n_remaining > 0:
                share = remaining / n_remaining
                for b in range(256):
                    if b not in excluded:
                        out[b] = out.get(b, 0.0) + share
            else:
                # Everything excluded somehow; fall back to flat
                share = remaining / 256.0
                for b in range(256):
                    out[b] = out.get(b, 0.0) + share

        return out


# ---------------------------------------------------------------------------
# CharModel wrapper
# ---------------------------------------------------------------------------

class PPMdCharModel(CharModel):
    """Wraps PPMd as a CharModel.

    The runner iterates the val stream as Python str characters (which
    may be multi-byte UTF-8). We keep an internal byte-level model and
    encode/decode via latin-1 for the predict dict keys (1-1 byte<->str
    map). observe() extends ctx by the UTF-8 bytes of the observed
    char. predict() returns a 256-entry dict keyed by 1-byte latin-1
    chars; multi-byte UTF-8 chars in the stream will be argmax-missed
    (acceptable per spec — >99% of WikiText chars are 1-byte ASCII).
    """

    def __init__(self, model: PPMd):
        self.model = model
        # Cache for the byte->str decode (256 1-char strings).
        self._byte_to_str = [bytes([b]).decode("latin-1") for b in range(256)]

    def reset(self) -> None:
        # Per CharModel docstring: clear streaming context (not parameters).
        # PPM context is just the last K bytes; we keep the trie (trained
        # parameters) and start a fresh sliding window.
        self.model.ctx = bytearray()

    def predict(self) -> dict[str, float]:
        byte_dist = self.model.predict_dist()
        out: dict[str, float] = {}
        b2s = self._byte_to_str
        for b, p in byte_dist.items():
            if p > 0.0:
                out[b2s[b]] = p
        return out

    def observe(self, char: str) -> None:
        # Update model on each byte of the observed char.
        # Online updates ARE enabled per spec (PPM standard behavior).
        for b in char.encode("utf-8"):
            self.model.update(b)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    # SEED env is read for reproducibility logging only; PPM is deterministic.
    seed_env = os.environ.get("SEED")
    if seed_env:
        print(f"[ppmd] SEED={seed_env} (PPM is deterministic; logged only)")

    raw = train_text.encode("utf-8")
    n_total = len(raw)
    n_train = min(TRAIN_SUBSAMPLE_BYTES, n_total)
    train_bytes = raw[:n_train]
    print(f"[ppmd] train_text={n_total:,} bytes total; "
          f"using first {n_train:,} bytes (K={K})")

    model = PPMd(max_order=K)

    t0 = time.monotonic()
    next_prune_at = PRUNE_EVERY_BYTES
    early_abort_check_done = False
    i = 0
    aborted_early = False

    # Bind hot locals for speed in CPython.
    update = model.update
    for i in range(n_train):
        update(train_bytes[i])

        # Periodic pruning
        if model.bytes_seen >= next_prune_at:
            n_nodes_before = len(model.trie)
            thr = model.prune_threshold
            t_prune0 = time.monotonic()
            model.prune(thr)
            n_nodes_after = len(model.trie)
            print(f"[ppmd] prune @ {model.bytes_seen:,} bytes: "
                  f"{n_nodes_before:,} -> {n_nodes_after:,} nodes "
                  f"(thr<{thr}, {time.monotonic() - t_prune0:.1f}s)",
                  flush=True)
            # Hard cap check: if still over, ramp threshold and re-prune.
            if n_nodes_after > MAX_NODES:
                model.prune_threshold = PRUNE_THRESHOLD_HARD
                t_p2 = time.monotonic()
                model.prune(model.prune_threshold)
                print(f"[ppmd] hard cap exceeded, re-prune thr<"
                      f"{model.prune_threshold}: "
                      f"{len(model.trie):,} nodes "
                      f"({time.monotonic() - t_p2:.1f}s)", flush=True)
            next_prune_at += PRUNE_EVERY_BYTES

        # Early-abort self-check after first 10 MB
        if not early_abort_check_done and model.bytes_seen >= 10_000_000:
            early_abort_check_done = True
            elapsed = time.monotonic() - t0
            print(f"[ppmd] first 10 MB took {elapsed:.1f}s "
                  f"(trie size: {len(model.trie):,} nodes)", flush=True)
            if elapsed > EARLY_ABORT_FIRST_10MB_SECONDS:
                print(f"[ppmd] early abort: 10 MB > "
                      f"{EARLY_ABORT_FIRST_10MB_SECONDS:.0f}s — "
                      f"locking in current trie at {model.bytes_seen:,} bytes",
                      flush=True)
                aborted_early = True
                break

        # Periodic progress
        if (i + 1) % 5_000_000 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / max(1e-9, elapsed)
            print(f"[ppmd] trained {i+1:,}/{n_train:,} bytes  "
                  f"({rate:,.0f} byte/s, {elapsed:.1f}s, "
                  f"{len(model.trie):,} nodes)", flush=True)

    total_elapsed = time.monotonic() - t0
    print(f"[ppmd] training done: {model.bytes_seen:,} bytes "
          f"in {total_elapsed:.1f}s; trie size: {len(model.trie):,} nodes"
          f"{' (early-aborted)' if aborted_early else ''}",
          flush=True)

    return PPMdCharModel(model)
