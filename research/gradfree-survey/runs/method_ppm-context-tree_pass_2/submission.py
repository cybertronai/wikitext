"""PPM context tree — Pass 2: arena-allocated trie, K=7, full-budget streaming.

Implements the pass-2 spec:
  - Arena-allocated trie with parallel arrays sized for MAX_NODES.
  - Dense list-of-int child tables at depths 0..2 (high-density).
  - Sparse dict-per-node child tables at depths 3..K (K=7).
  - PPMd method D with exclusion; escape e = n/(2c); order -1 = uniform 1/256.
  - node_path tracking — no bytes(ctx[i:]) slice per update step.
  - Periodic pruning every 75 M bytes (drop depth-K nodes with total < 3).
  - Online eval-time observe() updates are kept on (per spec).
  - K=6 fallback if first 10 MB throughput < 0.7 MB/s (≈ > 14 s).

See .survey/designs/method_ppm-context-tree_pass_2.md for full design notes.
"""
from __future__ import annotations

__author__ = "@survey-ppm-p2"

import array
import os
import time

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Hyperparameters (per spec — DO NOT TUNE)
# ---------------------------------------------------------------------------

K_DEFAULT = 7
K_FALLBACK = 6
MAX_NODES = 24_000_000
DENSE_DEPTHS = 3                # depths 0, 1, 2 use the dense child table
PRUNE_EVERY_BYTES = 75_000_000
PRUNE_THRESHOLD = 3             # drop depth-K nodes with total < this
TARGET_TRAIN_BYTES = 220_000_000
FALLBACK_FIRST_10MB_SECONDS = 14.0  # if first 10 MB > this → K=6
UNIFORM_PROB = 1.0 / 256.0

# Dense table for depths 0..2 (1 root + ≤256 depth-1 + ≤65,536 depth-2)
# Each dense row is a 256-entry array.array("i") (4 B each, ~1 KB row).
# At most ~66 K rows → ~66 MB. Allocated lazily.


# ---------------------------------------------------------------------------
# Arena-allocated PPMd trie
# ---------------------------------------------------------------------------

class ArenaPPMd:
    """Arena-allocated PPMd context trie (no numpy).

    Parallel parallel-indexed structures (per node id):
      - total:       array.array('l') — total observation count.
      - depth_arr:   array.array('b') — depth of each node (0 = root).
      - dense_rows:  list of (array.array('l') of length 256) | None — for
                     depth < DENSE_DEPTHS nodes; None otherwise.
      - sparse_children: list[dict[int,int]] | None — for depth >= DENSE_DEPTHS.
      - counts:      list[dict[int,int]] — byte→count.

    Node 0 is root.
    """

    def __init__(self, max_order: int):
        self.K = max_order
        self.n_nodes = 0
        self.bytes_seen = 0
        self.arena_full = False

        # array.array('l') is a signed long (8 B on x64 linux). We have
        # ≤24M nodes so 'l' is overkill for ids but cheap for totals.
        # Pre-allocate zeros for total/depth — these are by-id arrays.
        # We use array.array to avoid 24M PyObject ints in a list.
        self.total = array.array('l', [0] * MAX_NODES)
        self.depth_arr = array.array('b', [0] * MAX_NODES)

        # By-id lists; appended as nodes are created. We start at len==0
        # and grow via .append so memory only matches actual node count.
        self.dense_rows: list = []     # parallel to dense_node_ids
        # Per-node lookup: for dense nodes, give index into dense_rows;
        # for sparse nodes, -1. Encoded as a single array.array('l').
        self.dense_idx = array.array('l', [-1] * MAX_NODES)

        # Per-node sparse children dict (only set for depth >= DENSE_DEPTHS).
        # We over-allocate a None placeholder list — Python lists of None
        # are cheap (one shared PyObject for None).
        self.sparse_children: list = [None] * MAX_NODES
        # Per-node byte→count dict. Same approach.
        self.counts: list = [None] * MAX_NODES

        # Allocate root.
        self._new_node(depth=0)

    # -------------------- arena ops --------------------

    def _new_node(self, depth: int) -> int:
        """Append a new node; return its id. Returns -1 if arena full."""
        nid = self.n_nodes
        if nid >= MAX_NODES:
            self.arena_full = True
            return -1
        self.n_nodes = nid + 1
        self.depth_arr[nid] = depth
        self.counts[nid] = {}
        if depth < DENSE_DEPTHS:
            # Allocate a dense 256-entry row of zeros.
            row = array.array('l', [0] * 256)
            self.dense_idx[nid] = len(self.dense_rows)
            self.dense_rows.append(row)
        else:
            self.sparse_children[nid] = {}
        return nid

    def _get_or_create_child(self, nid: int, b: int, child_depth: int) -> int:
        """Return child id for (nid, b); create lazily. -1 on arena exhaustion."""
        di = self.dense_idx[nid]
        if di >= 0:
            row = self.dense_rows[di]
            c = row[b]
            if c != 0:
                return c
            if self.arena_full:
                return -1
            new_id = self._new_node(depth=child_depth)
            if new_id < 0:
                return -1
            row[b] = new_id
            return new_id
        else:
            d = self.sparse_children[nid]
            c = d.get(b)
            if c is not None:
                return c
            if self.arena_full:
                return -1
            new_id = self._new_node(depth=child_depth)
            if new_id < 0:
                return -1
            d[b] = new_id
            return new_id

    # -------------------- pruning --------------------

    def prune_depth_k(self, threshold: int) -> int:
        """Drop depth-K nodes whose total < threshold.

        Marks them dead (zero counts, no removal from arena) and clears
        their parent's child pointer. Pure-Python scan; we run this only
        ~3 times in a 220 MB run.
        """
        K = self.K
        n = self.n_nodes
        depth_arr = self.depth_arr
        total = self.total

        # Collect victims.
        victims: set[int] = set()
        for nid in range(1, n):  # skip root
            if depth_arr[nid] == K and total[nid] < threshold:
                victims.add(nid)
        if not victims:
            return 0

        parent_depth = K - 1
        # Walk parents at depth K-1 and clear pointers into victims.
        if parent_depth < DENSE_DEPTHS:
            for pid in range(n):
                if depth_arr[pid] != parent_depth:
                    continue
                di = self.dense_idx[pid]
                if di < 0:
                    continue
                row = self.dense_rows[di]
                for b in range(256):
                    cid = row[b]
                    if cid != 0 and cid in victims:
                        row[b] = 0
        else:
            for pid in range(n):
                if depth_arr[pid] != parent_depth:
                    continue
                d = self.sparse_children[pid]
                if not d:
                    continue
                rm = [b for b, cid in d.items() if cid in victims]
                for b in rm:
                    del d[b]

        # Zero out victims.
        dropped = 0
        for nid in victims:
            total[nid] = 0
            self.counts[nid] = {}
            if self.sparse_children[nid] is not None:
                self.sparse_children[nid] = {}
            dropped += 1
        return dropped


# ---------------------------------------------------------------------------
# Training (single left-to-right pass with node_path tracking)
# ---------------------------------------------------------------------------

def _train_loop(model: ArenaPPMd, train_bytes: bytes, n_bytes: int,
                t0: float) -> None:
    K = model.K
    total = model.total
    counts = model.counts
    depth_arr = model.depth_arr
    get_or_create_child = model._get_or_create_child

    # node_path[i] is the node id at order i (0 = root).
    node_path: list[int] = [0]

    next_prune_at = PRUNE_EVERY_BYTES

    for i in range(n_bytes):
        b = train_bytes[i]

        # 1) Update counts along the live path.
        for nid in node_path:
            c = counts[nid]
            c[b] = c.get(b, 0) + 1
            total[nid] += 1

        # 2) Advance.
        new_path: list[int] = [0]
        for nid in node_path:
            d = depth_arr[nid]
            if d >= K:
                continue
            cid = get_or_create_child(nid, b, d + 1)
            if cid < 0:
                continue
            new_path.append(cid)
        node_path = new_path

        model.bytes_seen = i + 1

        # Periodic pruning.
        if model.bytes_seen >= next_prune_at:
            t_prune0 = time.monotonic()
            n_before = model.n_nodes
            dropped = model.prune_depth_k(PRUNE_THRESHOLD)
            elapsed = time.monotonic() - t_prune0
            print(f"[ppmd2] prune @ {model.bytes_seen:,} bytes: "
                  f"dropped {dropped:,} depth-{K} nodes "
                  f"(arena_used={n_before:,}, {elapsed:.1f}s)",
                  flush=True)
            next_prune_at += PRUNE_EVERY_BYTES

        # Periodic progress.
        if (i + 1) % 5_000_000 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / max(1e-9, elapsed)
            print(f"[ppmd2] trained {i+1:,}/{n_bytes:,} bytes  "
                  f"({rate:,.0f} byte/s, {elapsed:.1f}s, "
                  f"arena={model.n_nodes:,} nodes)",
                  flush=True)


def _resume_train_loop(model: ArenaPPMd, train_bytes: bytes, n_bytes: int,
                       t0: float) -> None:
    """Same as _train_loop but preserves model.bytes_seen and aligns the
    next prune to the global 75 MB cadence.
    """
    K = model.K
    total = model.total
    counts = model.counts
    depth_arr = model.depth_arr
    get_or_create_child = model._get_or_create_child

    node_path: list[int] = [0]

    seen = model.bytes_seen
    next_prune_at = ((seen // PRUNE_EVERY_BYTES) + 1) * PRUNE_EVERY_BYTES

    for i in range(n_bytes):
        b = train_bytes[i]

        for nid in node_path:
            c = counts[nid]
            c[b] = c.get(b, 0) + 1
            total[nid] += 1

        new_path: list[int] = [0]
        for nid in node_path:
            d = depth_arr[nid]
            if d >= K:
                continue
            cid = get_or_create_child(nid, b, d + 1)
            if cid < 0:
                continue
            new_path.append(cid)
        node_path = new_path

        model.bytes_seen += 1

        if model.bytes_seen >= next_prune_at:
            t_prune0 = time.monotonic()
            n_before = model.n_nodes
            dropped = model.prune_depth_k(PRUNE_THRESHOLD)
            elapsed = time.monotonic() - t_prune0
            print(f"[ppmd2] prune @ {model.bytes_seen:,} bytes: "
                  f"dropped {dropped:,} depth-{K} nodes "
                  f"(arena_used={n_before:,}, {elapsed:.1f}s)",
                  flush=True)
            next_prune_at += PRUNE_EVERY_BYTES

        if (i + 1) % 5_000_000 == 0:
            elapsed = time.monotonic() - t0
            rate = model.bytes_seen / max(1e-9, elapsed)
            print(f"[ppmd2] trained {model.bytes_seen:,} bytes  "
                  f"({rate:,.0f} byte/s, {elapsed:.1f}s, "
                  f"arena={model.n_nodes:,} nodes)",
                  flush=True)


# ---------------------------------------------------------------------------
# Prediction (PPMd-D with exclusion, traversed via node_path)
# ---------------------------------------------------------------------------

class PPMdCharModel(CharModel):
    """CharModel wrapper around ArenaPPMd.

    Maintains its own node_path during eval. observe(c) updates counts
    AND advances the node_path (online learning, per spec §3).
    """

    def __init__(self, model: ArenaPPMd):
        self.model = model
        self.node_path: list[int] = [0]
        self._byte_to_str = [bytes([b]).decode("latin-1") for b in range(256)]

    def reset(self) -> None:
        self.node_path = [0]

    def predict(self) -> dict[str, float]:
        return self._predict_str()

    def observe(self, char: str) -> None:
        m = self.model
        total = m.total
        counts = m.counts
        depth_arr = m.depth_arr
        get_or_create_child = m._get_or_create_child
        K = m.K
        node_path = self.node_path
        for b in char.encode("utf-8"):
            for nid in node_path:
                c = counts[nid]
                c[b] = c.get(b, 0) + 1
                total[nid] += 1
            new_path = [0]
            for nid in node_path:
                d = depth_arr[nid]
                if d >= K:
                    continue
                cid = get_or_create_child(nid, b, d + 1)
                if cid < 0:
                    continue
                new_path.append(cid)
            node_path = new_path
            m.bytes_seen += 1
        self.node_path = node_path

    def _predict_str(self) -> dict[str, float]:
        m = self.model
        total = m.total
        counts = m.counts
        node_path = self.node_path
        b2s = self._byte_to_str

        out_b: dict[int, float] = {}
        excluded: set[int] = set()
        remaining = 1.0

        # node_path[0] is root; larger indices are deeper. Walk deepest→shallow.
        for k in range(len(node_path) - 1, -1, -1):
            nid = node_path[k]
            tot = total[nid]
            if tot <= 0:
                continue
            cnt = counts[nid]
            if not cnt:
                continue
            if excluded:
                eff = {b: c for b, c in cnt.items() if b not in excluded}
            else:
                eff = cnt
            n_eff = len(eff)
            if n_eff == 0:
                continue
            c_eff = 0
            for v in eff.values():
                c_eff += v
            if c_eff <= 0:
                continue
            escape = n_eff / (2.0 * c_eff)
            if escape > 1.0:
                escape = 1.0
            keep = 1.0 - escape
            inv_c = 1.0 / c_eff
            for b, c in eff.items():
                p = (c - 0.5) * inv_c * keep * remaining
                if p > 0.0:
                    out_b[b] = out_b.get(b, 0.0) + p
                excluded.add(b)
            remaining *= escape
            if remaining <= 0.0:
                break

        if remaining > 0.0:
            n_remaining = 256 - len(excluded)
            if n_remaining > 0:
                share = remaining / n_remaining
                for b in range(256):
                    if b not in excluded:
                        out_b[b] = out_b.get(b, 0.0) + share
            else:
                share = remaining / 256.0
                for b in range(256):
                    out_b[b] = out_b.get(b, 0.0) + share

        out: dict[str, float] = {}
        for b, p in out_b.items():
            if p > 0.0:
                out[b2s[b]] = p
        return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        print(f"[ppmd2] SEED={seed_env} (PPM is deterministic; logged only)")

    raw = train_text.encode("utf-8")
    n_total = len(raw)
    n_train = min(TARGET_TRAIN_BYTES, n_total)
    train_bytes = raw[:n_train]
    print(f"[ppmd2] train_text={n_total:,} bytes total; "
          f"target {n_train:,} bytes; K={K_DEFAULT}", flush=True)

    K = K_DEFAULT
    model = ArenaPPMd(max_order=K)

    t0 = time.monotonic()

    # Phase 1: first 10 MB. If too slow, restart at K=6.
    probe_n = min(10_000_000, n_train)
    _train_loop(model, train_bytes, probe_n, t0)
    elapsed = time.monotonic() - t0
    print(f"[ppmd2] first {probe_n:,} bytes in {elapsed:.1f}s "
          f"(arena={model.n_nodes:,} nodes)", flush=True)

    fell_back = False
    if elapsed > FALLBACK_FIRST_10MB_SECONDS and K == K_DEFAULT:
        print(f"[ppmd2] throughput < 0.7 MB/s → restart at K={K_FALLBACK}",
              flush=True)
        K = K_FALLBACK
        del model
        model = ArenaPPMd(max_order=K)
        t0 = time.monotonic()
        probe_n = 0
        fell_back = True

    # Phase 2: remaining bytes.
    if probe_n < n_train:
        remaining_bytes = bytes(train_bytes[probe_n:])
        _resume_train_loop(model, remaining_bytes, len(remaining_bytes), t0)

    total_elapsed = time.monotonic() - t0
    print(f"[ppmd2] training done: {model.bytes_seen:,} bytes "
          f"in {total_elapsed:.1f}s; arena={model.n_nodes:,} nodes; "
          f"K={K}{' (fell back)' if fell_back else ''}",
          flush=True)

    return PPMdCharModel(model)
