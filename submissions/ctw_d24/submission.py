"""Bit-level Context-Tree Weighting (CTW, Willems-Shtarkov-Tjalkens 1995).

Per research/non_nn_methods/spec_07_context_tree_weighting.md.

Mechanism:
  * Bit-level CTW with depth D=24 (3 bytes of bit context). Each tree
    node holds the per-context KT counts (n0, n1) and a cached
    log_alpha = log(P_e(s)) - log(P_w(left_child)) - log(P_w(right_child)).
  * KT estimator: P_e(b|n0,n1) = (n_b + 0.5) / (n0 + n1 + 1).
  * Weighted-mixture predictive recursion (Willems 1995, Section IV.A):
        R(node, b) = (alpha(node) * KT(node, b) + R(child(node, ctx_bit), b))
                     / (alpha(node) + 1)
    R(root, b) is the CTW predictive probability of bit b given the
    current context. At leaves R = KT, alpha is conceptually infinite
    so the recursion bottoms out.
  * Byte prediction: tree-expand 8 bits in nested fashion. At each
    bit-level we branch into 0/1 and accumulate log-probabilities, so
    we cover all 256 byte hypotheses in 1+2+4+...+128 = 255 bit-
    predictions per char.
  * Hash trie storage in a fixed-cap arena (no realloc); each node has
    integer ids for its two children. Path-only updates: only D nodes
    touched per bit observation.

C extension is built in-process via gcc (apt-installed on demand) — same
pattern as research/catalog/new_directions/ppm_c/.

Training: one streaming pass over a head slice of the train corpus.
Online updates remain on during eval (CharModel.observe()).

Energy story: pure CPU on a chunky byte-level workload. GPU idles. NVML
subtracts ~50W idle baseline; expected training_energy_J ~ few hundred J
if the streaming pass finishes in 60-120s wall.
"""
from __future__ import annotations

__author__ = "@worker-ctw"

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
from ctypes import POINTER, c_double, c_int, c_int32, c_int64, c_uint8, c_void_p
from pathlib import Path

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Embedded C source — bit-level CTW
# ---------------------------------------------------------------------------

CTW_C_SOURCE = r"""
/* Bit-level Context-Tree Weighting (Willems-Shtarkov-Tjalkens 1995).

   Numerically-stable formulation: each node stores β ∈ [0, 1] where
        β = P_e / (P_e + P_w_child_along * P_w_other_child)
   instead of the unbounded log_pw / log_pe. The CTW predictive
   recursion becomes
        R(node, b) = β * kt(b) + (1-β) * R(child_along_path, b)
   and the per-bit update is
        β_new = β * kt(b) / (β * kt(b) + (1-β) * R(child_along_path, b))
   which only involves bounded quantities. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D_MAX 32

/* Node layout: 20 bytes packed.
   - n0/n1: KT counts.
   - child0/child1: ids of the two children (0 = absent).
   - beta: stable mixture weight in [0, 1]. Initial value 1.0 (no
     children means CTW falls back to the KT estimator at this node
     alone). */
typedef struct {
    int32_t n0;
    int32_t n1;
    int32_t child0;
    int32_t child1;
    float   beta;
} Node;

typedef struct CTW {
    int      D;             /* tree depth in bits */
    Node    *nodes;
    int32_t  n_nodes;
    int32_t  cap_nodes;

    /* Bit history buffer — most recent D bits, wrapped. */
    uint8_t  hist[D_MAX];   /* hist[(head + i) % D] = bit at age i */
    int      head;          /* index of NEXT slot to write */
    int      hist_fill;     /* number of valid bits in hist (capped at D) */

    /* Path cache: nodes visited on the most recent walk, in order
       root → leaf. Updated lazily by observe_bit. */
    int32_t  path[D_MAX + 1];
    int      path_len;

    int64_t  bits_seen;
    int64_t  n_node_exhausted;
} CTW;


/* KT predictive probability of next bit b at a node with counts (n0, n1):
   P_e(b | n0,n1) = (n_b + 0.5) / (n0 + n1 + 1). */
static inline float kt_predict(int32_t n0, int32_t n1, int b) {
    return (b ? (n1 + 0.5f) : (n0 + 0.5f)) / ((float)(n0 + n1) + 1.0f);
}

CTW *ctw_create(int D, int64_t max_nodes) {
    if (D < 1) D = 1;
    if (D > D_MAX) D = D_MAX;
    CTW *p = (CTW *)calloc(1, sizeof(CTW));
    if (!p) return NULL;
    p->D = D;
    p->cap_nodes = (int32_t)max_nodes;
    p->nodes = (Node *)calloc((size_t)max_nodes, sizeof(Node));
    if (!p->nodes) {
        free(p);
        return NULL;
    }
    /* node 0 is the root. beta defaults to 1.0 — with no children we
       fall back to the KT estimator at this node alone. As children
       are observed, beta evolves toward P_e/(P_e + P_w_children). */
    p->n_nodes = 1;
    /* Initial β = P_e / (P_e + P_w_children) = 1 / (1 + 1) = 0.5 (virgin). */
    p->nodes[0].beta = 0.5f;
    p->head = 0;
    p->hist_fill = 0;
    return p;
}

void ctw_destroy(CTW *p) {
    if (!p) return;
    free(p->nodes);
    free(p);
}

void ctw_reset_path(CTW *p) {
    p->head = 0;
    p->hist_fill = 0;
    p->path_len = 0;
}

static inline int32_t alloc_node(CTW *p) {
    if (p->n_nodes >= p->cap_nodes) {
        p->n_node_exhausted++;
        return 0;  /* return root as a "no-grow" sentinel */
    }
    int32_t nid = p->n_nodes++;
    /* calloc already zeroed counts/children; init β to 0.5 (virgin
       subtree: P_e=1, P_w_children=1·1=1, β = 1/2). */
    p->nodes[nid].beta = 0.5f;
    return nid;
}

/* Read the i-th most-recent bit from the history (i=0 is most recent). */
static inline int hist_bit(const CTW *p, int i) {
    /* head points at the next slot to write; the most recent bit is at
       (head - 1) mod D. We index by age i in [0, hist_fill). */
    int idx = p->head - 1 - i;
    while (idx < 0) idx += D_MAX;
    return p->hist[idx];
}

/* Compute the path from root to depth D corresponding to the current
   context (the D most recent bits), allocating nodes as needed if
   `create` is non-zero. Returns the path length (== D+1) or fewer if
   create=0 and the trie doesn't have all the nodes.

   Path[0] is always the root (node 0). Path[d] is at depth d. */
static int walk_path(CTW *p, int32_t *path, int create) {
    int D = p->D;
    int fill = p->hist_fill;
    if (fill > D) fill = D;
    path[0] = 0;
    int32_t cur = 0;
    int d;
    for (d = 0; d < fill; d++) {
        int bit = hist_bit(p, d);   /* most-recent bits drive the path */
        Node *nd = &p->nodes[cur];
        int32_t cid = bit ? nd->child1 : nd->child0;
        if (cid == 0) {
            if (!create) {
                path[d + 1] = 0;
                return d + 1;
            }
            cid = alloc_node(p);
            if (cid == 0) {
                /* arena exhausted — stop growing path */
                return d + 1;  /* path[d+1] = 0 ; treat as missing */
            }
            if (bit) nd->child1 = cid; else nd->child0 = cid;
        }
        path[d + 1] = cid;
        cur = cid;
    }
    /* If hist_fill < D, the rest of the path is "absent" (id 0). */
    for (; d < D; d++) path[d + 1] = 0;
    return D + 1;
}

/* Compute leaf_depth: the deepest path node index that physically
   exists (after walk_path with create=0 or 1). path[0] is always root. */
static inline int find_leaf_depth(CTW *p, const int32_t *path) {
    int D = p->D;
    int fill = p->hist_fill;
    if (fill > D) fill = D;
    int leaf_depth = fill;
    while (leaf_depth > 0 && path[leaf_depth] == 0) leaf_depth--;
    return leaf_depth;
}

/* Update the path's beta and counts bottom-up after observing bit b
   at the deepest matched node. Uses the stable recursive formula:
        R(d, b) = β_d * kt_d(b) + (1-β_d) * R(d+1, b)
        β_d_new = (β_d * kt_d(b)) / R(d, b)
   where R(leaf, b) = kt(leaf, b). All quantities in [0, 1]. */
static void update_path(CTW *p, const int32_t *path, int b) {
    int leaf_depth = find_leaf_depth(p, path);

    /* Compute R at the leaf first. */
    int32_t leaf_id = path[leaf_depth];
    Node *leaf = &p->nodes[leaf_id];
    float R = kt_predict(leaf->n0, leaf->n1, b);
    /* Bump leaf count. */
    if (b) leaf->n1 = leaf->n1 + 1; else leaf->n0 = leaf->n0 + 1;
    /* Leaf's beta stays 1.0 — at this node, there are no observed
       child-subtrees yet (or it's at maximum depth). */

    /* Climb bottom-up. */
    for (int d = leaf_depth - 1; d >= 0; d--) {
        int32_t nid = path[d];
        Node *nd = &p->nodes[nid];
        int32_t n0 = nd->n0;
        int32_t n1 = nd->n1;
        float kt_b = kt_predict(n0, n1, b);

        float beta_old = nd->beta;
        float R_new = beta_old * kt_b + (1.0f - beta_old) * R;

        /* Update beta. R_new should be > 0; clamp to avoid div-by-zero. */
        if (R_new < 1e-30f) R_new = 1e-30f;
        float beta_new = (beta_old * kt_b) / R_new;
        if (beta_new < 0.0f) beta_new = 0.0f;
        if (beta_new > 1.0f) beta_new = 1.0f;
        nd->beta = beta_new;

        /* Bump count. */
        if (b) nd->n1 = n1 + 1; else nd->n0 = n0 + 1;

        /* Propagate R up. */
        R = R_new;
    }
}

/* Push a bit into the history buffer. Call AFTER observe_bit has used
   the OLD context to find the path. */
static inline void hist_push(CTW *p, int bit) {
    p->hist[p->head] = (uint8_t)(bit & 1);
    p->head = (p->head + 1) % D_MAX;
    if (p->hist_fill < D_MAX) p->hist_fill++;
}

/* Observe a single bit: walk the current-context path (creating
   nodes), bump KT counts at every node on the path, recompute log_pw
   bottom-up, then push the bit into the history. */
void ctw_observe_bit(CTW *p, int b) {
    int32_t path[D_MAX + 1];
    walk_path(p, path, 1);
    update_path(p, path, b & 1);
    hist_push(p, b);
    p->bits_seen++;
}

/* Bulk: observe `n` bits from a buffer of bytes (one bit per call).
   For training. Returns # of bits processed. */
int64_t ctw_observe_bytes(CTW *p, const uint8_t *data, int64_t n_bytes) {
    int64_t total = 0;
    for (int64_t i = 0; i < n_bytes; i++) {
        uint8_t byte = data[i];
        /* MSB-first bit order. */
        for (int b = 7; b >= 0; b--) {
            int bit = (byte >> b) & 1;
            ctw_observe_bit(p, bit);
            total++;
        }
    }
    return total;
}

/* CTW predictive probability of next bit = b given the current
   context. Returns P(bit=b) in [0, 1] (linear, not log).
   This does NOT mutate the tree. */
static float ctw_prob_bit(CTW *p, int b) {
    int32_t path[D_MAX + 1];
    walk_path(p, path, 0);  /* read-only prediction */
    int leaf_depth = find_leaf_depth(p, path);

    /* R(leaf, b) = kt(leaf, b). */
    Node *leaf = &p->nodes[path[leaf_depth]];
    float R = kt_predict(leaf->n0, leaf->n1, b);

    /* Climb bottom-up:
        R(d, b) = β_d · kt_d(b) + (1 - β_d) · R(d+1, b). */
    for (int d = leaf_depth - 1; d >= 0; d--) {
        Node *nd = &p->nodes[path[d]];
        float kt_b = kt_predict(nd->n0, nd->n1, b);
        float beta = nd->beta;
        R = beta * kt_b + (1.0f - beta) * R;
    }
    return R;
}

/* Log version for argmax-byte (sum 8 log-probs per candidate byte). */
static float ctw_log_prob_bit(CTW *p, int b) {
    float p_b = ctw_prob_bit(p, b);
    if (p_b < 1e-30f) p_b = 1e-30f;
    return logf(p_b);
}

/* Predict next byte: tree-expand 8 bits, compute log P for each of 256
   possible bytes, return argmax. Each "branch" extends history by 1
   bit, computes bit-CTW log-prob, recurses.

   Approach: enumerate all 256 candidate bytes; for each, walk 8 bit-
   predictions while tentatively shifting bits into the history. We
   revert history mutations after each candidate so the global state is
   unchanged.

   Naive cost: 256 * 8 = 2048 ctw_log_prob_bit calls per char. */
int ctw_predict_argmax_byte(CTW *p) {
    double best_logp = -1e30;
    int best_byte = 0;
    /* Save history state. */
    int saved_head = p->head;
    int saved_fill = p->hist_fill;
    /* hist[] is mutated; we'll save & restore the prefix we touch. */
    uint8_t saved_hist[D_MAX];
    memcpy(saved_hist, p->hist, sizeof(saved_hist));

    for (int byte = 0; byte < 256; byte++) {
        double logp = 0.0;
        /* Reset history state for each candidate. */
        p->head = saved_head;
        p->hist_fill = saved_fill;
        memcpy(p->hist, saved_hist, sizeof(saved_hist));

        for (int bit_idx = 7; bit_idx >= 0; bit_idx--) {
            int bit = (byte >> bit_idx) & 1;
            float lp = ctw_log_prob_bit(p, bit);
            logp += (double)lp;
            /* Shift this bit into history for the next bit's prediction. */
            hist_push(p, bit);
        }

        if (logp > best_logp) {
            best_logp = logp;
            best_byte = byte;
        }
    }

    /* Restore history. */
    p->head = saved_head;
    p->hist_fill = saved_fill;
    memcpy(p->hist, saved_hist, sizeof(saved_hist));

    return best_byte;
}

/* Faster: tree-expanded byte prediction. Visit each of the 8 levels
   in order, branching from N leaves into 2N (bit=0 or bit=1) at each
   level. Cost: 1+2+4+...+128 = 255 ctw_log_prob_bit calls per char. */
int ctw_predict_argmax_byte_fast(CTW *p) {
    /* leaves[i] holds log P for the byte-prefix i (of length depth_so_far). */
    static double leaves[256];
    leaves[0] = 0.0;
    int n_leaves = 1;

    /* Save history. */
    int saved_head = p->head;
    int saved_fill = p->hist_fill;
    uint8_t saved_hist[D_MAX];
    memcpy(saved_hist, p->hist, sizeof(saved_hist));

    /* For each depth in [0..7], expand each prefix into bit=0 and bit=1. */
    static double next_leaves[256];
    for (int depth = 0; depth < 8; depth++) {
        int next_n = 0;
        for (int i = 0; i < n_leaves; i++) {
            /* The byte-prefix of length `depth` corresponding to leaf i
               is given by i itself (bits read MSB-first). Restore
               history then push those `depth` bits. */
            p->head = saved_head;
            p->hist_fill = saved_fill;
            memcpy(p->hist, saved_hist, sizeof(saved_hist));
            for (int b = depth - 1; b >= 0; b--) {
                int bit = (i >> b) & 1;
                hist_push(p, bit);
            }
            /* Predict log P(bit=0) and log P(bit=1). */
            float lp0 = ctw_log_prob_bit(p, 0);
            float lp1 = ctw_log_prob_bit(p, 1);
            next_leaves[next_n++] = leaves[i] + (double)lp0;  /* prefix (i<<1)|0 */
            next_leaves[next_n++] = leaves[i] + (double)lp1;  /* prefix (i<<1)|1 */
        }
        /* Copy into leaves for next iteration. */
        for (int i = 0; i < next_n; i++) leaves[i] = next_leaves[i];
        n_leaves = next_n;
    }

    /* Restore history. */
    p->head = saved_head;
    p->hist_fill = saved_fill;
    memcpy(p->hist, saved_hist, sizeof(saved_hist));

    int best = 0;
    double best_lp = leaves[0];
    for (int b = 1; b < 256; b++) {
        if (leaves[b] > best_lp) {
            best_lp = leaves[b];
            best = b;
        }
    }
    return best;
}

/* Observe a single byte (8 bits), MSB-first. */
void ctw_observe_byte(CTW *p, uint8_t byte) {
    for (int b = 7; b >= 0; b--) {
        int bit = (byte >> b) & 1;
        ctw_observe_bit(p, bit);
    }
}

int64_t ctw_n_nodes(CTW *p)        { return (int64_t)p->n_nodes; }
int64_t ctw_bits_seen(CTW *p)      { return p->bits_seen; }
int64_t ctw_node_exhausted(CTW *p) { return p->n_node_exhausted; }
"""


# ---------------------------------------------------------------------------
# Build glue: ensure gcc, write source, compile, load
# ---------------------------------------------------------------------------

def _ensure_gcc() -> str:
    """Return path to a C compiler, apt-installing gcc if necessary."""
    for cc in ("cc", "gcc"):
        path = shutil.which(cc)
        if path:
            return path
    print("[ctw] gcc not found; apt-installing ...", flush=True)
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "--no-install-recommends", "gcc", "libc6-dev"],
        check=True,
    )
    path = shutil.which("gcc")
    if not path:
        raise RuntimeError("gcc still missing after apt-get install")
    return path


def _build_lib() -> ctypes.CDLL:
    cc = _ensure_gcc()
    tmp = Path(tempfile.mkdtemp(prefix="ctw_"))
    src = tmp / "ctw_core.c"
    so = tmp / "ctw_core.so"
    src.write_text(CTW_C_SOURCE)
    print(f"[ctw] compiling {src.name} ...", flush=True)
    t0 = time.monotonic()
    subprocess.run(
        [cc, "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC",
         "-o", str(so), str(src), "-lm"],
        check=True,
    )
    print(f"[ctw] compiled in {time.monotonic()-t0:.1f}s -> {so}", flush=True)

    lib = ctypes.CDLL(str(so))

    lib.ctw_create.argtypes = [c_int, c_int64]
    lib.ctw_create.restype = c_void_p

    lib.ctw_destroy.argtypes = [c_void_p]
    lib.ctw_destroy.restype = None

    lib.ctw_reset_path.argtypes = [c_void_p]
    lib.ctw_reset_path.restype = None

    lib.ctw_observe_bit.argtypes = [c_void_p, c_int]
    lib.ctw_observe_bit.restype = None

    lib.ctw_observe_byte.argtypes = [c_void_p, c_uint8]
    lib.ctw_observe_byte.restype = None

    lib.ctw_observe_bytes.argtypes = [c_void_p, POINTER(c_uint8), c_int64]
    lib.ctw_observe_bytes.restype = c_int64

    lib.ctw_predict_argmax_byte.argtypes = [c_void_p]
    lib.ctw_predict_argmax_byte.restype = c_int

    lib.ctw_predict_argmax_byte_fast.argtypes = [c_void_p]
    lib.ctw_predict_argmax_byte_fast.restype = c_int

    for name in ("ctw_n_nodes", "ctw_bits_seen", "ctw_node_exhausted"):
        getattr(lib, name).argtypes = [c_void_p]
        getattr(lib, name).restype = c_int64

    return lib


# ---------------------------------------------------------------------------
# CharModel wrapper
# ---------------------------------------------------------------------------

_ASCII_CHARS: list[str | None] = [
    chr(b) if b < 0x80 else None for b in range(256)
]


class CTWCharModel(CharModel):
    """Streaming CharModel backed by the bit-level CTW C trie.

    predict() runs the tree-expanded byte argmax (255 bit-predictions);
    observe(char) bumps tree counts along the path for each of the
    char's UTF-8 bytes.
    """
    def __init__(self, lib: ctypes.CDLL, handle: c_void_p):
        self._lib = lib
        self._p = handle

    def reset(self) -> None:
        self._lib.ctw_reset_path(self._p)

    def predict(self) -> dict[str, float]:
        argmax_byte = int(self._lib.ctw_predict_argmax_byte_fast(self._p))
        ch = _ASCII_CHARS[argmax_byte]
        if ch is not None:
            return {ch: 1.0}
        # Non-ASCII argmax — pick a space character as a safe fallback
        # (still a valid char, won't crash the dict-based evaluator).
        # In practice CTW on English text predicts ASCII almost always.
        return {" ": 1.0}

    def observe(self, char: str) -> None:
        for byte in char.encode("utf-8"):
            self._lib.ctw_observe_byte(self._p, c_uint8(byte))


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

# CTW configuration.
# Depth D=24 (3 bytes of bit context). Per spec_07: "Bit-level CTW with
# byte-aligned context depth D=24 bits". Standard published configurations
# for CTW on text use D in [24, 48].
D_BITS = 24

# Hash trie size. Each node is 24 B; 40M nodes = ~1 GB. Sized for a
# 60-120 MB train slice; if we exhaust we stop growing (existing nodes
# keep updating their counts). On 60 MB train, empirical CTW node counts
# on English text are ~5-15 M for D=24.
MAX_NODES = 40_000_000

# Reserve a small slice of the 300 s wall-clock budget for compile +
# accounting. The bulk of the time goes into ctw_observe_bytes. We
# stream the head of the train split; CTW saturates fast on English.
TRAIN_BUDGET_S = 270.0

# Chunk size for the bulk loop. ~4 MB chunks keep the ctypes buffer
# allocation amortised over many bits.
TRAIN_CHUNK_BYTES = 4_000_000

# How much of the train corpus to ingest. We can't realistically process
# the full 541 MB at D=24 within 300 s (would need ~500 s pure CTW time).
# Stream the head; CTW's KT-mixture saturates the first ~30-100 MB.
MAX_TRAIN_BYTES = 80_000_000


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    del valid_text

    seed_env = os.environ.get("SEED")
    if seed_env:
        print(f"[ctw] SEED={seed_env} (ignored — CTW is deterministic)")

    t0 = time.monotonic()
    lib = _build_lib()

    print(
        f"[ctw] ctw_create D={D_BITS} max_nodes={MAX_NODES:,}",
        flush=True,
    )
    handle = lib.ctw_create(D_BITS, MAX_NODES)
    if not handle:
        raise RuntimeError("ctw_create failed (OOM)")

    train_bytes = train_text.encode("utf-8")
    n_total = min(len(train_bytes), MAX_TRAIN_BYTES)
    print(
        f"[ctw] train bytes available={len(train_bytes):,}  "
        f"ingesting head {n_total:,} bytes",
        flush=True,
    )

    pos = 0
    last_print = time.monotonic()
    while pos < n_total:
        elapsed = time.monotonic() - t0
        if elapsed >= TRAIN_BUDGET_S:
            print(f"[ctw] hit time budget at pos={pos:,}", flush=True)
            break
        end = min(n_total, pos + TRAIN_CHUNK_BYTES)
        chunk = train_bytes[pos:end]
        buf = (c_uint8 * len(chunk)).from_buffer_copy(chunk)
        lib.ctw_observe_bytes(handle, buf, len(chunk))
        pos = end

        now = time.monotonic()
        if now - last_print >= 5.0 or pos == n_total:
            elapsed = now - t0
            rate = pos / max(1e-9, elapsed)
            n_nodes = lib.ctw_n_nodes(handle)
            bits = lib.ctw_bits_seen(handle)
            print(
                f"[ctw] {pos:>11,} / {n_total:,} bytes "
                f"({100.0 * pos / n_total:5.1f}%)  "
                f"{rate / 1e6:5.2f} MB/s  elapsed={elapsed:6.1f}s  "
                f"nodes={n_nodes:>11,}  bits={bits:,}",
                flush=True,
            )
            last_print = now

    train_elapsed = time.monotonic() - t0
    n_nodes = lib.ctw_n_nodes(handle)
    bits = lib.ctw_bits_seen(handle)
    exhausted = lib.ctw_node_exhausted(handle)
    print(
        f"[ctw] train done: {pos:,} bytes / {bits:,} bits in {train_elapsed:.1f}s "
        f"({pos / max(1e-9, train_elapsed) / 1e6:.2f} MB/s)  "
        f"nodes={n_nodes:,}  node_exhausted={exhausted}",
        flush=True,
    )

    return CTWCharModel(lib, handle)
