"""PPMd-D context-trie submission via a C extension compiled at start-up.

Per research/catalog/new_directions/spec_7_ppm_neural_residual.md, Part A
(PPM-only, no neural residual). The C trie is the same one in
research/ppm-c-extension/code/ppm_core.c, embedded here as a string and
compiled to /tmp/ppm_core.so inside train(). We rely on the prebuilt
Modal image only for Python + libc; gcc is apt-installed on demand.

Why a single Modal submission file: submit.py ships the user's
submission.py as bytes; there is no sibling-file channel. The C source
and the build glue therefore live inline.

Energy story: the workload is entirely CPU. NVML measures GPU energy
and subtracts a 50 W idle baseline; with the GPU at rest, the reported
training_energy_J should be close to zero. The survey's pure-Python
PPMd-D reached 0.6300 / 633 J on a slow local CPU; the C core targets
≥ 5 MB/s on the Modal box (see scaling_analysis.md), enough to ingest
the full 541 MB train split inside the 300 s wall-clock.

Online updates remain on during eval — observe() bumps trie counts as
the harness streams val chars through. Matches the pass-1 setup.
"""
from __future__ import annotations

__author__ = "@ab-10"

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
# Embedded C source — same algorithm as research/ppm-c-extension/code/ppm_core.c
# with three public wrappers added (ppm_predict_argmax,
# ppm_predict_dist, ppm_observe_byte) so we can drive the trie from a
# streaming CharModel rather than only via the bulk train/eval paths.
# ---------------------------------------------------------------------------

PPM_C_SOURCE = r"""
/* PPMd-D (Cleary-Witten method D) over a byte-level context trie. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K_MAX 8

typedef struct {
    int32_t count;
    int32_t child_id;
    uint8_t byte;
    uint8_t _pad[3];
} Entry;

typedef struct PPM {
    int K;

    int32_t *node_total;
    int8_t  *node_depth;
    int16_t *node_cap;
    int16_t *node_n_kids;
    int32_t *node_entries;
    int32_t  n_nodes;
    int32_t  cap_nodes;

    Entry   *entries;
    int64_t  entries_used;
    int64_t  entries_cap;

    int32_t  path[K_MAX + 1];
    int      path_len;

    int64_t  bytes_seen;
    int64_t  n_node_exhausted;
    int64_t  n_entries_exhausted;
} PPM;

static int32_t alloc_node(PPM *p, int8_t depth) {
    if (p->n_nodes >= p->cap_nodes) {
        p->n_node_exhausted++;
        return -1;
    }
    int32_t nid = p->n_nodes++;
    p->node_total[nid] = 0;
    p->node_depth[nid] = depth;
    p->node_cap[nid] = 0;
    p->node_n_kids[nid] = 0;
    p->node_entries[nid] = -1;
    return nid;
}

static int32_t alloc_entries(PPM *p, int cap) {
    if (p->entries_used + cap > p->entries_cap) {
        p->n_entries_exhausted++;
        return -1;
    }
    int32_t off = (int32_t)p->entries_used;
    p->entries_used += cap;
    memset(&p->entries[off], 0, sizeof(Entry) * (size_t)cap);
    return off;
}

static void grow_node(PPM *p, int32_t nid) {
    int old_cap = p->node_cap[nid];
    int new_cap = old_cap * 2;
    if (new_cap > 256) new_cap = 256;
    if (new_cap == old_cap) return;

    int32_t old_off = p->node_entries[nid];
    int32_t new_off = alloc_entries(p, new_cap);
    if (new_off < 0) return;

    Entry *old_e = &p->entries[old_off];
    Entry *new_e = &p->entries[new_off];

    if (new_cap == 256) {
        for (int i = 0; i < old_cap; i++) {
            if (old_e[i].count == 0 && old_e[i].child_id == 0) continue;
            new_e[old_e[i].byte] = old_e[i];
        }
    } else {
        int log2_new = __builtin_ctz((unsigned)new_cap);
        for (int i = 0; i < old_cap; i++) {
            if (old_e[i].count == 0 && old_e[i].child_id == 0) continue;
            uint8_t b = old_e[i].byte;
            uint32_t h = ((uint32_t)b * 0x9E3779B1U) >> (32 - log2_new);
            while (new_e[h].count != 0 || new_e[h].child_id != 0) {
                h = (h + 1) & (uint32_t)(new_cap - 1);
            }
            new_e[h] = old_e[i];
        }
    }

    p->node_cap[nid] = (int16_t)new_cap;
    p->node_entries[nid] = new_off;
}

static inline int32_t find_slot(PPM *p, int32_t nid, uint8_t b, int create) {
    int cap = p->node_cap[nid];
    if (cap == 0) {
        if (!create) return -1;
        int32_t off = alloc_entries(p, 4);
        if (off < 0) return -1;
        p->node_cap[nid] = 4;
        p->node_entries[nid] = off;
        cap = 4;
    }
    if (create && cap < 256 && (int)p->node_n_kids[nid] * 8 >= cap * 5) {
        grow_node(p, nid);
        cap = p->node_cap[nid];
    }

    int32_t off = p->node_entries[nid];
    Entry *e = &p->entries[off];

    if (cap == 256) {
        return off + (int32_t)b;
    }

    int log2_cap = __builtin_ctz((unsigned)cap);
    uint32_t h = ((uint32_t)b * 0x9E3779B1U) >> (32 - log2_cap);
    while (1) {
        Entry *slot = &e[h];
        if (slot->count == 0 && slot->child_id == 0) {
            return create ? (off + (int32_t)h) : -1;
        }
        if (slot->byte == b) return off + (int32_t)h;
        h = (h + 1) & (uint32_t)(cap - 1);
    }
}

PPM *ppm_create(int K, int64_t max_nodes, int64_t max_entries) {
    if (K < 1) K = 1;
    if (K > K_MAX) K = K_MAX;
    PPM *p = (PPM *)calloc(1, sizeof(PPM));
    if (!p) return NULL;
    p->K = K;
    p->cap_nodes = (int32_t)max_nodes;
    p->node_total = (int32_t *)malloc(sizeof(int32_t) * (size_t)max_nodes);
    p->node_depth = (int8_t  *)malloc(sizeof(int8_t)  * (size_t)max_nodes);
    p->node_cap   = (int16_t *)malloc(sizeof(int16_t) * (size_t)max_nodes);
    p->node_n_kids= (int16_t *)malloc(sizeof(int16_t) * (size_t)max_nodes);
    p->node_entries=(int32_t *)malloc(sizeof(int32_t) * (size_t)max_nodes);
    p->entries_cap = max_entries;
    p->entries    = (Entry   *)malloc(sizeof(Entry)   * (size_t)max_entries);
    if (!p->node_total || !p->node_depth || !p->node_cap || !p->node_n_kids
        || !p->node_entries || !p->entries) {
        fprintf(stderr, "[ppm] malloc failed\n");
        return NULL;
    }
    p->n_nodes = 1;
    p->node_total[0] = 0;
    p->node_depth[0] = 0;
    p->node_cap[0] = 0;
    p->node_n_kids[0] = 0;
    p->node_entries[0] = -1;
    p->path[0] = 0;
    p->path_len = 1;
    return p;
}

void ppm_destroy(PPM *p) {
    if (!p) return;
    free(p->node_total);
    free(p->node_depth);
    free(p->node_cap);
    free(p->node_n_kids);
    free(p->node_entries);
    free(p->entries);
    free(p);
}

void ppm_reset_path(PPM *p) {
    p->path[0] = 0;
    p->path_len = 1;
}

int64_t ppm_train_bulk(PPM *p, const uint8_t *data, int64_t n) {
    int K = p->K;
    int32_t path[K_MAX + 1];
    int path_len = p->path_len;
    memcpy(path, p->path, (size_t)path_len * sizeof(int32_t));

    for (int64_t i = 0; i < n; i++) {
        uint8_t b = data[i];

        int32_t new_path[K_MAX + 1];
        new_path[0] = 0;
        int new_len = 1;

        for (int j = 0; j < path_len; j++) {
            int32_t nid = path[j];
            int32_t slot_idx = find_slot(p, nid, b, 1);
            if (slot_idx < 0) continue;
            Entry *e = &p->entries[slot_idx];
            int was_new = (e->count == 0);
            e->count++;
            if (was_new) {
                e->byte = b;
                p->node_n_kids[nid]++;
            }
            p->node_total[nid]++;

            int8_t d = p->node_depth[nid];
            if (d < K && new_len <= K) {
                int32_t cid = e->child_id;
                if (cid == 0) {
                    cid = alloc_node(p, (int8_t)(d + 1));
                    if (cid < 0) continue;
                    e->child_id = cid;
                }
                new_path[new_len++] = cid;
            }
        }

        memcpy(path, new_path, (size_t)new_len * sizeof(int32_t));
        path_len = new_len;
    }

    p->bytes_seen += n;
    memcpy(p->path, path, (size_t)path_len * sizeof(int32_t));
    p->path_len = path_len;
    return n;
}

/* PPMd method D + exclusion. Returns argmax byte; fills out_prob if non-NULL. */
static int predict_argmax_internal(PPM *p, double *out_prob) {
    double prob[256];
    uint8_t excluded[256];
    int b;
    for (b = 0; b < 256; b++) { prob[b] = 0.0; excluded[b] = 0; }
    double remaining = 1.0;

    for (int k = p->path_len - 1; k >= 0; k--) {
        int32_t nid = p->path[k];
        int cap = p->node_cap[nid];
        if (cap == 0) continue;
        int32_t off = p->node_entries[nid];
        Entry *e = &p->entries[off];

        int64_t c_eff = 0;
        int n_eff = 0;
        for (int s = 0; s < cap; s++) {
            if (e[s].count > 0 && !excluded[e[s].byte]) {
                c_eff += e[s].count;
                n_eff++;
            }
        }
        if (n_eff == 0 || c_eff <= 0) continue;
        double escape = (double)n_eff / (2.0 * (double)c_eff);
        if (escape > 1.0) escape = 1.0;
        double keep = 1.0 - escape;
        double inv_c = 1.0 / (double)c_eff;

        for (int s = 0; s < cap; s++) {
            if (e[s].count > 0 && !excluded[e[s].byte]) {
                prob[e[s].byte] += ((double)e[s].count - 0.5) * inv_c * keep * remaining;
                excluded[e[s].byte] = 1;
            }
        }
        remaining *= escape;
        if (remaining <= 0.0) break;
    }

    if (remaining > 0.0) {
        int n_rem = 0;
        for (b = 0; b < 256; b++) if (!excluded[b]) n_rem++;
        if (n_rem > 0) {
            double share = remaining / (double)n_rem;
            for (b = 0; b < 256; b++) if (!excluded[b]) prob[b] += share;
        }
    }

    if (out_prob) memcpy(out_prob, prob, sizeof(prob));

    int best = 0;
    for (b = 1; b < 256; b++) if (prob[b] > prob[best]) best = b;
    return best;
}

static void observe_one(PPM *p, uint8_t b, int do_update) {
    int K = p->K;
    int32_t new_path[K_MAX + 1];
    new_path[0] = 0;
    int new_len = 1;

    for (int j = 0; j < p->path_len; j++) {
        int32_t nid = p->path[j];
        int32_t slot_idx = find_slot(p, nid, b, do_update ? 1 : 0);
        if (slot_idx < 0) continue;
        Entry *e = &p->entries[slot_idx];
        if (e->count == 0 && !do_update) continue;
        if (do_update) {
            int was_new = (e->count == 0);
            e->count++;
            if (was_new) {
                e->byte = b;
                p->node_n_kids[nid]++;
            }
            p->node_total[nid]++;
        }
        int8_t d = p->node_depth[nid];
        if (d < K && new_len <= K) {
            int32_t cid = e->child_id;
            if (cid == 0 && do_update) {
                cid = alloc_node(p, (int8_t)(d + 1));
                if (cid >= 0) e->child_id = cid;
            }
            if (cid != 0) new_path[new_len++] = cid;
        }
    }

    memcpy(p->path, new_path, (size_t)new_len * sizeof(int32_t));
    p->path_len = new_len;
}

/* Streaming API used by the Python CharModel wrapper. */
int ppm_predict_argmax(PPM *p) {
    return predict_argmax_internal(p, NULL);
}

void ppm_predict_dist(PPM *p, double *out_prob) {
    (void)predict_argmax_internal(p, out_prob);
}

void ppm_observe_byte(PPM *p, uint8_t b, int do_update) {
    observe_one(p, b, do_update);
}

int64_t ppm_n_nodes(PPM *p)        { return p->n_nodes; }
int64_t ppm_entries_used(PPM *p)   { return p->entries_used; }
int64_t ppm_bytes_seen(PPM *p)     { return p->bytes_seen; }
int64_t ppm_node_exhausted(PPM *p) { return p->n_node_exhausted; }
int64_t ppm_entries_exhausted(PPM *p) { return p->n_entries_exhausted; }
"""


# ---------------------------------------------------------------------------
# Build glue: ensure gcc, write source, compile to /tmp/ppm_core.so, load
# ---------------------------------------------------------------------------

def _ensure_gcc() -> str:
    """Return path to a C compiler, apt-installing gcc if necessary."""
    for cc in ("cc", "gcc"):
        path = shutil.which(cc)
        if path:
            return path
    # python:3.11-slim ships without gcc; install on demand.
    print("[ppm_c] gcc not found; apt-installing ...", flush=True)
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
    tmp = Path(tempfile.mkdtemp(prefix="ppm_c_"))
    src = tmp / "ppm_core.c"
    so = tmp / "ppm_core.so"
    src.write_text(PPM_C_SOURCE)
    print(f"[ppm_c] compiling {src.name} ...", flush=True)
    t0 = time.monotonic()
    subprocess.run(
        [cc, "-O3", "-march=native", "-shared", "-fPIC",
         "-o", str(so), str(src)],
        check=True,
    )
    print(f"[ppm_c] compiled in {time.monotonic()-t0:.1f}s -> {so}", flush=True)

    lib = ctypes.CDLL(str(so))

    lib.ppm_create.argtypes = [c_int, c_int64, c_int64]
    lib.ppm_create.restype = c_void_p

    lib.ppm_destroy.argtypes = [c_void_p]
    lib.ppm_destroy.restype = None

    lib.ppm_train_bulk.argtypes = [c_void_p, POINTER(c_uint8), c_int64]
    lib.ppm_train_bulk.restype = c_int64

    lib.ppm_reset_path.argtypes = [c_void_p]
    lib.ppm_reset_path.restype = None

    lib.ppm_predict_argmax.argtypes = [c_void_p]
    lib.ppm_predict_argmax.restype = c_int

    lib.ppm_predict_dist.argtypes = [c_void_p, POINTER(c_double)]
    lib.ppm_predict_dist.restype = None

    lib.ppm_observe_byte.argtypes = [c_void_p, c_uint8, c_int]
    lib.ppm_observe_byte.restype = None

    for name in ("ppm_n_nodes", "ppm_entries_used", "ppm_bytes_seen",
                 "ppm_node_exhausted", "ppm_entries_exhausted"):
        getattr(lib, name).argtypes = [c_void_p]
        getattr(lib, name).restype = c_int64

    return lib


# ---------------------------------------------------------------------------
# CharModel wrapper
# ---------------------------------------------------------------------------

# Bytes 0x00-0x7F decode as single-char UTF-8; the harness's predict()
# dict is keyed by char and `evaluate()` takes max over the dict, so we
# pre-build a (byte -> char) table for ASCII and skip the rest.
_ASCII_CHARS: list[str | None] = [
    chr(b) if b < 0x80 else None for b in range(256)
]


class PPMCharModel(CharModel):
    """Streaming CharModel backed by the PPMd-D C trie.

    The harness calls reset() once and then alternates predict() /
    observe(char). Each observe(char) decomposes the (potentially
    multi-byte) UTF-8 char into bytes and observes each, with online
    updates enabled — the trie continues to grow during eval. This
    matches the pass-1 survey setup that reached val char-acc 0.6300.
    """
    def __init__(self, lib: ctypes.CDLL, handle: c_void_p):
        self._lib = lib
        self._p = handle
        # Reusable 256-double output buffer for ppm_predict_dist.
        self._dist_buf = (c_double * 256)()
        # Cheap argmax-only fast path returns just the int; we fall back
        # to the full dist only when the argmax byte is non-ASCII (rare).

    def reset(self) -> None:
        self._lib.ppm_reset_path(self._p)

    def predict(self) -> dict[str, float]:
        argmax_byte = int(self._lib.ppm_predict_argmax(self._p))
        ch = _ASCII_CHARS[argmax_byte]
        if ch is not None:
            return {ch: 1.0}
        # Argmax byte is in 0x80-0xFF (a UTF-8 continuation or lead byte
        # by itself). The harness only matches ASCII single-char keys, so
        # report the best ASCII byte we can find instead.
        self._lib.ppm_predict_dist(self._p, self._dist_buf)
        best_b = 0
        best_p = -1.0
        for b in range(0x80):
            p = self._dist_buf[b]
            if p > best_p:
                best_p = p
                best_b = b
        # _ASCII_CHARS[best_b] is non-None since best_b < 0x80.
        return {_ASCII_CHARS[best_b]: 1.0}  # type: ignore[dict-item]

    def observe(self, char: str) -> None:
        for byte in char.encode("utf-8"):
            self._lib.ppm_observe_byte(self._p, c_uint8(byte), 1)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

# PPM trie sizes. At K=6 / WikiText train (541 MB), the survey's local
# run grew to ~6M nodes and ~50M entries; the budgets here are 3x that
# to give headroom for online updates during the eval phase.
K_ORDER = 6
MAX_NODES = 20_000_000
MAX_ENTRIES = 200_000_000

# Reserve a small slice of the 300 s wall-clock budget for compilation
# and final accounting. Compile is ~1 s, gcc apt-install is ~10–20 s,
# train_bulk is the rest.
TRAIN_BUDGET_S = 270.0
TRAIN_CHUNK_BYTES = 10_000_000


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        print(f"[ppm_c] SEED={seed_env} (ignored — PPMd-D is deterministic)")

    t0 = time.monotonic()
    lib = _build_lib()

    print(
        f"[ppm_c] ppm_create K={K_ORDER} "
        f"max_nodes={MAX_NODES:,} max_entries={MAX_ENTRIES:,}",
        flush=True,
    )
    handle = lib.ppm_create(K_ORDER, MAX_NODES, MAX_ENTRIES)
    if not handle:
        raise RuntimeError("ppm_create failed (OOM)")

    train_bytes = train_text.encode("utf-8")
    n_total = len(train_bytes)
    print(f"[ppm_c] train bytes={n_total:,}", flush=True)

    pos = 0
    last_print = time.monotonic()
    while pos < n_total:
        elapsed = time.monotonic() - t0
        if elapsed >= TRAIN_BUDGET_S:
            break
        end = min(n_total, pos + TRAIN_CHUNK_BYTES)
        chunk = train_bytes[pos:end]
        buf = (c_uint8 * len(chunk)).from_buffer_copy(chunk)
        lib.ppm_train_bulk(handle, buf, len(chunk))
        pos = end

        now = time.monotonic()
        if now - last_print >= 5.0 or pos == n_total:
            elapsed = now - t0
            rate = pos / max(1e-9, elapsed - 0.0)
            n_nodes = lib.ppm_n_nodes(handle)
            entries = lib.ppm_entries_used(handle)
            print(
                f"[ppm_c] {pos:>11,} / {n_total:,} bytes "
                f"({100.0 * pos / n_total:5.1f}%)  "
                f"{rate / 1e6:5.2f} MB/s  elapsed={elapsed:6.1f}s  "
                f"nodes={n_nodes:>10,}  entries={entries:>11,}",
                flush=True,
            )
            last_print = now

    train_elapsed = time.monotonic() - t0
    bytes_seen = lib.ppm_bytes_seen(handle)
    n_nodes = lib.ppm_n_nodes(handle)
    entries = lib.ppm_entries_used(handle)
    print(
        f"[ppm_c] train done: {bytes_seen:,} bytes in {train_elapsed:.1f}s "
        f"({bytes_seen / max(1e-9, train_elapsed) / 1e6:.2f} MB/s)  "
        f"nodes={n_nodes:,}  entries={entries:,}",
        flush=True,
    )

    return PPMCharModel(lib, handle)
