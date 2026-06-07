"""Spectral Newton — Muon + PPM hybrid for byte-level LM.

Backprop (Muon + AdamW) trains the modded-nanogpt transformer. Closed-form /
counting pieces: PPM trie (CPU), bigram-SVD embedding warm-start, and
transformer+PPM logit ensemble at inference. Per-layer Muon LRs use spectral
curvature exponents α from the Spectral Alignment Decomposition paper (cheap α^0.5 scaling, not
full T(σ;α) SN optimizer).

NOTE (ported by @miyuhoriuchi, 2026-06-04): the only change from @atbcalvo's
original is `predict()` returns a single committed `str` (current 2026-05-28
contract) instead of `dict[str, float]` (old contract). Semantics preserved:
it commits the argmax over ASCII bytes of the same ensemble distribution.
"""
from __future__ import annotations

__author__ = "@miyuhoriuchi + @atbcalvo"

import base64
import ctypes
import os
import numpy as np
import shutil
import subprocess
import tempfile
import time
from ctypes import POINTER, c_double, c_int, c_int64, c_uint8, c_void_p
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel, evaluate

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

K_ORDER = 7
MAX_NODES = 20_000_000
MAX_ENTRIES = 200_000_000
PPM_TRAIN_BUDGET_S = 25.0
PPM_CHUNK_BYTES = 10_000_000
PPM_PROGRESS_EVERY_S = 5.0
TRAIN_BUDGET_S = 290.0

BETA_CANDIDATES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0)
TUNE_VAL_CHARS = 2_000
TUNE_BUDGET_S = 15.0
LOG_EPS = 1e-12

# Populated by build_ppm_so.sh + embed_ppm_so.py (linux amd64); None = compile at runtime
PPM_SO_B64: str | None = None


_ASCII_CHARS: list[str | None] = [chr(b) if b < 0x80 else None for b in range(256)]

# ---------------------------------------------------------------------------
# PPM C extension (inference + bulk train only)
# ---------------------------------------------------------------------------

PPM_C_SOURCE = r"""
#include <stdint.h>
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
    if (p->n_nodes >= p->cap_nodes) { p->n_node_exhausted++; return -1; }
    int32_t nid = p->n_nodes++;
    p->node_total[nid] = 0;
    p->node_depth[nid] = depth;
    p->node_cap[nid] = 0;
    p->node_n_kids[nid] = 0;
    p->node_entries[nid] = -1;
    return nid;
}

static int32_t alloc_entries(PPM *p, int cap) {
    if (p->entries_used + cap > p->entries_cap) { p->n_entries_exhausted++; return -1; }
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
            while (new_e[h].count != 0 || new_e[h].child_id != 0)
                h = (h + 1) & (uint32_t)(new_cap - 1);
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
    if (cap == 256) return off + (int32_t)b;
    int log2_cap = __builtin_ctz((unsigned)cap);
    uint32_t h = ((uint32_t)b * 0x9E3779B1U) >> (32 - log2_cap);
    while (1) {
        Entry *slot = &e[h];
        if (slot->count == 0 && slot->child_id == 0)
            return create ? (off + (int32_t)h) : -1;
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
        || !p->node_entries || !p->entries) return NULL;
    p->n_nodes = 1;
    p->path[0] = 0;
    p->path_len = 1;
    return p;
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
            if (was_new) { e->byte = b; p->node_n_kids[nid]++; }
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

static int predict_argmax_internal(PPM *p, double *out_prob) {
    double prob[256];
    uint8_t excluded[256];
    for (int b = 0; b < 256; b++) { prob[b] = 0.0; excluded[b] = 0; }
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
            if (e[s].count > 0 && !excluded[e[s].byte]) { c_eff += e[s].count; n_eff++; }
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
        for (int b = 0; b < 256; b++) if (!excluded[b]) n_rem++;
        if (n_rem > 0) {
            double share = remaining / (double)n_rem;
            for (int b = 0; b < 256; b++) if (!excluded[b]) prob[b] += share;
        }
    }
    if (out_prob) memcpy(out_prob, prob, sizeof(prob));
    int best = 0;
    for (int b = 1; b < 256; b++) if (prob[b] > prob[best]) best = b;
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
            if (was_new) { e->byte = b; p->node_n_kids[nid]++; }
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

void ppm_predict_dist(PPM *p, double *out_prob) { (void)predict_argmax_internal(p, out_prob); }
void ppm_observe_byte(PPM *p, uint8_t b, int do_update) { observe_one(p, b, do_update); }
int64_t ppm_n_nodes(PPM *p) { return p->n_nodes; }
int64_t ppm_entries_used(PPM *p) { return p->entries_used; }
int64_t ppm_bytes_seen(PPM *p) { return p->bytes_seen; }
"""


def _ensure_gcc() -> str:
    for cc in ("cc", "gcc"):
        path = shutil.which(cc)
        if path:
            print(f"[sn] using {path}", flush=True)
            return path
    print("[sn] gcc not found; apt-installing ...", flush=True)
    t0 = time.monotonic()
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "--no-install-recommends", "gcc", "libc6-dev"],
        check=True,
    )
    path = shutil.which("gcc")
    if not path:
        raise RuntimeError("gcc still missing after apt-get install")
    print(f"[sn] gcc installed in {time.monotonic() - t0:.1f}s", flush=True)
    return path


def _configure_ppm_lib(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.ppm_create.argtypes = [c_int, c_int64, c_int64]
    lib.ppm_create.restype = c_void_p
    lib.ppm_train_bulk.argtypes = [c_void_p, POINTER(c_uint8), c_int64]
    lib.ppm_train_bulk.restype = c_int64
    lib.ppm_reset_path.argtypes = [c_void_p]
    lib.ppm_reset_path.restype = None
    lib.ppm_predict_dist.argtypes = [c_void_p, POINTER(c_double)]
    lib.ppm_predict_dist.restype = None
    lib.ppm_observe_byte.argtypes = [c_void_p, c_uint8, c_int]
    lib.ppm_observe_byte.restype = None
    for name in ("ppm_n_nodes", "ppm_entries_used", "ppm_bytes_seen"):
        getattr(lib, name).argtypes = [c_void_p]
        getattr(lib, name).restype = c_int64
    return lib


def _build_ppm_lib() -> ctypes.CDLL:
    bundled = Path(__file__).resolve().parent / "ppm_core.so"
    if bundled.is_file():
        print(f"[sn] loading bundled PPM {bundled.name}", flush=True)
        return _configure_ppm_lib(ctypes.CDLL(str(bundled)))

    if PPM_SO_B64:
        so_dir = Path(tempfile.mkdtemp(prefix="sn_ppm_b64_"))
        so_path = so_dir / "ppm_core.so"
        so_path.write_bytes(base64.b64decode(PPM_SO_B64))
        print("[sn] loading embedded PPM .so", flush=True)
        return _configure_ppm_lib(ctypes.CDLL(str(so_path)))

    cc = _ensure_gcc()
    tmp = Path(tempfile.mkdtemp(prefix="sn_ppm_"))
    src = tmp / "ppm_core.c"
    so = tmp / "ppm_core.so"
    src.write_text(PPM_C_SOURCE)
    print("[sn] compiling PPM ...", flush=True)
    t0 = time.monotonic()
    subprocess.run(
        [cc, "-O3", "-shared", "-fPIC", "-o", str(so), str(src)],
        check=True,
    )
    print(f"[sn] PPM compiled in {time.monotonic() - t0:.1f}s", flush=True)
    return _configure_ppm_lib(ctypes.CDLL(str(so)))


def _train_ppm(lib: ctypes.CDLL, train_bytes: bytes) -> c_void_p:
    print(f"[sn] ppm_create K={K_ORDER}", flush=True)
    handle = lib.ppm_create(K_ORDER, MAX_NODES, MAX_ENTRIES)
    if not handle:
        raise RuntimeError("ppm_create failed")
    n_total = len(train_bytes)
    pos = 0
    t0 = time.monotonic()
    last_print = t0
    while pos < n_total:
        if time.monotonic() - t0 >= PPM_TRAIN_BUDGET_S:
            print(f"[sn] PPM budget hit at {pos:,}/{n_total:,} bytes", flush=True)
            break
        end = min(n_total, pos + PPM_CHUNK_BYTES)
        chunk = train_bytes[pos:end]
        buf = (c_uint8 * len(chunk)).from_buffer_copy(chunk)
        lib.ppm_train_bulk(handle, buf, len(chunk))
        pos = end
        now = time.monotonic()
        if lib.ppm_n_nodes(handle) >= MAX_NODES - 1:
            print(
                f"[sn] PPM stopping early at {pos:,} bytes — node cap full "
                f"(nodes={lib.ppm_n_nodes(handle):,})",
                flush=True,
            )
            break
        if now - last_print >= PPM_PROGRESS_EVERY_S or pos == n_total:
            rate = pos / max(1e-9, now - t0)
            print(
                f"[sn] PPM {pos:>11,}/{n_total:,} ({100.0 * pos / n_total:5.1f}%) "
                f"{rate / 1e6:.2f} MB/s  nodes={lib.ppm_n_nodes(handle):,}",
                flush=True,
            )
            last_print = now
    print(
        f"[sn] PPM done: {lib.ppm_bytes_seen(handle):,} bytes in "
        f"{time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return handle


# ---------------------------------------------------------------------------
# Architecture (modded-nanogpt, vocab_size=256, RoPE offset support)
# ---------------------------------------------------------------------------

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

    def forward(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.relu().square()
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim=head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h, new_kv = self.attn(self.norm1(x), kv_cache, offset=offset)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        head_dim: int = 64,
        max_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.model_dim = model_dim
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(
            [Block(model_dim, head_dim=head_dim) for _ in range(num_layers)]
        )
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward_body(
        self,
        inputs: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        x = self.norm1(self.embed(inputs))
        new_caches: list[tuple[Tensor, Tensor]] = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv, offset=offset)
            new_caches.append(new_kv)
        return self.norm2(x), new_caches

    def forward(
        self,
        inputs: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        h, new_caches = self.forward_body(inputs, kv_caches, offset=offset)
        logits = self.proj(h).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits, new_caches


# ---------------------------------------------------------------------------
# Muon optimizer (fast spectral flattening — from modded-nanogpt)
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
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


def muon_update(grad: Tensor, momentum: Tensor, mu: float = 0.95, nesterov: bool = True) -> Tensor:
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, mu: float = 0.95):
        params = list(params)
        assert len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):  # type: ignore[override]
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


ALPHA_ATTN_QKV = 1.43
ALPHA_ATTN_PROJ = 1.17
ALPHA_MLP_UP = 0.84
ALPHA_MLP_DOWN = 1.15


def _build_optimizers(
    model: GPT,
    cfg: "TrainConfig",
    device: torch.device,
) -> list[torch.optim.Optimizer]:
    scalars = [p for p in model.parameters() if p.ndim < 2]
    optimizer_adam = AdamW(
        [
            dict(params=[model.embed.weight], lr=cfg.embed_lr),
            dict(params=[model.proj.weight], lr=cfg.head_lr),
            dict(params=scalars, lr=cfg.scalar_lr),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_adam]

    layer_groups: dict[str, tuple[list[nn.Parameter], float]] = {
        "attn_qkv": ([], ALPHA_ATTN_QKV),
        "attn_proj": ([], ALPHA_ATTN_PROJ),
        "mlp_up": ([], ALPHA_MLP_UP),
        "mlp_down": ([], ALPHA_MLP_DOWN),
    }
    for name, p in model.blocks.named_parameters():
        if p.ndim < 2:
            continue
        if ".attn.q." in name or ".attn.k." in name or ".attn.v." in name:
            layer_groups["attn_qkv"][0].append(p)
        elif ".attn.proj." in name:
            layer_groups["attn_proj"][0].append(p)
        elif ".mlp.fc." in name:
            layer_groups["mlp_up"][0].append(p)
        elif ".mlp.proj." in name:
            layer_groups["mlp_down"][0].append(p)

    for gname, (params, alpha) in layer_groups.items():
        if not params:
            continue
        lr_mult = alpha ** 0.5
        lr = cfg.muon_lr * lr_mult
        opt = Muon(params, lr=lr, weight_decay=cfg.muon_wd)
        optimizers.append(opt)
        n_el = sum(p.numel() for p in params)
        print(
            f"[sn] Muon {gname}: {len(params)} params ({n_el/1e6:.2f}M) "
            f"α={alpha:.2f} lr_mult={lr_mult:.2f} lr={lr:.4f}",
            flush=True,
        )
    return optimizers


# ---------------------------------------------------------------------------
# Init + warm-start
# ---------------------------------------------------------------------------

def _init_modded(model: GPT) -> None:
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


def _ppm_warmstart_embed(model: GPT, train_bytes: bytes, device: torch.device) -> None:
    """Initialize embeddings from bigram co-occurrence SVD (closed-form)."""
    t0 = time.monotonic()
    sample = train_bytes[: min(len(train_bytes), 50_000_000)]
    arr = torch.frombuffer(bytearray(sample), dtype=torch.uint8)
    if arr.numel() < 2:
        return
    prev = arr[:-1].long()
    nxt = arr[1:].long()
    counts = torch.zeros(256, 256, dtype=torch.float64)
    counts.index_put_((prev, nxt), torch.ones(prev.numel(), dtype=torch.float64), accumulate=True)
    row_sum = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    trans = counts / row_sum
    trans = trans + 1e-6
    u, s, vh = torch.linalg.svd(trans, full_matrices=False)
    d = model.model_dim
    # Left + right singular features give up to 512 dims from 256x256 transition matrix.
    us = u * s.sqrt().unsqueeze(0)
    vs = vh.T * s.sqrt().unsqueeze(0)
    combined = torch.cat([us, vs], dim=1)
    if combined.size(1) < d:
        pad = torch.randn(256, d - combined.size(1), dtype=combined.dtype) * 0.01
        combined = torch.cat([combined, pad], dim=1)
    embed = combined[:, :d].to(device=device, dtype=torch.bfloat16)
    model.embed.weight.data.copy_(embed)
    print(f"[sn] PPM warm-start embed done in {time.monotonic() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainConfig:
    def __init__(
        self,
        model_dim=384,
        num_layers=6,
        head_dim=64,
        max_len=1024,
        batch_size=32,
        n_steps=2150,
        cooldown_frac=0.7,
        embed_lr=0.3,
        head_lr=1.0 / 320,
        scalar_lr=0.01,
        muon_lr=0.035,
        muon_wd=0.025,
        log_every=200,
        use_compile=True,
    ):
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.cooldown_frac = cooldown_frac
        self.embed_lr = embed_lr
        self.head_lr = head_lr
        self.scalar_lr = scalar_lr
        self.muon_lr = muon_lr
        self.muon_wd = muon_wd
        self.log_every = log_every
        self.use_compile = use_compile

    def __repr__(self) -> str:
        return (
            f"TrainConfig(d={self.model_dim} L={self.num_layers} "
            f"bs={self.batch_size} T={self.max_len} steps={self.n_steps})"
        )


def _train_spectral_newton(
    train_bytes: bytes,
    cfg: TrainConfig,
    device: torch.device,
) -> GPT:
    raw_tensor = torch.frombuffer(bytearray(train_bytes), dtype=torch.uint8).to(device)
    n = raw_tensor.numel()
    if n < cfg.max_len + 1:
        raise ValueError(f"need at least {cfg.max_len + 1} bytes; got {n}")

    model = GPT(
        vocab_size=256,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        head_dim=cfg.head_dim,
        max_len=cfg.max_len,
    ).to(device)
    _init_modded(model)
    _ppm_warmstart_embed(model, train_bytes, device)

    optimizers = _build_optimizers(model, cfg, device)
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sn] {n_params / 1e6:.2f}M params  cfg={cfg}", flush=True)

    if cfg.use_compile and device.type == "cuda":
        print("[sn] torch.compile (static) ...", flush=True)
        model = torch.compile(model)

    def set_lr(step: int) -> None:
        progress = step / cfg.n_steps
        if progress < 1 - cfg.cooldown_frac:
            eta = 1.0
        else:
            eta = max(0.0, (1 - progress) / cfg.cooldown_frac)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * eta

    model.train()
    use_amp = device.type == "cuda"
    t0 = time.monotonic()
    for step in range(cfg.n_steps):
        set_lr(step)
        idx = torch.randint(0, n - cfg.max_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(cfg.max_len + 1, device=device)[None, :]
        flat = raw_tensor[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        for opt in optimizers:
            opt.step()

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            print(
                f"[sn] step {step:5d}/{cfg.n_steps}  loss {loss.item():.4f}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    del optimizers
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return model


# ---------------------------------------------------------------------------
# Streaming CharModel with PPM ensemble
# ---------------------------------------------------------------------------

class SpectralNewtonCharModel(CharModel):
    def __init__(
        self,
        model: GPT,
        lib: ctypes.CDLL,
        ppm_handle: c_void_p,
        device: torch.device,
        mix_beta: float = 1.0,
    ):
        self.model = model
        self._lib = lib
        self._ppm = ppm_handle
        self.device = device
        self.mix_beta = mix_beta
        self._dist_buf = (c_double * 256)()
        self.model.eval()
        self._kv: list[tuple[Tensor, Tensor]] | None = None
        self._next_logits: Tensor | None = None
        self._pos: int = 0

    @torch.no_grad()
    def reset(self) -> None:
        self._lib.ppm_reset_path(self._ppm)
        self._kv = None
        self._pos = 0
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        logits, self._kv = self.model(x, None, offset=self._pos)
        self._next_logits = logits[0, -1]
        self._pos = 1

    def _ppm_probs(self) -> Tensor:
        self._lib.ppm_predict_dist(self._ppm, self._dist_buf)
        arr = np.frombuffer(self._dist_buf, dtype=np.float64, count=256).copy()
        p = torch.from_numpy(arr).to(device=self.device, dtype=torch.float32)
        return p.clamp_min(LOG_EPS) / p.clamp_min(LOG_EPS).sum()

    @torch.no_grad()
    def predict(self) -> str:
        # PORT: current contract requires a single committed `str`. Compute the
        # same transformer+PPM ensemble distribution as the original, then
        # commit the argmax over ASCII bytes (identical to the byte the old
        # dict-returning version would have argmax'd to).
        if self._next_logits is None:
            raise RuntimeError("predict() called before reset()")
        p_tf = F.softmax(self._next_logits.float(), dim=-1)
        if self.mix_beta >= 0.999:
            probs = p_tf
        else:
            p_ppm = self._ppm_probs()
            probs = self.mix_beta * p_tf + (1.0 - self.mix_beta) * p_ppm
            probs = probs / probs.sum()
        for byte_id in torch.argsort(probs, descending=True).tolist():
            ch = _ASCII_CHARS[byte_id]
            if ch is not None:
                return ch
        return ""

    @torch.no_grad()
    def observe(self, char: str) -> None:
        if self._kv is None:
            raise RuntimeError("observe() called before reset()")
        for byte in char.encode("utf-8"):
            self._lib.ppm_observe_byte(self._ppm, c_uint8(byte), 1)
            self._maybe_trim_cache()
            x = torch.tensor([[byte]], dtype=torch.long, device=self.device)
            logits, self._kv = self.model(x, self._kv, offset=self._pos)
            self._next_logits = logits[0, -1]
            self._pos += 1

    def _maybe_trim_cache(self) -> None:
        if self._kv is None:
            return
        cur = self._kv[0][0].shape[2]
        if cur < self.model.max_len:
            return
        keep = self.model.max_len - 1
        self._kv = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in self._kv]


def _tune_mix_beta(model: SpectralNewtonCharModel, valid_text: str) -> float:
    tune_text = valid_text[:TUNE_VAL_CHARS]
    if len(tune_text) < 500:
        return model.mix_beta
    best_beta = 1.0
    best_acc = -1.0
    t0 = time.monotonic()
    print(f"[sn] tuning mix_beta on {len(tune_text):,} val chars ...", flush=True)
    for beta in BETA_CANDIDATES:
        if time.monotonic() - t0 >= TUNE_BUDGET_S:
            break
        model.mix_beta = beta
        model.reset()
        acc = evaluate(model, tune_text).accuracy
        print(f"[sn]   beta={beta:.2f}  acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_beta = beta
    print(f"[sn] selected mix_beta={best_beta:.2f} (acc={best_acc:.4f})", flush=True)
    model.mix_beta = best_beta
    model.reset()
    return best_beta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[sn] SEED={seed}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    print(f"[sn] device={device} cfg={cfg}", flush=True)

    t0 = time.monotonic()
    deadline = t0 + TRAIN_BUDGET_S
    print("[sn] phase 1/3: build PPM", flush=True)
    lib = _build_ppm_lib()
    train_bytes = train_text.encode("utf-8")
    ppm_handle = _train_ppm(lib, train_bytes)

    print("[sn] phase 2/3: spectral newton training", flush=True)
    model = _train_spectral_newton(train_bytes, cfg, device)

    print("[sn] phase 3/3: assemble inference model", flush=True)
    char_model = SpectralNewtonCharModel(model, lib, ppm_handle, device)
    if valid_text and time.monotonic() < deadline - 8.0:
        _tune_mix_beta(char_model, valid_text)
    elif valid_text:
        print("[sn] beta tuning skipped — time budget", flush=True)

    print(f"[sn] train() complete in {time.monotonic() - t0:.1f}s", flush=True)
    return char_model
