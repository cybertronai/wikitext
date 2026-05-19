/* PPMd-D (Cleary-Witten method D) over a byte-level context trie.
 *
 * Arena-allocated trie. Per-node open-addressing hashtable for the
 * children (entries combine [byte -> count, child_id]). Bulk-mode
 * train and argmax eval entry points so the per-byte Python<->C FFI
 * overhead is zero.
 *
 * ABI is plain extern "C" functions; load with ctypes.
 *
 * Build: gcc -O3 -march=native -shared -fPIC -o ppm_core.so ppm_core.c
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K_MAX 8

typedef struct {
    int32_t count;        /* 0 = empty slot (after memset)         */
    int32_t child_id;     /* 0 = no child node yet (root is id 0 but
                             root is never a child, so safe)        */
    uint8_t byte;
    uint8_t _pad[3];
} Entry;  /* 12 bytes */

typedef struct PPM {
    int K;

    /* SoA per node */
    int32_t *node_total;     /* sum of counts at node                */
    int8_t  *node_depth;     /* 0..K                                  */
    int16_t *node_cap;       /* capacity of children table (power 2) */
    int16_t *node_n_kids;    /* distinct child bytes seen            */
    int32_t *node_entries;   /* offset into entries arena, or -1     */
    int32_t  n_nodes;
    int32_t  cap_nodes;

    /* Entries arena (grow-only; old slots leaked on resize)         */
    Entry   *entries;
    int64_t  entries_used;
    int64_t  entries_cap;

    /* Live node path (root..deepest valid node along current ctx)   */
    int32_t  path[K_MAX + 1];
    int      path_len;

    /* Stats */
    int64_t  bytes_seen;
    int64_t  n_node_exhausted;
    int64_t  n_entries_exhausted;
} PPM;

/* ----------------------------------------------------------------- */
/* Allocation                                                        */
/* ----------------------------------------------------------------- */

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

/* Returns absolute entry index in p->entries, or -1.
 * For create=1, may allocate or grow the node's table.
 * Empty slot test: count==0 && child_id==0 (true after memset). */
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

/* ----------------------------------------------------------------- */
/* Public API                                                        */
/* ----------------------------------------------------------------- */

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
    /* Allocate root (id 0). */
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

/* Train on `n` bytes. Returns bytes processed (== n unless we ran out
 * of arena, in which case training continues with reduced depth). */
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

/* Argmax byte under the current path. PPMd method D + exclusion.
 * If out_prob != NULL, fills it with the 256-dim distribution. */
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

/* Observe one byte: advance path; if do_update, also bump counts. */
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

/* Eval: for each byte in `val`, take argmax over our distribution
 * under the current path, compare to truth, then observe. If
 * do_online_update, observe also bumps counts. */
int64_t ppm_eval_bulk(PPM *p, const uint8_t *val, int64_t n_val,
                      int do_online_update) {
    int64_t correct = 0;
    ppm_reset_path(p);
    for (int64_t i = 0; i < n_val; i++) {
        int pred = predict_argmax_internal(p, NULL);
        if ((uint8_t)pred == val[i]) correct++;
        observe_one(p, val[i], do_online_update);
    }
    return correct;
}

/* Char-aware eval: take a UTF-8 string and a parallel int32 array of
 * "byte offsets where each char starts" of length n_chars + 1. Returns
 * the number of chars where (a) argmax byte equals first byte of true
 * char AND (b) the true char is single-byte (i.e. ASCII). Multi-byte
 * chars are always counted as wrong but still observed.
 *
 * This matches what the CharModel runner sees: argmax over our
 * latin-1 1-byte chars vs the runner's UTF-8 char stream. */
int64_t ppm_eval_chars(PPM *p, const uint8_t *val_bytes, int64_t n_bytes,
                       const int32_t *char_offsets, int64_t n_chars,
                       int do_online_update) {
    int64_t correct = 0;
    ppm_reset_path(p);
    for (int64_t c = 0; c < n_chars; c++) {
        int32_t lo = char_offsets[c];
        int32_t hi = char_offsets[c + 1];
        int char_len = (int)(hi - lo);
        int pred = predict_argmax_internal(p, NULL);
        if (char_len == 1 && (uint8_t)pred == val_bytes[lo]) correct++;
        for (int b = 0; b < char_len; b++) {
            observe_one(p, val_bytes[lo + b], do_online_update);
        }
    }
    return correct;
}

/* Stats accessors. */
int64_t ppm_n_nodes(PPM *p)        { return p->n_nodes; }
int64_t ppm_entries_used(PPM *p)   { return p->entries_used; }
int64_t ppm_bytes_seen(PPM *p)     { return p->bytes_seen; }
int64_t ppm_node_exhausted(PPM *p) { return p->n_node_exhausted; }
int64_t ppm_entries_exhausted(PPM *p) { return p->n_entries_exhausted; }
