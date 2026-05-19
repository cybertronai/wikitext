# PPM Context Tree — Pass 2 (arena-allocated trie, K=7, full-budget streaming)

## 1. Hypothesis

Pass 1's accuracy bottleneck was **data starvation**, not algorithm choice: the early-abort guard locked the trie in after only 10 MB of training (1.56 M nodes). PPMd on English with K=6 has well-documented learning curves — accuracy keeps climbing through at least 100–200 MB of training text. **Predicted lift: 0.6300 → 0.72–0.75** when we (a) replace the Python-`dict`-keyed-by-`bytes` trie with an arena-allocated child-table trie that runs **~6–8× faster per byte**, and (b) push max order from K=6 to **K=7**. Closing the 0.07-acc gap is plausible because pass 1 saw 0.63 on essentially 2% of the train budget; the marginal-character learning curve of byte-level PPM is steep in that regime.

## 2. Model

- **Unit:** raw bytes (uint8 0–255). Same UTF-8 encode / latin-1 round-trip trick as pass 1 for the `predict()` dict keys. `observe(c)` extends ctx by `c.encode("utf-8")` bytes.
- **Data structure — arena trie (the key change):**
  - Three parallel pre-allocated arrays sized to **MAX_NODES = 24,000,000**:
    - `child: int32[MAX_NODES, 256]` — child node id per (node, byte); 0 means "absent". Created lazily as a `dict[int, dict[int,int]]` for sparse non-root depths, but a **flat `numpy.int32` array of shape (MAX_NODES, 256)** for **depth 0..2** only (where density is high). Depths 3..K use sparse `dict[int,int]` per node, indexed by a flat list `sparse_children: list[dict]` of length MAX_NODES.
    - `total: int32[MAX_NODES]` — total count at each node.
    - `counts: list[dict[int,int]]` of length MAX_NODES — byte→count at the node. (Could be a flat array but most leaves are sparse; dict-per-node is the right trade-off and matches PPM cache patterns.)
  - Node 0 is root. New nodes append to the arena; node id returned. Arena exhaustion → stop creating depth-K nodes (graceful degradation to lower-order learning).
- **Smoothing:** **PPMd method D** with **exclusion** (same as pass 1). Escape `e = n/(2c)`; per-symbol `(count - 0.5)/c * (1 - e)`. Order -1 = uniform 1/256.
- **Active context:** ring buffer of last K=7 observed bytes.

## 3. Training procedure

```
# Single left-to-right pass over train_bytes; same loop over val at eval time.
node_path = [0]            # node ids at orders 0..k for the current ctx
for b in train_bytes:
    # Update counts along the live path (orders 0..min(len(ctx), K))
    for nid in node_path:
        counts[nid][b] = counts[nid].get(b, 0) + 1
        total[nid] += 1
    # Advance: descend from each path entry one step on byte b to form
    # the next-step path. Create child nodes lazily.
    new_path = [0]                                # root always present
    for nid in node_path[-K:]:                    # cap depth at K
        c = _get_or_create_child(nid, b)
        new_path.append(c)
    node_path = new_path
```

- **Pruning:** every 75 M bytes, walk the arena and zero out depth-K nodes whose total < 3 (re-use their slots via a small free-list).
- **Throughput target: 1.5 MB/s** (vs pass 1's 230 KB/s). That is ~6.5×. Justification: per-byte work is ~K dict updates + ~K child lookups; pass 1's hot path stringified `bytes(ctx[i:])` every step (allocation per order per byte) — eliminating that alone is ~3–4×. Maintaining `node_path` removes the `bytes()` slice/hash; that's the remaining ~2×. **No numba/cython** — pure CPython + numpy.
- **Eval-time updates (verified):** `wikitext.py:130` calls `model.observe(true_char)` per char. So during eval the trie keeps updating from the 60K val chars — small absolute lift but free. **Keep online updates on.**

## 4. Hyperparameters

| name | value |
|---|---|
| max order K | **7** |
| smoothing | PPMd-D with exclusion |
| escape | e = n/(2c) |
| MAX_NODES | 24,000,000 |
| dense child table for depths | 0, 1, 2 (numpy int32) |
| sparse child dicts for depths | 3..7 |
| prune trigger | every 75 M bytes |
| prune rule | drop depth-K nodes with total < 3 |
| target train bytes | 220 MB (subsample head of train_bytes) |
| throughput target | ≥1.5 MB/s sustained |
| online eval updates | **yes** |
| seed | unused |

## 5. Expected wall time on A100-80GB

CPU-bound. Budget = 300 s wall. Reserve ~10 s for boot/imports/data-load, ~5 s for two prunes, ~5 s slack = **~280 s training**. At target 1.5 MB/s sustained → 280 × 1.5 = **420 MB**, comfortably above the 220 MB target. We cap at 220 MB to leave headroom for trie-fill slowdown (later bytes are slower because deeper paths exist). If throughput in the first 10 MB measures **< 0.7 MB/s** (~14 s for 10 MB), the executor **drops K to 6** mid-init and continues (no SIGALRM abort). Memory ceiling at 24 M nodes ≈ ~6 GB Python heap — fine on the 80 GB-RAM A100 host.

## 6. Success criterion

**Pass if `val_char_acc ≥ 0.70` AND `training_energy_J ≤ 8,000 J`.** Target reading: **(0.73, ~3.5 kJ)**. Pass 1 was (0.6300, 633 J); we expect ~5–7× more wall time → ~5× energy under the 50 W idle-subtraction model (GPU stays cold), so 3–5 kJ is realistic. Even at 8 kJ this is ~6× better than the 51.7 kJ transformer baseline and clears the floor.

## 7. Failure modes anticipated

- **Throughput still below 1.5 MB/s** → less train data ingested; accuracy may land at 0.68–0.70. Mitigation: the K=6 fallback (see §5) buys ~30% throughput.
- **Memory blow-up past 24 M nodes at K=7 on 220 MB English text** → arena fills; new depth-K nodes stop being created. Lower orders keep learning, so degradation is graceful, not crash.
- **`node_path` correctness bug** (off-by-one on context length, stale ids after pruning) → silent acc loss. Mitigation: unit-style sanity assert in the first 1 M bytes that the depth-k node id retrieved via path matches the one retrieved via a fresh trie walk from root; disable assert after 1 M.
- **Dense numpy child-tables at depths 0–2 overcount memory** (256 × 3 levels × 24M nodes is impossible; only ~1 + 256 + 65536 nodes exist at those depths). Allocation must be sized by the actual count: 1 root + ≤256 depth-1 + ≤65,536 depth-2 nodes → ~17 M cells × 4 B = **68 MB**. Trivial.
- **PPMd escape underflow at unseen contexts** → guard `if total[nid] == 0: continue` and recurse to order k-1; uniform at -1.
- **Eval-time `observe()` updates blocked by harness** → confirmed *not* blocked (line 130 of `wikitext.py`); no action needed.

## 8. What we will NOT do

- No K ≥ 8 (memory + throughput risk this pass; defer to pass 3 if pass 2 lands ≥0.70).
- No blended/mixture-of-orders predictor (defer to a separate pass; this pass isolates the throughput+order lever).
- No char-class auxiliary trie (option D rejected — adds prediction-time work that costs throughput we need for training data).
- No numba, cython, cffi, or rewrite to C extension.
- No torch tensors on GPU (CPU-bound problem; GPU stays cold and that's fine — energy meter loves it).
- No multi-epoch training, no shuffling, no held-out tuning.
- No tokenization beyond raw bytes.
- No reading of `valid_text` during the train phase.

---

**Chosen variant:** Option A (arena-allocated trie at K=7, full-budget streaming).
**Success criterion:** `val_char_acc ≥ 0.70` AND `training_energy_J ≤ 8,000 J`; target (0.73, ~3.5 kJ).
