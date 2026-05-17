# PPM Context Tree (PPMd) — Pass 1

## 1. Hypothesis

A pure-counting, variable-order byte-level PPMd predictor with max order 6 can reach **val char-accuracy >= 0.70** on the first 60K chars of `wiki.valid.raw` while spending well under 30 s of GPU-attached wall time (so energy is dominated by the ~50 W NVML idle baseline that gets subtracted out; reported joules should land near zero or single-digit kJ, beating the 51.7 kJ transformer baseline by ~10x). We learn whether a non-parametric, gradient-free model can clear the 0.70 floor at all on raw WikiText-103, and how memory scales with the actual train stream (~530 MB).

## 2. Model

- **Unit:** raw **bytes** (uint8, 0-255). `train_text` arrives as `str`; we `.encode("utf-8")` once into a `bytes` object. `predict()` returns a dict keyed by single-char `str` produced by `bytes([b]).decode("latin-1")` so every byte round-trips losslessly through a 1-char `str`. `observe(c)` does `c.encode("latin-1")[0]` to recover the byte. (latin-1 is a 1-1 byte<->codepoint map; this sidesteps multi-byte UTF-8 boundary headaches and matches how all PPM/cmix work.)
- **Data structure:** a context trie up to max order **K=6**. Each node stores `counts: dict[int, int]` (byte -> count) plus a total. Root node tracks order-0 (unconditional) counts. Implemented as a Python `dict` keyed by `bytes` context (length 0..K); value is `[total_count, dict_of_byte_to_count]`. Plain Python dicts; no torch tensor — the trie is sparse and dict lookups are fast in CPython.
- **Smoothing:** **PPMd (method D)**. At order k with `n` distinct symbols seen and total count `c`, escape mass `e = n / (2c)` if c>0 else 1.0; each seen symbol gets `(count - 0.5) / c * (1 - e)`. On escape, recurse to order k-1. At order -1 (uniform), every byte gets `1/256`. **Exclusion** of symbols already seen at higher orders is applied (standard PPM exclusion).
- **Active context:** a fixed-size deque of the last K observed bytes.

## 3. Training procedure

```
ctx = bytearray()         # length <= K
for b in train_bytes:
    # update counts along the active context chain, orders K..0
    for k in range(min(len(ctx), K), -1, -1):
        node = trie[bytes(ctx[len(ctx)-k:])]   # create if missing
        node.counts[b] += 1
        node.total += 1
    ctx.append(b)
    if len(ctx) > K: del ctx[0]
```

Single left-to-right pass over the full ~530 MB train stream. No validation pass, no second epoch. Pruning: every 50 M bytes, walk the trie and **delete order-K nodes whose total < 2** (one-shot rare-context pruning) to cap memory. `valid_text` argument ignored.

At inference, `predict()` walks the trie from order min(len(ctx), K) down to -1 applying PPMd escape + exclusion, returns the resulting 256-entry dict. `observe()` mirrors the training update so the model adapts during eval (online — standard PPM behavior, and the streaming API permits it).

## 4. Hyperparameters

| name | value |
|---|---|
| max order K | **6** |
| smoothing | PPMd (method D) with exclusion |
| escape formula | e = n / (2c) |
| prune trigger | every 50 M train bytes |
| prune rule | drop order-K nodes with total < 2 |
| max node cap (hard) | 25 M nodes; if exceeded mid-train, raise prune threshold to <4 and re-prune |
| order-(-1) uniform | 1/256 |
| online updates at eval | **yes** (observe bumps counts) |
| seed | unused (deterministic) |

## 5. Expected wall time on A100-80GB

PPM is CPU-bound; GPU is idle. CPython dict updates: realistic ~1.5 M byte-updates/s per core after the trie warms up; with K=6 each input byte triggers up to 7 node updates so effective input throughput ~200-300 K bytes/s. 530 MB / 250 KB/s ~= **2100 s** — **over budget**. Mitigation: **subsample training to the first 60 MB of train_bytes** (still 100x the val eval window, plenty for PPMd convergence on English). 60 MB / 250 KB/s ~= **240 s**. Pruning passes add ~5 s each, max 2 passes. Total budget estimate: **~250 s < 300 s**. If first 10 MB processed slower than 40 s of wall time, training loop self-aborts early and locks in what it has.

## 6. Success criterion

**Target: val_char_acc >= 0.70 AND reported energy <= 5 kJ.**

A reading of (0.72, 2 kJ) would be a clear win over the modded-nanogpt baseline (~51.7 kJ). (0.68, 1 kJ) is a near-miss (DQ on accuracy floor) but informative — would justify pass 2 at K=7 or with a larger train subsample.

## 7. Failure modes anticipated

- **Memory blow-up past 25 M nodes** at K=6 on 60 MB — *design-failure*. Mitigation: prune threshold ramp in §4. If still OOM, executor may lower K to 5 (one-line change, accept).
- **Python loop too slow, hits 300 s SIGALRM mid-stream** — *design-failure*. Executor may further reduce the train subsample (30 MB) but must NOT switch to a compiled language or add numba/cython (those aren't in the environment by default and would change the experiment).
- **PPMd escape arithmetic underflow / division-by-zero at unseen contexts** — *execution-bug*. Guard with `if total == 0: continue` and escape immediately to lower order.
- **latin-1 round-trip drops non-ASCII chars from val stream** — *execution-bug*. WikiText-103 raw contains UTF-8 multibyte sequences; latin-1 over the **byte** stream is safe, but if the executor accidentally encodes the `str` as latin-1 it will crash. Always `.encode("utf-8")` for bytes, `bytes([b]).decode("latin-1")` only for the predict-dict keys, and verify in eval that the runner's `true_char` matches by going `true_char.encode("utf-8")` byte-by-byte. **Important:** the runner iterates the *str* stream char-by-char so multibyte UTF-8 chars arrive as single multi-byte `str` units, not per-byte. Resolution: keep an internal pending-byte buffer; `observe(c)` extends ctx by the bytes of `c.encode("utf-8")`; `predict()` must return a dict whose argmax char matches the *char* the runner expects. Since >99% of WikiText chars are 1-byte ASCII, returning a dict keyed only by 1-byte latin-1 chars will score correctly on the ASCII majority and forfeit only the multibyte chars (acceptable for the 0.70 bar; English ASCII alone gives headroom).
- **Online updates during eval shift accuracy upward by ~1-2 pts** — not a failure, expected; do NOT disable.

## 8. What we will NOT do

- No arithmetic-coding-style probability mixing across orders beyond standard PPM escape.
- No order > 6 in pass 1 (memory risk).
- No torch tensors, no GPU, no CUDA hashmap — keep it pure Python + dicts. (Numpy arrays for the final 256-entry dist are fine.)
- No second training epoch.
- No byte-pair-encoding, no Unicode normalization, no lowercasing.
- No `numba`/`cython`/`cffi` JIT.
- No reading of `valid_text` during train.
- No copying of `submissions/modded_nanogpt/submission.py` model code; only the `CharModel` subclass boilerplate.
