# P2-E — Bigram Input Encoding

**Phase.** FF investigation Phase 2 (diagnostics). **Axis varied.** Input encoding (within axis D, layer-1 treatment). **Purpose.** Test whether replacing the K-char one-hot input with a denser, slightly-structured bigram encoding improves FF's signal.

## 1. Hypothesis
Pass-2's input is K=24 char one-hots concatenated (dim 6400, only 24 non-zeros). Sum-of-sq goodness on such a sparse input gives layer 1's frozen random projection very little to discriminate on. Replacing the encoding with **(last char one-hot) ++ (last bigram one-hot)** packs more local structure into the same input dimension and may give the FF stack a stronger first-layer signal. Direct prior: pass-1's PPM-context-tree solved char-LM with explicit local-bigram statistics, suggesting bigram-level structure carries substantial signal.

## 2. Model
- **Backbone.** 5×384 FC FF stack (identical to pass-2).
- **Input — KEY CHANGE.** Replace `K=24 char one-hot concat` with a **bigram-aware encoding**:
  - Last byte one-hot (256 dims).
  - Last bigram one-hot — a hashed one-hot over the 65536 bigram space, projected to a fixed 2048-dim sketch via a frozen-random sign hash (count-sketch / SimHash, deterministic given SEED).
  - Last 8 chars one-hot concatenated (8 × 256 = 2048 dims) — keeps short-context-recency signal.
  - One candidate-byte slot (256 dims) — needed for FF positive/negative construction.
  - **Total input dim: 256 + 2048 + 2048 + 256 = 4608** (vs pass-2's 6400).
- **Training rule.** Identical to pass-2 (sum-of-sq goodness, logistic loss, hard-neg refresh).
- **Readout.** Pass-2 ridge on concat(LN(a_2..a_5)) — feature dim 1536.

## 3. Training procedure
- Identical to pass-2 with the new input function. The hash-based bigram sketch is deterministic from SEED; bookkeeping uses a small `(256, 256, 2048)` sign tensor pre-built at init.
- Same N_STEPS = 14000, hard-neg every 500, etc.

## 4. Hyperparameters
- L = 5, WIDTH = 384, K = 8 (only the recent-8-char block; the bigram block carries the longer-range signal).
- BIGRAM_HASH_DIM = 2048.
- theta = 2.0, per-layer Adam lr = 3e-4, B = 256, N_STEPS = 14000.
- Hard-neg every 500, 50% replacement, top-K=5.
- N_fit = 80000, λ = 1.0.
- SEED honoured (also seeds the bigram sign-hash).

## 5. Expected wall time (A100-80GB)
- FF training: ~50 s (slightly cheaper than pass-2 due to smaller input).
- Ridge fit: ~25 s.
- Eval: ~75 s.
- **Total: ~160 s.** Comfortable.

## 6. Success criterion
**Diagnostic.** The number we want is val char-acc(bigram input) relative to pass-2's 0.279.
- **Lift ≥ 0.03:** input encoding bottlenecks FF; Phase 4 should add a conv-stem or hashed-n-gram encoding as default for all backbone variants.
- **Lift 0.0–0.03:** modest; encoding matters but is not the dominant axis.
- **Lift < 0:** the bigram hash loses information that the K=24 one-hot retained (probably positional). Argues for keeping the simple one-hot input through Phase 4.

## 7. Failure modes anticipated
- **Hash collisions:** 65536 bigrams → 2048 sketch dims means ~32:1 collision rate. Sign hash mitigates (cancellation), but information loss is real. Diagnostic absorbs this.
- **L2-LN after layer 1 squashes the dense input contribution:** plausible. If layer-1 goodness collapses, the diagnostic still informs us cleanly (we see the failure).
- **Hidden bug — the candidate slot stops getting injected at the right offset:** carefully unit-test `build_input` before dispatch; offsets are easy to get wrong with structured inputs.

## 8. What we will NOT do
- NOT use a *learned* embedding (that's pretrained-feel and starts to look like backprop into layer 1; also outside Phase 2 scope).
- NOT change rule, width, depth, or readout.
- NOT use BPE / subword tokenisation (rule 2).
