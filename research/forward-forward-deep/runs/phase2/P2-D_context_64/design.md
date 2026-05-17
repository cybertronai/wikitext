# P2-D — Context K=64

**Phase.** FF investigation Phase 2 (diagnostics). **Axis varied.** I (context length). **Purpose.** Measure the slope of val char-acc vs FF input context. K=24 was a pass-1/pass-2 choice; the real benchmark is char-LM where K ≥ 100 is the norm for transformer baselines. We need to know whether FF *can* exploit longer context.

## 1. Hypothesis
Pass-2 was K=24, val 0.279. Char-LM signal is partly local (next-char bigram/trigram statistics) and partly distant (word-completion, syntax). A 256-byte transformer baseline gets ≥ 0.6; the 0.32 gap is large and some of it is plausibly context-length. If K=64 gives ≥ 0.03 lift, context matters and Phase 4/7 should compound K=128+ with backbone choices. If K=64 gives no lift, FF cannot extract distant-context signal at this width/depth, and longer context is not the bottleneck.

## 2. Model
- **Backbone.** 5×384 FC FF stack (identical to pass-2).
- **Input — KEY CHANGE.** **K = 64** (was 24). Input dim = (K+1) × 256 = 16640 (was 6400).
- **Training rule.** Identical to pass-2 (sum-of-sq goodness, logistic loss, hard-neg refresh).
- **Readout.** Pass-2 ridge on concat(LN(a_2..a_5)) — feature dim still 1536 (unchanged, since width is unchanged).

## 3. Training procedure
- Identical to pass-2 but with K=64. Larger input increases per-step FLOPs: layer 1 maps 16640 → 384 (was 6400 → 384), so layer-1 forward is 2.6× costlier. Layers 2–5 are width × width and unchanged. Layer 1 is frozen so it has no backward — net step cost ~1.4× pass-2.
- **N_STEPS — adjusted.** Cut from 14000 to 10000 to fit budget with the larger input.
- Hard-neg refresh every 350 steps (scaled from 500/14000).

## 4. Hyperparameters
- L = 5, WIDTH = 384, **K = 64**, INPUT_DIM = 16640.
- theta = 2.0, per-layer Adam lr = 3e-4, B = 256, **N_STEPS = 10000**.
- Hard-neg every **350** steps, 50% replacement, top-K=5.
- N_fit = 80000, λ = 1.0.
- SEED honoured. The K-byte rolling context in `FFRidgeCharModel` must update to K=64 (constant lives in the submission, not a kwarg).

## 5. Expected wall time (A100-80GB)
- FF training: 10000 × ~5 ms ≈ 50 s + hard-neg refits ~10 s → ~60 s.
- Ridge fit: ~25 s (same dim as pass-2; forward marginally slower).
- Eval: ~85 s (one slightly-bigger forward per char; same batching).
- **Total: ~190 s.** Comfortable.

## 6. Success criterion
**Diagnostic.** The number we want is val char-acc(K=64) relative to pass-2's 0.279.
- **Lift ≥ 0.03:** longer context helps. Phase 7 P7-3 / P7-4 prioritise K=128 (compounded with width).
- **Lift 0.0–0.03:** marginal. Context is not load-bearing at this depth/width.
- **Lift < 0:** FF cannot use the extra context — input becomes too sparse for sum-of-sq goodness to discriminate. Argues against any context expansion until Phase 4 picks a backbone with better-suited inductive bias (conv, recurrent).

## 7. Failure modes anticipated
- **Per-step cost worse than estimated:** if N_STEPS=10k overruns, the run DQs but the partial-train val accuracy is still measurable. Reduce to 7000 in a follow-up if needed.
- **Sparse one-hot K-input doesn't activate enough hidden units:** plausible — only 64 of 16640 input positions are non-zero. Activations may collapse. Diagnostic in itself.
- **Ridge feature dim unchanged but features less informative:** if K=64 hurts, the ridge readout (which doesn't change shape) reads off the degradation cleanly.

## 8. What we will NOT do
- NOT change width, depth, rule, or readout.
- NOT introduce position embeddings or any structured input encoding — that's a backbone change (Phase 4).
- NOT mix K=24 features with K=64 features.
