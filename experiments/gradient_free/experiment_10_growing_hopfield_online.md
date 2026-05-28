# Experiment 10: Online-Grown Hopfield Memory (Memorizing-Transformer-Style)

## Hypothesis
Replacing the *frozen, init-time-built* Hopfield memory bank with one that **grows during training** — appending the most-recently-seen (key, value) pairs every K steps and discarding the oldest, FIFO — improves val acc over the frozen-init bank because the memory tracks the encoder's learned representation as the encoder co-trains. Tests whether the Hopfield win is mainly from "any retrievable memory" or specifically from "memory aligned to current encoder."

## Motivation
The frozen-K_mem design of `hopfield_layer` is the simplest version; Wu 2022 "Memorizing Transformers" specifically grows memory over time. The prior Hopfield run noted that the encoder ends up *adapting to* its random-init K_mem geometry — i.e., the model paid an alignment cost. An online memory removes that cost. The empirical question: is the alignment penalty worth the simplicity, or does Wu-style online memory dominate?

This is the natural extension of the Hopfield direction, and shares its paradigm-B character. Builds directly on the only winning prior experiment.

## Method
Same arch as `hopfield_layer`. Change: K_mem and V_mem are **circular buffers** with capacity M=4096. Every K=50 training steps:
1. Sample 256 fresh (context, next-byte) pairs from the train batch.
2. Forward through current encoder (with no_grad) to get K_new (256, d), V_new = embed(next_bytes) (256, d).
3. Append to K_mem/V_mem at the head pointer; wrap around (FIFO eviction).

The buffer remains a `register_buffer` (no gradients flow into it) but is *updated* by an explicit in-place op outside the gradient pass — a true Hebbian-style local write rule for the memory.

## Memory-Movement Analysis
- M=4096, refresh 256 patterns every 50 steps. Total writes: (n_steps / 50) × 256 × d × 2 B = 2150/50 × 256 × 768 = ~8.5 MB total writes across training. Negligible.
- The K=50 refresh is cheap because the encoder forward for 256 contexts is one minibatch worth of compute, ~0.5% of one training step.
- Forward through Hopfield layer per training step unchanged from exp 11.
- Total energy delta vs `hopfield_layer`: <1% extra. The hypothesis must show the acc lift makes that <1% worth it.

## Setup
- Identical to `hopfield_layer` except for the online refresh.
- Refresh hyperparameters: K (refresh interval) ∈ {25, 50, 200}, refresh batch size 256.
- Baseline: `hopfield_layer` (40.2 kJ / 0.7293).

## Procedure
1. `cp -r submissions/hopfield_layer submissions/hopfield_online_mem`
2. Add an `_refresh_memory(model, train_bytes, n_new=256, ctx_len=256, head_ptr=…)` helper that does one encoder forward + scatter-write.
3. In the training loop, every K steps call `_refresh_memory`.
4. Maintain a Python-int `_mem_head_ptr` on the model so writes wrap around.
5. Submit. If results look good, A/B over K ∈ {25, 200}.

## Success Criteria
- **Strong**: val ≥ 0.735 at energy ≤ 41 kJ → online memory closes the gap to baseline modded_nanogpt acc.
- **Pass**: val ≥ 0.73, energy ≤ 41 kJ → confirms online > frozen for Hopfield.
- **Refutation**: val ≤ 0.7293 (frozen result) → frozen is good enough; encoder-alignment cost is not what's limiting Hopfield.

## Failure Modes & Diagnostics
- Memory distribution shift during training causes loss spikes: log loss at every refresh step; if a spike of >0.1 appears, refresh too aggressive — increase K.
- Memory becomes entirely "current encoder" with no diversity: log K_mem singular spectrum every 500 steps; if condition number > 1e6, memory is collapsing.
- Streaming inference at eval time: do we refresh memory during eval? Spec choice: **no** (eval uses the final-state memory from training; this matches how kNN-LM and Memorizing Transformers' "fixed at eval" recipes work). Document explicitly.

## Estimated Cost
1 Modal run primary, +1 ablation on K if first passes ≈ $0.40–$0.85.

## References
- Wu et al. 2022 "Memorizing Transformers" (ICLR) — closest practical analog
- Khandelwal et al. 2020 "Generalization through Memorization: Nearest Neighbor Language Models" (kNN-LM)
- `submissions/hopfield_layer/submission.py` and exp 01 (memory size sweep) — orthogonal axes for cross-validation
