# Research Specification 8: Fast-Weight Programmer with Delta Rule for Byte-Level LM

**Status:** Hypothesis evaluation
**Priority:** High
**Estimated effort:** 2–3 days

---

## Hypothesis

A Fast-Weight Programmer (FWP) with the delta-rule write — `W_fast ← W_fast + outer(v_t − W_fast k_t, k_t)` — used as a constant-state autoregressive backbone for byte-level WikiText-103, achieves the 0.70 character-accuracy floor at lower joules than the softmax-attention modded-nanogpt baseline. The energy win comes from the structural absence of a KV-cache: per-token compute is O(d_k · d_v + d²) regardless of sequence length, vs. O(T · d) per step for full attention.

The novel research question is the **optimal W_fast reset cadence** on byte streams. No published guidance exists. The wrong cadence either starves long-range structure (reset too often) or causes key-interference collapse (reset too rarely). The sweep is the load-bearing experiment.

---

## Background

Schmidhuber's Fast-Weight Programmer (1992) treats a recurrent network as a generator of weight updates for an inner network. The 2021 reformulation by Schlag, Irie, and Schmidhuber showed that the delta-rule variant — a single rank-1 outer-product write per token — is mathematically equivalent to a particular form of linear attention and avoids the key-interference collapse of vanilla outer-product writes.

The state is a fixed `d_v × d_k` matrix `W_fast`, written at each step with:
```
W_fast ← W_fast + outer(v_t − W_fast k_t, k_t)
```
and read with `W_fast @ q_t`. This is local in time (uses only step-t quantities) and constant in space (independent of context length).

For byte-level text, the relevant scale is small (256-way output, d_k, d_v ≈ 128) which is well below the regime where dense attention is dominant in joules. The natural reset cadences:

- **Never:** maximum long-range memory, risks interference at very long contexts.
- **Every paragraph boundary (`\n\n`):** matches WikiText's intrinsic structure.
- **Every fixed K tokens (e.g., 256):** matches typical attention window.
- **Every token:** degenerate (no memory; reduces to feedforward).

WikiText-103 has hierarchical structure: paragraphs (~200 chars), articles (~5000 chars), and within-document entity repetition. The right reset cadence is an empirical question.

---

## What to build

**Backbone:** L stacked fast-weight blocks. Each block applies:
1. Layer-norm on `x_t`.
2. Linear projections `q_t, k_t, v_t = W_q x_t, W_k x_t, W_v x_t` (slow weights, BPTT-trained).
3. Delta-rule write: `W_fast ← W_fast + outer(v_t − W_fast k_t, k_t)`.
4. Read: `y_t = W_fast @ q_t`.
5. MLP residual.

**Slow weights** (projections, MLPs, embeddings): standard backprop with AdamW.
**Fast weights** (`W_fast`): no gradient; updated by the delta rule only.

**Reset gating:** each block has a learnable scalar `g_t ∈ [0, 1]` computed from x_t; the update becomes `W_fast ← g_t · W_fast + outer(v_t − g_t · W_fast k_t, k_t)`. This lets the model learn a soft reset rather than relying on a hard schedule. **Compare against** hard schedules in the sweep.

**Initialization:** `W_fast = 0` at the start of every sequence.

**Hyperparameters:** d_k = d_v = 128, L = 6, d_model = 256, byte vocab 256. Total slow-weight params ~10 M.

---

## First experiment (go/no-go gate)

**Goal:** identify the reset cadence that maximizes val char-acc at fixed compute, and verify that the resulting model is energy-competitive with modded-nanogpt at the 0.70 floor.

**Procedure:**

1. Implement the delta-rule FWP block with both hard (fixed K) and soft (learned gate) reset variants.

2. Run a reset-cadence sweep at fixed training budget (60 minutes A100) on WikiText-103:
   - Hard reset every {32, 256, 1024, 8192} bytes
   - Hard reset at every `\n\n` boundary
   - Never reset (per-document only)
   - Soft gate (learned)

   Total: 6 configurations.

3. For each configuration, record:
   - Val char-acc
   - Val cross-entropy
   - NVML joules
   - Per-layer fast-weight Frobenius norm trajectory (does it collapse to zero or saturate?)
   - Effective recall depth: at each position, the maximum back-distance d such that a key from position t−d retrieves its value within cosine 0.9. Average over val.

4. Take the top-2 reset configurations and rerun within the 300-second harness. Report joules to reach 0.70 if hit, or best accuracy and joules within budget if not.

5. Ablation: replace the delta rule with the vanilla outer-product write (`W_fast ← W_fast + outer(v_t, k_t)`) for the best reset configuration. Measure accuracy degradation. The gap quantifies the delta rule's contribution.

**Measurements to record:**

- Val char-acc and joules for each of the 6 reset configurations at 60 minutes
- Val char-acc and joules for top-2 configurations within the harness budget
- Fast-weight Frobenius norm trajectory for each config
- Effective recall depth for each config
- Ablation: delta rule vs. vanilla outer product, accuracy delta

---

## Go/no-go criteria

**Go (pursue further):** at least one reset configuration reaches val char-acc ≥ 0.70 within the 300-second harness, AND total joules at the 0.70 crossing are ≤ 40,000 J (≥ 20% energy reduction vs. modded-nanogpt baseline).

**No-go:** no reset configuration crosses 0.70 within the harness budget, AND best val char-acc within budget is below 0.65.

**Borderline (best 0.65–0.70):** the delta-rule FWP is on a viable curve but undersized. Increase L to 8 and d_model to 384 and rerun the best reset configuration only. If it crosses 0.70 with joules ≤ 50,000, treat as Go.

The most likely failure mode is fast-weight norm collapse: if `||W_fast||` decays toward zero throughout training, the delta rule is over-correcting and the model is effectively memoryless. Check this in the per-layer norm trajectory before declaring no-go.

A second failure mode is **per-layer cadence mismatch**: lower layers may benefit from short-cadence resets (local n-gram structure) while higher layers benefit from long cadences (document-level structure). If the homogeneous-cadence sweep produces no clear winner, try a per-layer cadence schedule before giving up.

---

## What a positive result means

A positive result puts a constant-state autoregressive architecture on the leaderboard, with a clear extension path: hierarchical fast weights (a second `W_fast` updated at slow cadence on top of the per-token one) could extend the recall depth further. The key scientific question after go/no-go is: **does fast-weight norm correlate with the model's effective context window, and how does that correlation evolve over training?**

A secondary line: the FWP is compositional with the chunker (Spec 3). The chunker's surprise stream is shorter than the byte stream, so an FWP-delta backbone on the surprise stream gets both constant state AND reduced effective sequence length. See Spec 12 (chunker + FWP hybrid).

---

## What a negative result means

A negative result on byte-level WikiText-103 is informative: it tells us that the delta-rule write capacity at d_k = 128 is insufficient for the variety of keys needed to track byte-level structure, even with optimal reset cadence. The next investigation would be hierarchical fast weights (a learned key-slot allocator) rather than larger flat fast weights.

If the failure is at the norm-collapse mode, increasing the learning rate of the projections or using a Wasserstein-style normalization on the keys could rescue it — but that is research engineering, not a 300-second submission.

---

## Resources

- Paper: Schmidhuber, 1992 — "Learning to control fast-weight memories"
- Paper: Schlag, Irie, Schmidhuber, ICML 2021 — "Linear Transformers Are Secretly Fast Weight Programmers" — https://arxiv.org/abs/2102.11174
- Repository stub: `cybertronai/schmidhuber-problems`, branches `fast-weights-unknown-delay`, `linear-transformers-fwp`
- DeltaNet reference implementation: https://github.com/sustcsonglin/flash-linear-attention
- Baseline to beat: modded-nanogpt, 51,704 J, val char-acc 0.7374
- Harness: 300-second wall-clock, A100-80GB, NVML joule measurement
