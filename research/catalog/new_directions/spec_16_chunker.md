# Research Specification 16: Schmidhuber Chunker — Hierarchical Surprise-Gated Byte LM

**Status:** Hypothesis evaluation, **D1 diagnostic is Phase 0 gate**
**Priority:** High (largest theoretical energy bet in RESEARCH_DIRECTIONS catalog)
**Estimated effort:** 2 days, gated by a ~5-minute Modal diagnostic

---

## Hypothesis

A two-level Schmidhuber chunker (low-level **automatizer** RNN/transformer + high-level **chunker** that consumes only surprise events) trained from scratch on bytes reaches val char-acc ≥ 0.70 within 300 s on A100-80GB at training energy **≤ 30 kJ**, the largest energy reduction among all current candidates (~40% below `lwta_k2`).

The bet: WikiText byte streams are extremely compressible — letter trigrams and English n-grams are near-deterministic. After warm-up, a small automatizer should predict the next byte at ≥ 50% accuracy on its own (the empirical surprise rate is the D1 diagnostic). The chunker — the expensive part — then receives roughly **half** as many training steps and operates on a shorter effective sequence, compounding to substantial energy savings.

---

## Background

Schmidhuber 1991/1993 ("Neural Sequence Chunker", "Learning Complex, Extended Sequences Using the Principle of History Compression"). Architecture:

- **Automatizer** `L`: small autoregressive net that predicts the next byte at every position. Trained with cross-entropy on all positions.
- **Surprise gate**: position `t` is a *surprise* if `P_L(true_byte_t | context) < τ`. Threshold τ is a hyperparameter (typical: 0.1).
- **Chunker** `H`: larger autoregressive net that consumes **only surprise bytes** (with their original-stream positions encoded). It predicts the *next surprise byte* given the prior surprise history.
- **Output combiner at inference:** at non-surprise positions, output `argmax P_L`. At surprise positions, output `argmax P_H` (chunker prediction).

**Why energy can drop:**

- The automatizer is tiny and trains on every byte but at low per-step cost.
- The chunker — the bulk of compute — sees only the *fraction-of-bytes-that-are-surprises* = `p_s`. Its effective sequence is `p_s · N`. If `p_s ≈ 0.4` and the chunker is the dominant cost, total training compute drops by ~60% relative to a single big model on all bytes.

**The whole architecture only works if `p_s` is genuinely low.** This is D1.

---

## Phase 0: D1 surprise-rate diagnostic (gating)

**Goal.** Empirically measure `p_s(τ)` for a small transformer automatizer after warm-up. This determines whether Phase 1 is worth running.

**Procedure (~5 min Modal time, no submit.py needed — diagnostic only):**

1. Implement a tiny submission `research/catalog/new_directions/chunker_d1/d1.py` that:
   - Trains a **2-layer, 128-d, seq_len=512** transformer on WikiText train bytes for **60 seconds** under the standard harness.
   - On the last 1M training bytes after warm-up, computes:
     - The cumulative distribution of `P_L(true_byte | context)` per byte.
     - The surprise rate `p_s(τ) = fraction of bytes where P_L(true_byte) < τ` for τ ∈ {0.01, 0.05, 0.1, 0.3, 0.5}.
   - Dumps a `d1_report.json` with the surprise-rate-vs-τ table.

2. Dispatch via `python submit.py research/catalog/new_directions/chunker_d1/` (no need for a `train()` returning a CharModel — implement a dummy CharModel that records the diagnostic and predicts space; **the submission will DQ on accuracy**, that's expected and fine, we only care about the dumped JSON).

   Alternative path that avoids DQ noise: bypass submit.py and run the diagnostic directly with `task.py`-honoring code on a Modal A100 via a small custom script. Agent's choice.

**D1 gate criteria:**

- **Pass:** `p_s(τ=0.1) ≤ 0.50` AND `p_s(τ=0.3) ≤ 0.70` after warm-up. Proceed to Phase 1.
- **Borderline (`p_s(τ=0.1)` in [0.50, 0.65]):** marginal. Phase 1 risk increases — proceed only with reduced chunker capacity, since the chunker may not get a meaningful step-count reduction.
- **Fail (`p_s(τ=0.1) > 0.65`):** ABORT. The chunker degenerates to "a big model that processes most bytes anyway." Document the failure in `research/forward-forward-deep`'s parent catalog (chunker is not viable on this corpus at the chosen automatizer size). **No Phase 1 work in this case.**

---

## Phase 1 — Chunker submission (conditional on D1 Pass)

**Submission:** `submissions/chunker/submission.py`.

**Automatizer `L`:**
- 2-layer transformer, d=128, heads=4, seq_len=512.
- Standard byte embed → RMSNorm → CausalSelfAttention → MLP → output head over 256 bytes.
- Trained jointly with `H` (one shared step) on all positions via cross-entropy.

**Surprise gate:**
- Compute `P_L(true_byte_t)` per position during the forward pass.
- A position is a surprise iff `P_L(true_byte) < τ`. Use τ from D1's chosen operating point (probably τ=0.1).
- The gate output for each position is a boolean. **Trainable τ is out of scope for Phase 1** — fix it to the D1 value.

**Chunker `H`:**
- 6-layer transformer, d=384, heads=6.
- Input: sequence of surprise byte embeddings, concatenated with the **original-stream position** (sinusoidal positional encoding using the absolute byte index, not the surprise-stream index — preserves how far apart consecutive surprises are).
- Predicts the next surprise byte given the prior surprise history.
- Trained with cross-entropy only at surprise positions.

**Joint training loop (per batch):**
1. Forward `L` on the byte batch → per-position logits + losses + per-position surprise mask.
2. Gather surprise positions, feed their embeddings + positions into `H`. Compute `H`'s loss only at next-surprise targets.
3. Total loss = `loss_L + α · loss_H` with α=1.0 initially; sweep α ∈ {0.5, 1, 2} in remediation if needed.
4. Backward and step (Muon for 2-D, AdamW for 1-D and embeddings).

**Streaming inference (CharModel.predict):**
- Buffer the last 512 bytes as `L`'s context; the last K=32 surprise bytes (and their positions) as `H`'s context.
- On `predict()`: run `L` on current context → get `P_L`.
- If `max(P_L) ≥ 1 − τ`, return argmax of `P_L` (non-surprise prediction — fast).
- Else (surprise predicted): run `H` on the surprise buffer + positional info → return argmax of `P_H`.
- On `observe(c)`: append `c` to `L`'s buffer. Compute `P_L(c)` — if `< τ`, also append `(c, position)` to `H`'s surprise buffer.

**Sizing.** Total params ≈ `L`(~0.5M) + `H`(~10M) = ~10.5M, much smaller than modded-nanogpt (~36M). The bet: chunker only runs on a fraction of positions, so the larger H is amortized.

---

## First experiment (go/no-go gate for Phase 1)

**Goal:** confirm joint training reaches the floor at substantially lower energy than 46 kJ.

**Procedure:**

1. Implement and submit.

2. Record val char-acc, training joules, training duration, plus:
   - **Realized surprise rate** during training (should match D1's estimate).
   - **Per-mode accuracy:** when the prediction came from `L` alone vs. from `H`.
   - **Chunker step count** as a fraction of total steps.

3. If val char-acc < 0.70:
   - **Remediation A:** lower τ from 0.1 to 0.05 (fewer surprises → smaller chunker workload but `H` sees only the hardest bytes; if `H` learns these, accuracy may recover).
   - **Remediation B:** larger chunker (d=512, 8 layers) at τ=0.1. Trades energy for accuracy.
   - One remediation only.

---

## Go/no-go criteria

**Go:** val char-acc ≥ 0.70 AND training joules ≤ 35 kJ. Chunker is the new leaderboard top; mechanism is genuinely novel relative to attention/SSM trend.

**Soft-pass:** val char-acc ≥ 0.70 AND joules in (35 kJ, 46 kJ]. Beats baseline; doesn't beat LWTA. Hierarchical architecture finding is still publishable as evidence that surprise-gating works at byte level — the first such demonstration on a modern benchmark.

**No-go:** val char-acc < 0.70 after one remediation. Two diagnostic paths:
- If automatizer accuracy alone (no chunker contribution) is already ≥ 0.6 (i.e., `L` does most of the work), the chunker is failing to lift performance on the hard cases. **The hard cases are exactly the hard cases** — surprise positions are where deep history matters; `H` may be too small. Document.
- If automatizer accuracy alone is < 0.4, the joint training is harming `L`. The surprise gate signal is too noisy. Document.

---

## Phase 2 (conditional on Go)

1. **Chunker + LWTA in `H`.** `H` is the bulk of params; LWTA on its MLP compounds savings (matches H5 in the catalog).
2. **Trainable τ** with a small overhead penalty term to prevent τ→0 collapse. Removes the hyperparameter sensitivity.
3. **Three-level chunker** (chunker on top of chunker on top of automatizer). Only justified if Phase 1's `H` itself shows a high surprise rate at its own level — measure first.

---

## What a positive result means

The first successful **hierarchical surprise-gated architecture** on a modern char-LM benchmark. Demonstrates that the 1991 Schmidhuber idea has real legs on natural language at scale (byte-level WikiText is "scale" in the relevant sense). The energy story isn't just paper-claim; it's NVML-measured joules on a leaderboard.

For the program, this opens chunker-FWP hybrid (Spec 12), three-level chunker, and **surprise-gated routing for any architecture** (LWTA-only-on-surprise, attention-only-on-surprise) as natural follow-ups.

---

## What a negative result means

Two distinct negative results, each informative:

- **D1 fails** (surprise rate too high): the corpus isn't compressible enough at byte level with a small automatizer to make chunking worthwhile. Either a bigger automatizer (but then we lose the energy win) or a different corpus would be needed. Definitive: chunker is a dead direction on WikiText-103 byte-level.
- **D1 passes, Phase 1 fails:** the architecture is sound in principle but joint training of L+H doesn't converge in 300 s. The chunker idea remains alive but requires more careful training (curriculum: pretrain `L` for half the budget, freeze, train `H` on the rest). Document and propose Phase 1.5.

---

## Resources

- Schmidhuber 1991 — "Neural Sequence Chunker" (TR FKI-148-91)
- Schmidhuber 1993 — "Learning Complex, Extended Sequences Using the Principle of History Compression" — Neural Computation 4(2)
- Catalog: `research/catalog/RESEARCH_DIRECTIONS.md` § A1 (parent reference)
- Hybrid: `spec_12_chunker_fwp_hybrid.md` (this spec is the standalone counterpart)
- Baseline to modify: `submissions/modded_nanogpt/` (for the per-block primitives only — overall architecture is new)
- Current leader: `submissions/lwta_k2/` at 46.1 kJ / 0.7146
- Harness: 300 s, A100-80GB, NVML joules, val char-acc ≥ 0.70 on 60K val chars
