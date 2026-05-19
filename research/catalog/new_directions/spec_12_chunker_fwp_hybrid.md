# Research Specification 12: Hybrid — Chunker Preprocessor with Delta-Rule FWP on the Surprise Stream

**Status:** Hypothesis evaluation (depends on Spec 3 and Spec 8 outcomes)
**Priority:** High (if both prerequisites pass)
**Estimated effort:** 3–5 days (after Spec 3 and Spec 8 specifically pass their gates)

---

## Hypothesis

Composing the hierarchical surprise-gated chunker (Spec 3) with a delta-rule Fast-Weight Programmer (Spec 8) gives compounding energy savings: the chunker reduces the effective sequence length by a factor of `1/(1−p_surprise)`, and the FWP eliminates the KV-cache memory cost over what remains. The total joules to reach 0.70 char-acc is below either method alone.

The compositional claim is the load-bearing one. Two methods that each save energy could conflict on the activation/update path; the experiment measures whether they actually stack.

---

## Background

**Spec 3 (chunker)** routes only surprising bytes to a higher-level model, where "surprise" is defined by a small automatizer's predicted probability of the actual next byte falling below a threshold τ. Empirically, after automatizer warm-up on WikiText, the surprise rate plateaus somewhere in 0.3–0.6 (the Spec 3 diagnostic determines this).

**Spec 8 (FWP-delta)** uses a constant-state fast-weight matrix updated by a delta rule. There is no KV-cache; per-token compute is O(d²) regardless of sequence length.

**The composition:** the chunker's higher-level model operates on a compressed stream of length L · p_surprise (where L is the byte length). Using FWP-delta as that higher-level model gives:
- Reduced step count (chunker): 1/p_surprise fewer steps.
- Constant memory per step (FWP): no KV-cache growth.

The two savings are along different axes (step count vs. per-step memory) and should compose without conflict. The risk is a third-order interaction: if surprise positions are clustered (multi-byte rare-word starts), the FWP-delta key cache is asked to retain entries from far back in the original byte stream, which stresses its capacity. This is the spec's empirical question.

---

## What to build

**Component 1 — automatizer:** small transformer (2 layers, 128 hidden, byte vocab). Outputs P(next byte) at every position. Trained with cross-entropy. (Reuse Spec 3 implementation.)

**Component 2 — surprise gate:** position t is a surprise if `P_automatizer(byte_t) < τ`. τ chosen from Spec 3 diagnostic.

**Component 3 — FWP-delta higher-level model:** 4 stacked FWP-delta blocks (Spec 8 implementation) operating on the surprise stream. Inputs at the higher level are the byte embeddings at surprise positions, with positional encoding using the *original* byte index (so the model knows how far apart consecutive surprises are).

**Component 4 — output combiner:** at non-surprise positions, output the automatizer's prediction. At surprise positions, output the FWP-delta's prediction.

**Joint training:** automatizer trained with cross-entropy on all positions. FWP-delta trained with cross-entropy at surprise positions only (no gradient at non-surprise positions for the higher model). Slow weights of the FWP-delta updated via standard backprop; fast weights via delta rule.

---

## First experiment (go/no-go gate)

**Prerequisite:** Spec 3 must have passed its surprise-rate gate (surprise fraction ≤ 0.5 at τ=0.1). Spec 8 must have a working FWP-delta with a known good reset cadence on flat byte streams.

**Goal:** measure whether the composed model reaches 0.70 char-acc at lower energy than either method alone, and whether the FWP-delta's effective recall depth on the surprise stream is sufficient.

**Procedure:**

1. Train the automatizer alone for 60 seconds on WikiText-103 train, within the harness. Snapshot.

2. Train the FWP-delta higher-level model on the surprise stream from the snapshotted automatizer for the next ~150 seconds. Use the best reset cadence from Spec 8 — but note that "every 256 bytes" in the original byte stream is "every 256 · p_surprise surprise events," which may not be the optimal cadence in surprise-stream coordinates. **Sweep the FWP reset cadence again on the surprise stream**: every {32, 128, 512} surprise events.

3. Reserve ~30 seconds for evaluation.

4. Compare against:
   - Spec 3 alone (chunker with a vanilla transformer higher-level model) at matched 300-second budget.
   - Spec 8 alone (FWP-delta on flat byte stream) at matched 300-second budget.
   - Modded-nanogpt baseline.

5. **Diagnostic on the surprise stream's structure:** at each surprise position, record the original-byte-stream distance to the previous surprise. Plot the distribution. If the distribution is heavy-tailed (occasional gaps of >1000 bytes), the FWP's effective recall over the surprise stream must span these gaps — measure recall vs. surprise-stream gap to verify.

**Measurements to record:**

- Joules and val char-acc for: composed model (best surprise-stream reset cadence), Spec 3 alone, Spec 8 alone, modded-nanogpt baseline.
- Surprise rate empirically observed during training.
- FWP-delta fast-weight norm trajectory on the surprise stream.
- Distribution of inter-surprise gaps in original byte stream.
- Composed-model accuracy broken down: at non-surprise positions (automatizer-only) vs. at surprise positions (FWP-only).

---

## Go/no-go criteria

**Go (this is a leaderboard candidate):** composed model reaches val char-acc ≥ 0.70 within the 300-second harness, AND total joules are ≤ 80% of `min(Spec 3 joules, Spec 8 joules)` at matched val accuracy.

The second condition tests genuine composition: if the composed model just matches the better of the two alone, the composition is wasted engineering.

**No-go:** composed model below 0.70, OR composed-model joules ≥ joules of the better of Spec 3 / Spec 8 alone.

The most likely failure modes:

- **Surprise-stream recall too short:** FWP-delta on the surprise stream cannot bridge the inter-surprise gaps. Accuracy at surprise positions is below either component's standalone surprise-position accuracy. Remediation: hierarchical fast weights (two FWPs, one with short reset and one with long reset, on the surprise stream).

- **Automatizer overpowers FWP:** because the automatizer is trained on all positions, it gets more gradient updates per parameter than the FWP. Late in training, automatizer accuracy on borderline positions saturates and the FWP's contribution shrinks. Remediation: τ schedule (start permissive, tighten over training) so the FWP gets more surprise positions early.

- **Joules dominated by automatizer:** the automatizer is small and cheap per step, but it still runs at every byte. If the automatizer cost matches the FWP cost, the composition saves nothing on the dominant term. Profile and report.

**Borderline (≥ 0.70 but joules between min(Spec 3, Spec 8) and 100% of it):** composition is neutral — works but doesn't help. Report and do not pursue without architectural changes.

---

## What a positive result means

A positive result puts a multi-mechanism gradient-free-adjacent architecture on the leaderboard at substantially lower energy than any single-mechanism baseline. The next step is the 3-way composition with LWTA (Spec 9) on the FWP's slow-weight MLPs.

The key scientific question after go/no-go is: **does the chunker's surprise-position distribution change qualitatively when the higher-level model is FWP-delta vs. a transformer?** If so, the choice of higher-level model is altering what the automatizer learns to find surprising. This is a co-evolutionary dynamic worth understanding.

---

## What a negative result means

A negative result tells us that the chunker and FWP, while each useful individually, conflict on the activation/update path in a way that erases the structural compositionality argument. The specific failure mode matters: short recall (architectural fix) vs. automatizer dominance (training-schedule fix) vs. flat joule curve (the savings were always asymptotic and don't apply at 300-s scale).

Document the failure mode precisely. Future hybrid attempts (H2: FF + FWP; H5: LWTA + chunker) should check for the same failure modes before investing in the engineering.

---

## Resources

- Spec 3 (chunker): `files_extracted/spec_3_chunker.md`
- Spec 8 (FWP-delta): `new_research_directions/spec_8_fwp_delta_byte_lm.md`
- Original chunker: Schmidhuber 1991/1993
- Original FWP: Schmidhuber 1992; modern reformulation Schlag et al. ICML 2021
- Repository stubs: `cybertronai/schmidhuber-problems`, branches `chunker-22-symbol`, `fast-weights-unknown-delay`
- Baseline to beat: modded-nanogpt, 51,704 J, val char-acc 0.7374
- Harness: 300-second wall-clock, A100-80GB, NVML joule measurement
