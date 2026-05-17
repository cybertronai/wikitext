# Research Specification 11: Neural Bucket Brigade as Gradient-Free Local Rule

**Status:** Hypothesis evaluation
**Priority:** Medium (high-information experiment)
**Estimated effort:** 1 day (diagnostic) + 3–5 days (full port, conditional)

---

## Hypothesis

The Neural Bucket Brigade (NBB) — a strictly local update rule with no gradient signal, only per-output reward credit redistributed back along active connections — reaches val char-acc ≥ 0.25 on character bigrams within 60 s of A100 wall-clock, beating the unigram baseline (~0.16). If this floor holds, scaling NBB to a 32-byte context with a multi-subset WTA hidden layer reaches val char-acc ≥ 0.55 within the 300-second harness.

The hypothesis tests a stronger claim than any other spec in this program: that a non-gradient, biologically motivated credit-assignment rule from 1989 can produce non-trivial language-modeling behavior on natural text. A positive bigram result is a small claim; a positive 32-byte-context result is a large one.

---

## Background

Schmidhuber's Neural Bucket Brigade (1989) is a credit-assignment rule for networks with binary (or stochastic) activations. The mechanism:

1. At each time step, each unit fires (1) or doesn't (0). The activation rule is typically winner-take-all within groups.
2. Active connections (from active source to active target) accumulate "substance" — a real-valued scalar per connection.
3. When the network produces a correct output, the substance on the input-side of correct-output connections is transferred back along active connections, recursively.
4. Substance acts as the connection's effective weight on future steps.

There is no gradient. There is no backward sweep. The rule is local in time (only uses current and previous activations) and local in space (only updates active connections). It is genuinely gradient-free.

NBB has not been demonstrated on natural-text tasks. The 1989 demonstrations were XOR and toy moving-light prediction. The four open concerns from the survey:

1. **Information rate:** binary reward signal carries ~1 bit per byte vs. cross-entropy's ~8 bits. The substance redistribution may or may not amplify the signal enough.
2. **Stochastic-target stability:** XOR has deterministic targets; English bytes do not. Does NBB have a stable equilibrium under stochastic targets?
3. **Anti-GPU dynamics:** sparse activation, per-connection scalar updates — A100 tensor cores are not well matched.
4. **35-year scaling silence:** if NBB worked at scale, someone would have shown it. Either it doesn't, or no one tried — both possibilities are interesting.

The user's correction (recorded in `memory/feedback_charmodel_api.md`) clarified that the harness only uses argmax, so WTA outputs are fine — removing one historical concern.

---

## What to build

**Day 1 — bigram diagnostic:**

A minimal NBB on (byte_t-1) → (byte_t):
- 256 input units (one-hot prev byte).
- 1024 hidden units organized as 128 groups of 8 (WTA within group).
- 256 output units (WTA on output).
- Substance scalar per connection (256 × 1024 + 1024 × 256 connections).
- Substance update: on correct output, transfer α · substance back along active connections.
- Activation: WTA within each hidden group; output is argmax of output units (linear in substance).

Train on a streaming bigram pass through 100 MB of WikiText-103 train.

**Days 2–5 — full architecture (conditional on diagnostic):**

- Context window: 32 bytes (vs. 1 in the diagnostic).
- Multi-subset hidden: 64 subsets × 16 units each (capsule-style distributed coding). This is the architecture the NBB paper's "follow-up" mentioned but never demonstrated.
- GPU port using a custom CUDA kernel for substance update (the per-connection scalar update is not well-served by tensor cores; a hand-written kernel is required to fit in the budget).
- Substance update with two terms: positive reinforcement on correct outputs, decay on inactive connections (the "dissipation" term from the original paper).

---

## First experiment (go/no-go gate, Day 1)

**Goal:** measure whether NBB on character bigrams beats unigram, and identify the failure mode if it does not.

**Procedure:**

1. Implement the minimal bigram NBB.

2. Run on 100 MB of WikiText-103 train bytes, streaming. Wall-clock budget: 60 seconds.

3. Evaluate on 1 MB of val bytes. Report:
   - Top-1 char-acc
   - Distribution of NBB's predictions: what's the entropy? Does it predict only "the" / space / e / t (the unigram modes)?
   - Mean substance per connection — has it grown or collapsed?
   - Fraction of connections that ever fired (received any substance) — is the network using its capacity?

4. **Stability probe:** repeat the run with two different random seeds. NBB has non-gradient dynamics that may be sensitive to initialization. Report variance.

**Measurements to record:**

- Val char-acc on bigrams (single seed and across 3 seeds)
- Prediction entropy
- Substance distribution (histogram across connections)
- Fraction of connections active
- Wall-clock training time

---

## Go/no-go criteria

**Go (proceed to full port):** val char-acc ≥ 0.25 on bigrams within 60 s GPU, AND prediction entropy > 1.5 bits (rules out collapse to a single top byte), AND seed-to-seed variance in val acc < 0.02 (rules out chaotic dynamics).

**No-go on any of these:** the rule does not function on natural-text bigrams; do not pursue the multi-byte context. Stop with a written diagnostic of which failure mode held.

The most likely failure modes:

- **Unigram collapse:** prediction entropy near 0, val acc near 0.16. Substance dissipates from all but the most-frequent-byte output connections. Caused by the dissipation term overwhelming the reinforcement term on rare bytes. Could be patched with frequency-aware substance decay; this is research.

- **Chaotic dynamics:** seed-to-seed variance high, no stable accuracy. The substance redistribution forms positive feedback loops without a stable fixed point under stochastic targets. Fundamental issue, not a bug.

- **No-learning:** substance distribution barely changes from initialization; val acc near random (1/256). The rule is not transferring credit; check that active-connection identification is correct.

**Borderline (val acc 0.20–0.25, entropy OK, low variance):** the rule is learning but slowly. Extend the diagnostic run to 5 minutes and rerun. If acc reaches 0.25, treat as Go.

---

## Phase 2 (conditional on Go)

Only if Day-1 diagnostic passes:

1. Implement the 32-byte-context multi-subset architecture.
2. Build the CUDA kernel for substance updates (~2 days of CUDA engineering).
3. Run within the 300-second harness on full WikiText-103 train.
4. Report val char-acc and joules.

The energy story in Phase 2: NBB updates are sparse (only active connections), and the substance update is a scalar add — no matmul, no momentum, no chain rule. If the CUDA kernel achieves ≥ 50% of A100 memory bandwidth on the substance update, the total joules should be substantially below the modded-nanogpt baseline.

---

## What a positive result means

A positive bigram result (the Day-1 gate) is already publishable as the first non-trivial NBB demonstration on natural text in 35 years.

A positive Phase-2 result (≥ 0.55 char-acc within 300 s) would be a major claim: that a purely local, non-gradient rule from 1989 can do meaningful language modeling. The follow-up question is *how it scales*: NBB's substance-distribution dynamics may behave differently at large depth vs. wide+shallow. The research program that opens from a positive result is substantial.

The key scientific question after the gate is: **does NBB substance distribution converge to something interpretable** — e.g., do high-substance connections correspond to high-mutual-information byte pairs? A simple correlation analysis would answer this.

---

## What a negative result means

A negative result settles a long-standing question: NBB's local rule does not transfer to stochastic-target natural-text regimes. Note which of the three failure modes held; each has different implications:

- Unigram collapse → frequency-aware substance decay is the research path.
- Chaotic dynamics → fundamentally incompatible with stochastic targets; abandon.
- No-learning → implementation issue; the spec is not falsified.

A negative result does not invalidate the spec; it produces a clear, citable finding about the limits of local non-gradient rules.

---

## Resources

- Paper: Schmidhuber, 1989 — "Networks adjusting networks" / bucket-brigade weight credit
- Repository stubs: `cybertronai/schmidhuber-problems`, branches `nbb-xor`, `nbb-moving-light`
- Internal stub: `experiments/nbb_bigram/` (exists, may be a useful starting point)
- Baseline to beat (energy): modded-nanogpt, 51,704 J, val char-acc 0.7374
- Unigram floor: ~0.16 character-level accuracy
- Harness: 300-second wall-clock, A100-80GB, NVML joule measurement
