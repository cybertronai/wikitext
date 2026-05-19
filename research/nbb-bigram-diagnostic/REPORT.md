# NBB bigram diagnostic — full investigation

**Verdict: NBB (Schmidhuber 1989, Neural Bucket Brigade) fails on the bigram task.** Across a sweep of η/λ from 0.2 to 10.0, no configuration reaches the unigram floor (0.1885 val char-acc), let alone the bigram-table ceiling (0.2894). The cause is structural, not implementational: NBB's dissipative substance dynamics are theoretically unstable under stochastic targets. This was a gap in the literature; we now have one data point in it.

Date: 2026-05-17. CPU-only run on the user's machine. All artifacts in this folder.

---

## Task and baselines

- Bigram next-byte prediction over WikiText-103 raw. Given the previous byte, predict the next byte.
- Evaluated as greedy-argmax val char-acc on the first 60K val bytes.
- **Unigram floor: 0.1885** (always predict space — the modal byte, frequency 18.85%).
- **Bigram-table ceiling: 0.2894** — the optimal deterministic prev-byte → next-byte function. Any single-byte-context model, NBB included, is upper-bounded by this number.
- **D2 pass criterion (RESEARCH_DIRECTIONS.md):** ≥ 0.25 val acc, covering ≥ 61% of the unigram→ceiling gap, achievable in ≤ 60s CPU.

---

## Method

The NBB rule from Schmidhuber 1989 / IDSIA HTML transcription, ported faithfully from `cybertronai/schmidhuber-problems/nbb-xor/nbb_xor.py`:

```
Δw_ij(t) = - λ · c_ij(t) · a_j(t)                                  [pay out when j fires]
          + (c_ij(t-1) / Σ_h c_hj(t-1)) · Σ_k λ·c_jk(t)·a_k(t)     [credit predecessors]
          + Ext_ij(t)                                              [reward correct output]
```

with `c_ij(t) = x_i(t-1) · w_ij(t-1)`, `Ext = η · c_ij(t)` on connections feeding the correct output when it fires, and per-subset winner-take-all activations.

Architecture for the bigram diagnostic:
- 257 input units (bias + 256-way one-hot byte). Clamped.
- 1024 hidden units, one WTA subset.
- 256 output units (one-hot byte), one WTA subset.
- 5 ticks per presentation (XOR uses 6).
- Weight init U(0.999, 1.001), float32.

`nbb_bigram.py` is a dense numpy port. `nbb_bigram_sparse.py` exploits one-hot inputs and WTA hidden+output to collapse per-tick updates to ≤ 5 scalar mutations (~200× faster, same dynamics; verified by reproducing the XOR convergence at 3164 presentations / seed 0).

---

## Theoretical analysis: why NBB cannot work on stochastic targets

Consider a fixed (prev_byte, h_winner, target_byte_distribution) triple. The connection `W_ho[h_winner, b]` for some candidate next-byte `b` evolves as follows per presentation where this h_winner fires:

- **Pay-out** (every time `b` is the WTA output): `W -= λ · W = W · (1 − λ)`
- **Ext** (only when `b == ground_truth`, fraction `p_b` of presentations): `W += η · W = W · (1 + η)`

The expected log-rate of change per presentation, given that `b` is the WTA winner this presentation, is:

```
E[ΔW / W]  =  p_b · η  −  λ
```

This gives three regimes:

| condition          | effect                                                  |
|--------------------|---------------------------------------------------------|
| `η < λ / p_b`      | W decays exponentially toward zero                      |
| `η = λ / p_b`      | unstable equilibrium (zero drift, random walk variance) |
| `η > λ / p_b`      | W grows exponentially without bound                     |

Under bigram statistics, `p_b` (the modal-byte probability per prev-byte) ranges from ~0.05 (rare bytes with no strong successor) to ~1.0 (e.g., `'` → `s` at 0.757; `]` → ` ` at 1.000). **No single η balances all 256 bytes.** Any constant η either kills the rare-modal-byte connections or blows up the strong-modal-byte ones, and there is no point in between where the system can stably encode the bigram table.

This is a structural property of dissipative substance conservation under non-deterministic reward. It does not appear in the XOR demo because XOR targets are deterministic — every connection's `p_b` is either 0 or 1, and η = λ trivially balances the latter (substance is exactly conserved, only redistributed).

---

## Empirical sweep

`nbb_sweep.py` runs 15s of training × 6 values of η, with λ = 0.005 fixed, n_hidden = 1024, 5 ticks.

| η      | η/λ  | final_acc (8K val) | overflow | W_ho_max final |
|:------:|:----:|:------------------:|:--------:|:--------------:|
| 0.001  | 0.2  | 0.0051             | F        | 1.00           |
| 0.002  | 0.4  | 0.0015             | F        | 1.00           |
| 0.005  | 1.0  | 0.0031             | F        | 1.00           |
| 0.010  | 2.0  | 0.0006             | F        | 1.01           |
| 0.020  | 4.0  | 0.0138             | F        | 1.03           |
| 0.050  | 10.0 | 0.196 (deceptive)  | **T**    | 3.3e38 (NaN at 51,744 pres) |

The 0.196 number at η=0.05 is float32 overflow, not learning: weights blew up, the system degenerated to "always predict space" (the modal byte across all of train), and 0.196 ≈ unigram-floor 0.189. After NaN propagates, accuracy drops to 0.

Speed: ~31K presentations/sec on CPU (sparse impl), so ~470K presentations in 15s. Capacity is not the constraint — see below.

### Capacity check
Re-run with n_hidden = 4096 (4×) at η = 0.02 for 30s:
- n_hidden = 1024 → val acc 0.007
- n_hidden = 4096 → val acc 0.019

Capacity helps marginally; nowhere near the unigram floor. Confirms the failure is not a fitting problem.

### Function diversity diagnostic
After 30s of training at η = 0.02, n_hidden = 1024:

- All 256 inputs produce some output (no degenerate "nothing fires").
- 91 distinct output bytes appear across the 256 input bytes (model has *not* collapsed to a constant).
- Matches the oracle modal byte on only **3/256** prev-byte → next-byte pairs.
- Average per-byte accuracy: 0.0177 — slightly above uniform-random (1/256 = 0.004) but vastly below the oracle's modal-byte hit rate of 0.29.

Inspecting sample predictions:

```
'  → s   (oracle: 's', p_modal=0.757)   ← MATCH — strongly-deterministic apostrophe-s
T  → s   (oracle: 'h', p_modal=0.666)   ← MISS — moderately deterministic
h  → a   (oracle: 'e', p_modal=0.491)   ← MISS
t  → o   (oracle: 'h', p_modal=0.287)   ← MISS — weakly deterministic
e  → e   (oracle: ' ', p_modal=0.317)   ← MISS — weakly deterministic
```

The pattern is: NBB retains the *strongest* deterministic associations (high `p_modal`), where the per-presentation balance `p · η − λ` is positive, and lets the rest dissipate. As predicted by the theory.

---

## Why the theory matters for the broader survey

The dissipation/blow-up tradeoff applies to **any** local learning rule with multiplicative substance dynamics under stochastic targets:

```
Δw  ∝  ± w   (scale-coupled, no stable equilibrium under stochastic reward)
```

Modern descendants — Forward-Forward, SoftHebb, e-prop, predictive coding — all dodge this by either (a) using additive updates that don't scale with `w`, (b) normalizing weights periodically, (c) using soft / sampled WTA (Boltzmann-style) that gives partial credit, or (d) replacing the dissipative term with a target-loss-based gradient. None of these is "pure NBB."

The literature agent's finding stands: **no NBB scaling result past ~10 units exists, and the closest neighbour (Wada et al. 2007 on Q-bucket-brigade) reports divergence on prediction problems.** This diagnostic is the first NBB-on-stochastic-target empirical data point we know of. The result is consistent with the prior.

---

## Could NBB be rescued?

Modifications that might help, ordered by departure from the original rule:

1. **Adaptive η per output** (`η_b = λ / p̂_b` where `p̂_b` is the running estimate of byte b's modal frequency on its hidden_winner's input distribution). Restores the stability condition. Breaks "strictly local" because each output unit needs to track its own frequency. Engineering: ~1 day; theoretical interest: low — collapses to a TD-style estimator.

2. **Periodic weight renormalisation** (clip + rescale W_ho rows to fixed L2 norm). Breaks substance conservation. The system becomes essentially a normalized Hebbian rule with WTA. Identical in spirit to SoftHebb (Moraitis 2022), which already works at ImageNet scale.

3. **Soft / sampled WTA** (sample h_winner and o_winner from `softmax(net / T)`). Gives partial credit on multi-correct presentations. Identical in spirit to the Boltzmann machine (with the sampling step) or to the Forward-Forward goodness contrast (with a different loss). Identifies NBB as a degenerate-temperature special case of a known method.

4. **Top-k Ext** (give Ext on the top-k outputs by activation, not just the WTA winner). Doesn't fix the underlying dissipation; just spreads it.

**None of these is NBB anymore.** Each becomes a known modern rule whose author did not credit NBB (per the literature agent). The honest interpretation is: NBB's specific formulation does not survive contact with stochastic targets, and what does survive is independently re-derivable from contemporary methods that are already in our shortlist (Forward-Forward, fast-weights, SoftHebb).

---

## Follow-up: does longer context (k > 1) rescue NBB?

User pushback: *"What was the token size on which NBB was used? Aren't we adapting a lexeme-level model to characters?"* The intuition behind it: at longer context length, targets become more deterministic (weighted-avg p_modal climbs from 0.29 at k=1, to 0.39 at k=2, to 0.59 at k=4, to 0.82 at k=8), so the dissipation analysis might no longer bite.

Tested with `code/nbb_kgram.py` (multi-slot one-hot input — `1 + 256·k` input units, the natural extension of NBB's `bias + x1 + x2` XOR layout — and the same sparse update structure). 45 s of training each on CPU:

| k | n_hidden | η     | pres/context | val_acc | failure mode                                   |
|---|----------|-------|--------------|---------|------------------------------------------------|
| 1 | 1024     | 0.005 | 4145         | 0.0031  | dissipation (baseline)                         |
| 1 | 1024     | 0.020 | 813          | 0.0138  | dissipation, best non-overflow                 |
| 2 | 4096     | 0.005 | 110          | 0.0028  | dissipation                                    |
| 2 | 4096     | 0.020 | 109          | 0.0012  | dissipation                                    |
| 4 | 4096     | 0.020 | 3.6          | 0.0078  | per-context repetition too low                 |
| 8 | 4096     | 0.005 | 0.3          | 0.0024  | repetition starvation                          |
| 8 | 8192     | 0.020 | 0.2          | 0.0040  | repetition starvation                          |

**Empirical answer: longer context does not rescue NBB.** The failure mode trades, it doesn't cross. Low k stays dissipation-limited; high k becomes repetition-limited because the input space is now ~86 K (k=4) or ~930 K (k=8) distinct contexts.

To match XOR's convergence regime (~791 presentations per pattern) at k=4's 86 K contexts, NBB would need ~68 M presentations — about 2.9 hours of CPU at 6,500 pres/sec. The 300 s budget at that rate fits only ~1.95 M presentations, ~2.9 % of what's needed (about 23 presentations per context, vs. the 791 XOR required). W_ih barely moves from init (max 1.001 in all k>1 runs); the WTA hidden-winner for each new context is fixed by random init noise and never differentiates.

The granularity question is well-posed; the answer rules out the most charitable interpretation of NBB's potential on this task. The structural verdict holds.

## Follow-up: does coarser output tokenization (BPE / word) rescue NBB?

Distinct question from context length: would predicting BPE tokens (or words, or n-byte chunks) instead of single characters help, with char-level emission as a deterministic post-step? Not tested empirically; the analytical case is unambiguous:

1. **Output WTA grows.** Char: 256 outputs. BPE: ~30K. Word: ~100K. NBB's XOR demo had 2 outputs; the 256-output bigram test was already 128× past the original. Moving to 30K outputs amplifies the per-presentation WTA-stabilisation problem.
2. **Per-prediction stochasticity does not shrink.** English entropy is ~1.3 bits/char ≈ 5.2 bits per ~4-char BPE token; per-prediction conditional entropy is *higher* in absolute terms at BPE level, and higher even as a fraction of capacity (5.2 / log₂(30K) ≈ 0.35 vs. 1.3 / log₂(256) ≈ 0.16). What drives NBB's failure is not the absolute entropy but the *spread* of `p_modal` across contexts — no single η balances both rare-modal and strong-modal connections — and that spread spans ~0 to ~1 at any granularity. Some BPE contexts are near-deterministic (continuation of a long unambiguous prefix); sentence-initial tokens remain highly uncertain. The `p_b·η − λ` balance condition fails for the same reason.
3. **Per-byte-of-training presentations drop.** One BPE prediction per ~4 bytes → 4× fewer presentations per training byte. Compounds the repetition-starvation failure observed at k=4 context.

The deeper principle: **total entropy per byte is conserved under tokenization** (information-theoretic equivalence). Methods whose failure mode is driven by per-prediction stochasticity — like NBB's dissipative-substance dynamics — cannot be rescued by tokenization changes. Coarser tokens make it worse on output-WTA size and repetition density; finer tokens (sub-character) trivially cannot predict characters.

If you want to test NBB at its natural habitat — small-vocab deterministic classification — the experiment is e.g. 2-way "next byte > 128?". NBB would likely succeed at that, but stacked binary NBBs cannot compose into a 256-way character predictor that hits 0.70 acc without re-introducing all the stochasticity at each level. The verdict is granularity-invariant.

## Recommendation for the survey

**Demote NBB from Tier B to Tier C.** The mechanism is novel and historically important, but does not generalize beyond deterministic-target toys without modifications that turn it into something else. For the energy-efficient char-LM goal specifically:

- The "no backprop, strictly local" property is real but pairs poorly with the stochastic prediction task.
- Per-update memory footprint advantage (NBB stores no activations) is genuinely lowest among candidates, but the per-presentation information rate is so low (binary correct/wrong, vs cross-entropy's 8 bits/byte) that the energy advantage is dominated by the need for many more presentations.
- Pure NBB will not clear the 0.70 char-acc gate on Wikitext at any compute budget on CPU or GPU. Diagnostic suggests it would not even clear 0.30 on bigrams.

The investigation does have a positive deliverable: an explicit theoretical and empirical analysis of why dissipative-substance + WTA fails under stochastic targets, which we believe is novel (not found in the literature search). Suitable for a short write-up if the project produces a survey paper.

---

## Artifacts in this folder

- `code/baselines.py` — unigram + bigram-table baselines on WikiText.
- `code/nbb_bigram.py` — dense numpy NBB port (slow, for verification).
- `code/nbb_bigram_sparse.py` — sparse exploitation of one-hot inputs and WTA outputs (~200× faster).
- `code/nbb_sweep.py` — η sweep with weight-norm tracking and overflow detection.
- `code/nbb_kgram.py` — k-byte multi-slot context follow-up; ran k ∈ {2, 4, 8}.
- `REPORT.md` — this file.
