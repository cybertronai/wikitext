# Research Specification 7: PPM Context Tree as Prior with a Tiny Neural Residual

**Status:** Hypothesis evaluation (anchored on empirical survey result)
**Priority:** High
**Estimated effort:** 2–4 days

---

## Hypothesis

A PPM (Prediction by Partial Matching) context tree, reimplemented to remove the CPython throughput ceiling, will close most of the gap to the 0.70 character-accuracy floor on its own; and the residual gap can be closed by a small neural correction term trained on the positions where PPM is least confident, at a fraction of the energy of any full-strength neural baseline.

The empirical pre-condition is already known: a pure-Python order-6 PPMd reached val char-acc **0.6300 at 633 J** in the local survey, processing only ~10 MB of the planned 60 MB before its early-abort guard fired. The marginal-character learning curve of byte-level PPM is steep in this regime; the bet is that "more data, same algorithm" lifts PPM past 0.70 once the substrate ceiling is removed.

---

## Background

PPM (Cleary & Witten 1984) maintains a variable-order character n-gram model with escape probabilities to lower-order contexts. At each position, the prediction is a mixture of counts at the longest matching context with backoff to shorter contexts. PPMd is the version used in modern compressors (`cmix`, `7z`, `rar`).

Order-6 PPMd on English Wikipedia typically reaches ~1.5–1.8 bits/char on cross-entropy, which corresponds to top-1 character accuracy in the 0.65–0.75 band. The survey's pure-Python implementation confirmed the lower end of this band at minimal training data.

PPM has structural advantages for the energy task:
- **No gradient computation.** Training is a single streaming pass that increments counts.
- **No optimizer state.** The trie is the entire state.
- **GPU is optional.** The work is memory-bound count updates; a Cython/C/Rust trie at 5–20 MB/s saturates the budget easily.

The argument against PPM alone is that bigram-level surface statistics cap out below 0.70 for long-range structure (named-entity copies, syntactic agreement across clauses). A tiny neural residual trained only on PPM's low-confidence positions can patch this without paying for the full byte distribution.

---

## What to build

**Part A — fast PPM core:**

A streaming order-K context trie (K=7 by default, fallback K=6 if memory pressure) implemented as a compiled extension (Cython, Rust+PyO3, or a CUDA hash-table). The trie supports two operations:
- `observe(byte)`: walk the context, increment counts at depths 0..K, age out via PPMd's escape mechanism.
- `predict(context)`: emit `P(next_byte | context)` as a length-256 distribution by mixing the longest matching context with PPMd-style escape backoff.

Target throughput: ≥ 5 MB/s on a single A100 host CPU at K=7 (vs. the survey's 230 KB/s pure-Python). This is the load-bearing engineering deliverable.

**Part B — neural residual (conditional):**

A small transformer (2 layers, 128 hidden, byte vocab) trained on positions where PPM's top-1 probability is below threshold τ_res (start with τ_res = 0.5). The neural model sees the same context window as PPM and outputs a 256-way correction `Δ_t`. The combined prediction is `softmax(log P_ppm + α · Δ_t)`, where α is a learned scalar mixing weight.

The neural residual is trained with cross-entropy on the next byte; the gradient only flows when PPM is uncertain, so the effective training set is much smaller than the full corpus.

---

## First experiment (go/no-go gate)

**Goal:** verify (a) the fast PPM core lifts accuracy past 0.65 at full data budget, and (b) the neural residual closes the remaining gap to 0.70 at energy cost dominated by PPM.

**Procedure:**

1. Implement the fast PPM core. Verify throughput on a 100 MB slice of WikiText-103 train. **Throughput must reach 5 MB/s before proceeding** — if it does not, the substrate is still the bottleneck and the rewrite needs another pass.

2. Run PPM alone at K=6 and K=7 on the full WikiText-103 train within the 300-second harness. Record val char-acc and joules. Choose the better K for the residual experiment.

3. From PPM's val predictions, extract the empirical distribution of top-1 probabilities. Choose τ_res such that ~20% of positions fall below it (this fraction is the neural model's effective workload).

4. Train the neural residual for the remaining wall-clock budget after PPM finishes. Use the PPM predictions as a fixed prior; only train the neural correction term on positions with `max P_ppm < τ_res`.

5. Evaluate the combined `softmax(log P_ppm + α · Δ_t)` model on val.

**Measurements to record:**

- PPM throughput (MB/s) at K=6 and K=7
- PPM-only val char-acc, training joules, training duration
- Distribution of PPM top-1 probabilities across val (histogram)
- Combined PPM+residual val char-acc, total training joules, total duration
- Fraction of positions where the residual changed the argmax
- Accuracy on positions where PPM was confident (top-1 ≥ τ_res) vs. where it was not

---

## Go/no-go criteria

**Go (this is a leaderboard candidate):** PPM alone exceeds 0.65 char-acc within the 300-second budget, AND the combined PPM+residual reaches ≥ 0.70 char-acc at total energy ≤ 25,000 J (less than half the modded-nanogpt baseline).

**No-go:** PPM alone is below 0.60 even with the fast core, OR the residual fails to lift combined accuracy by ≥ 0.05 absolute over PPM alone.

The most likely failure mode for the residual is that the positions where PPM is uncertain are *also* positions where a small neural model is uncertain — i.e., they are genuinely high-entropy (rare-word starts, OOV named entities). In that case, the residual cannot help without growing into a full-sized neural model, which defeats the energy argument. Check this by computing the residual's standalone calibration on the uncertainty subset before mixing.

**Borderline (PPM alone 0.65–0.70, no residual yet):** ship PPM-only as a submission first. It already establishes a new energy frontier. Pursue the residual only if PPM-only plateaus below 0.70 across multiple K and budget configurations.

---

## What a positive result means

A positive result establishes that the leaderboard's optimal point is dominated by classical sequence modeling with a minimal neural patch — a strong claim about what "language modeling" actually requires at this scale. The follow-up question is whether PPM's count table can be GPU-resident (CUDA hash-map), which would allow it to be updated and queried at byte-stream throughput inside the same kernel as the residual transformer.

The deeper scientific question after go/no-go is: **what fraction of WikiText's byte-level structure is captured by variable-order n-grams alone, and how does this fraction change with corpus scale?** This is directly answerable from the per-position PPM probabilities and connects to long-standing questions in compression-theoretic language modeling.

---

## What a negative result means

A negative result (PPM stuck below 0.65 even with full data) means byte-level n-gram statistics on WikiText-103 saturate below 0.70, and the survey's 0.63 point was already near the ceiling rather than the early part of a steep curve. This would falsify the "compression ≈ modeling" intuition at this scale.

If the residual fails (PPM > 0.65 but combined < 0.70), the conclusion is that PPM's residual entropy is concentrated at positions where a 2-layer transformer is also weak. In that case, scaling the residual would just recover a flat neural model and the decomposition adds nothing.

---

## Resources

- Paper: Cleary & Witten, 1984 — "Data compression using adaptive coding and partial string matching"
- Reference compressor: `cmix` (https://github.com/byronknoll/cmix)
- Survey result anchor: `.survey/designs/method_ppm-context-tree_pass_1.md` and `.json`
- Baseline to beat: modded-nanogpt, 51,704 J, val char-acc 0.7374
- Survey best: pass-1 PPM, 633 J, val char-acc 0.6300 (DQ on accuracy floor)
- Harness: 300-second wall-clock, A100-80GB, NVML joule measurement
