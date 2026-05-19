# Research directions — gradient-free methods for WikiText char-LM

Compiled from a survey of cybertronai/schmidhuber-problems (58 stubs) and cybertronai/hinton-problems (54 stubs), evaluated against the WikiText-103 character-LM leaderboard task. See `submissions/modded_nanogpt/result.json` for the baseline (51,704 J / 247 s / 0.7374 val char-acc on A100-80GB).

The 6-family taxonomy of mechanisms that survive the filter:

1. **Layer-local learning** — per-layer goodness or contrastive update; no chain across depth.
2. **Hebbian / fast-weight** — outer-product write to O(d²) matrix per token; time-local.
3. **Hierarchical / surprise-gated** — higher level only sees surprises; lower level uses bounded-truncation BPTT.
4. **Sparse activation** — 1/k of units win → 1/k of weights see gradient.
5. **Program / Levin search** — enumerate short programs in MDL order; no gradient at all.
6. **Adversarial / factorial codes** — local minimax for representation learning.

---

## Tier A — try this week

### A1. Hierarchical surprise-gated learning (chunker)
Source: `schmidhuber-problems/chunker-22-symbol`, `chunker-very-deep-1200` (Schmidhuber 1991, 1993).

A low-level automatizer RNN predicts next-char; a higher-level chunker RNN only receives bytes where the automatizer's predicted probability of the actual next byte fell below a threshold. Wikitext is highly compressible at the byte level (letter trigrams are near-deterministic), so the surprise rate after warm-up should drop ≥10× — the chunker then does BPTT over a much shorter compressed stream.

**Energy story:** Energy spent on predictable filler bytes is zero gradient flops (forward only). Automatizer's BPTT depth is bounded by truncation length (≈6-32), not corpus length. Chunker's BPTT is over the compressed stream.

**Pre-test (Day 1):** measure empirical per-byte surprise rate of a small transformer on the first 1M Wikitext bytes. If surprise plateaus above 50%, the energy gain evaporates and chunker is not worth pursuing.

**Engineering:** 1-2 days for a PyTorch GPU port with byte embeddings, batched truncated BPTT for the automatizer, online surprise gate, chunker over the surprise stream.

### A2. Fast-weight programmer with delta rule (FWP)
Source: `schmidhuber-problems/fast-weights-unknown-delay` (Schmidhuber 1992) + `linear-transformers-fwp` (Schlag/Irie/Schmidhuber 2021).

Constant-state O(d²) autoregressive inference. State is a fixed `d_v × d_k` fast-weight matrix, not a growing KV-cache. Per-token write: gated outer product `v_t k_t^T`. Per-token read: `W_fast @ q_t`. The delta-rule write `W ← W + outer(v_t - W k_t, k_t)` (5-line modification) prevents key-interference collapse on long contexts.

**Energy story:** No KV-cache. Per-token compute is O(d² + d_k·d_v) regardless of seq length. Versus softmax attention's O(seq² · d) per step, this is a structural win for long contexts.

**Engineering:** 2-3 days for a serious port. Requires byte embedding + LM head added (slow net's value head becomes vocab-projection), W_fast reset cadence sweep (full reset = no long-range, no reset = capacity saturation; try every 256 bytes, every paragraph, never), and GPU port of the manual BPTT.

**Caveat:** Same family as DeltaNet, RetNet, Mamba's S6 update. Not novel relative to modern linear attention; novelty would come from chunker-style preprocessing or NBB-style local update on top.

### A3. Forward-Forward on character text
Source: `hinton-problems/ff-aesop-sequences` (Hinton 2022).

The only sequence-native FF stub in either repo. Per-layer "goodness" (mean h²) high for real char-window substrings, low for negatives (teacher-forced wrong-last-char or autoregressive self-rollout). No backward pass across layers — each layer's update needs only its input, output, and its own positive/negative goodness.

**Energy story:** Zero activation stash, zero backward sweep. Per-layer update is one matmul + ReLU + per-sample goodness. Updates parallelizable across layers (each layer trains independently on its frozen input from the previous layer).

**Risks:** Hinton himself only hit 53% on a 30-symbol alphabet; reaching 0.70 on 256 bytes is a stretch. `predict()` requires V=256 forward passes per output byte (one per candidate). Must be batched across the eval window or it blows the 300-s wall-clock.

**Engineering:** 1-2 days. Bump vocab 30 → 256, window 10 → 32 or 64, widths 500 → ~2000, GPU port, batched candidate scoring.

### A4. Linear attention with delta rule (standalone)
Source: same as A2 but without the FWP slow-net structure.

Drop-in modification to the modded-nanogpt transformer: replace softmax attention with linear delta-rule attention. ~50 lines of CUDA/PyTorch from existing DeltaNet implementations. 4-8 hours.

**Caveat:** Probably not different enough from the baseline transformer family to win on a "novel paradigm" axis. List only because the engineering cost is so low and the energy comparison (O(T·d²) vs O(T²·d) attention) is clean.

---

## Tier B — worth a deeper read / combine

### B1. Local Winner-Take-All (LWTA / compete-to-compute)
Source: `schmidhuber-problems/compete-to-compute` (Srivastava 2013).

Drop-in activation: replace ReLU/GELU with LWTA. Hidden units in groups of k; forward only the max in each group; backprop only through the winner. Backward FLOPs and optimizer-state traffic both reduced by 1/k.

**Engineering: ~2 hours.** Swap activation function in the modded-nanogpt MLP blocks; sweep k ∈ {2, 4}. Lowest-effort experiment on this entire list.

### B2. Fast-weights associative retrieval (Ba 2016)
Source: `hinton-problems/fast-weights-associative-retrieval`.

RNN with per-sequence fast-weights matrix `A_t = λA_{t-1} + η·outer(h_{t-1}, h_{t-1})`, Hopfield-style read added to pre-activation. Slow weights still BPTT, but the *per-step memory* is O(d²) instead of a full KV cache.

**Caveat:** Repo's own reproduction is partial (38% vs paper's 98% on its toy task). Slow weights still backprop. Energy win only at d² < attention cost, i.e. d < ~250.

### B3. Linear Hebbian associator (fast-weights-rehearsal)
Source: `hinton-problems/fast-weights-rehearsal` (Hinton & Plaut 1987).

Linear associator with slow+fast weight matrices. Delta rule on `W_eff = W_slow + W_fast`. **Strongest no-backprop story in the entire survey** — one outer product per observe, zero activation storage, no backward ever.

**Risks:** Linear associator on raw context is ~bigram strength. To reach 0.70 needs (a) frozen random projection + nonlinearity then the associator on top, (b) a stack of associators with intermediate Hebbian features, or (c) a hash-of-context input. **Research-grade engineering.**

### B4. RBM + CD-1 conditional generation
Source: `hinton-problems/bars-rbm`, `encoder-40-10-40` (Hinton CD-1).

RBMs natively model `P(visible)`. With visible = `[context_window, next_byte_slot]`, sampling from `P(next | context)` gives the next-byte distribution. CD-1 update is purely local Hebbian — needs only data activations and one-step-reconstructed activations of the adjacent layer.

**Engineering:** 3-5 days. Design how to compose RBMs into a deep model and how to do conditional generation. Could just train one wide RBM on `[context, next]` and infer `P(next | context)` by clamping context.

### B5. Deep Belief Net (greedy layer-wise CD pretrain)
Source: `hinton-problems/dbn-mnist`.

Stack RBMs greedily. Each layer trained by CD-1 with the previous layer's hidden activations as its data. L separate CD-1 loops, trivially parallelizable across layers, no backward sweep through the stack.

**Mechanism is the strongest match to the repo's "backprop is inefficient in commute-to-compute" thesis.** No published RBM/DBN-only character LM that I know of, however.

### B6. Clockwork-RNN (multi-rate hidden)
Source: `schmidhuber-problems/clockwork-rnn`.

Hidden units partitioned into G modules with clock periods `T_g ∈ {1, 2, 4, 8, ...}`; module g only updates when `t mod T_g == 0`. At any timestep only ~log G modules update. Multi-rate structure matches text (char-bigram fast, word-level T=4-8, paragraph T=64+).

**Engineering:** ~1 day. GPU port + LM head + byte embeddings. Hand-designed clock schedule.

### B7. Neural Data Router (geometric attention)
Source: `schmidhuber-problems/neural-data-router`.

Geometric-distance-ordered scan attention: query `i` visits keys in order of distance, attention is a geometric distribution over "stops." Can early-exit when cumulative non-stop probability is small.

**Engineering:** ~1 week (research engineering). Stub is a 6-layer / 48-d toy.

### B8. Predictability Minimization (proto-GAN factorial codes)
Source: `schmidhuber-problems/predictability-min-binary-factors` (Schmidhuber 1992).

Adversarial: encoder maximizes reconstruction minus predictor loss; per-unit predictors minimize predicting each code unit from the others. Fixed point: factorial (independent) code.

**Use as unsupervised pretraining** for the byte embedding; the LM training proper still uses cross-entropy. Doesn't *replace* the LM loss; adds an adversarial inner loop.

### B9. Neural Bucket Brigade (NBB) — DEMOTED TO C after D2 failure
Source: `schmidhuber-problems/nbb-xor`, `nbb-moving-light` (Schmidhuber 1989).

Diagnostic D2 was run in full (see `experiments/nbb_bigram/REPORT.md`). NBB **does not clear the unigram floor** (0.1885) on bigrams at any tested η/λ ratio. Sweep results across η/λ ∈ {0.2, 0.4, 1.0, 2.0, 4.0, 10.0}: best non-overflow accuracy 0.014; only the η=10λ case crosses unigram and it does so only via float32 overflow degenerating to "always predict space."

**Root cause** (now derived analytically): per-presentation `E[ΔW/W] = p_b · η − λ` for any output connection with modal-byte probability `p_b`. Stable equilibrium requires `η = λ / p_b`, but `p_b` varies across all 256 bytes (0.05 to 1.0), so no single η balances the system. The dissipation/blow-up tradeoff is structural, not a tuning issue. See REPORT.md for the analytical argument and the empirical confirmation.

**Confirmed literature finding:** Wada et al. 2007 (Q-bucket-brigade divergence on prediction tasks) was the closest neighbour; our result is the first NBB-on-stochastic-target empirical data point. Consistent with the prior.

**Modifications that "rescue" it** (adaptive η, weight renormalisation, soft WTA, top-k Ext) all reduce NBB to a known contemporary method (SoftHebb, Forward-Forward, Boltzmann machine). Pure NBB is not viable for stochastic-target prediction; the novel-mechanism story collapses on inspection.

---

## Tier C — interesting in principle, adaptation is research-grade

### C1. Self-referential weight matrix (SRWM)
Source: `schmidhuber-problems/self-referential-weight-matrix`.

Net emits (row, col, val, gate) heads per step and rank-1-writes into its own effective recurrent weight matrix. Most natural fit for meta-learning across documents — W_fast adapts to topic/register per Wikitext article, resets at document boundary.

**Why C not B:** stub demonstrated only on 2-bit boolean meta-learning. Streaming-byte-LM mapping is non-obvious; requires redefinition of "episode."

### C2. Semilinear PM / LOCOCODE (unsupervised local feature learners)
Source: `schmidhuber-problems/semilinear-pm-image-patches`, `lococode-ica`.

Train an encoder under PM or LOCOCODE's flat-minima + L1 on byte n-gram patches. Recovers oriented filters in vision; analogous structure (character-tuple detectors) might emerge on bytes. Use as pretraining for the byte embedding; embedding becomes frozen for downstream transformer.

**Why C not B:** no evidence PM-style embeddings beat learned byte embeddings on a downstream LM. The pretraining stage is likely worse value than spending the same compute directly on the LM.

### C0. Neural Bucket Brigade (demoted from B9)
See the analysis in B9 (above) and the full diagnostic in `experiments/nbb_bigram/REPORT.md`. Kept here for completeness; not worth further investment for this task.

### C3. Levin search / OOPS / PIPE
Source: `schmidhuber-problems/levin-count-inputs`, `oops-towers-of-hanoi`, `pipe-symbolic-regression`.

Enumerate programs in MDL order; find the shortest one that fits the training set. Truly orthogonal to backprop — zero gradient, deterministic.

**Don't pursue.** Kolmogorov complexity of English bytes is ~8 bits/byte; not a Levin-search regime. Stubs find 5-instruction programs from 3 training examples; byte LM is many orders of magnitude out of distribution.

### C4. Curiosity / active-token-selection
Source: `schmidhuber-problems/curiosity-three-regions`, `subgoal-obstacle-avoidance`, `saccadic-target-detection`.

Curiosity-driven exploration could choose *which bytes to train on* — active learning over the corpus. Could meaningfully reduce training tokens needed if predictable filler is skipped.

**Why C not B:** none of these stubs target sequence prediction; all are RL/exploration in tiny gridworlds. The bridge is conceptual only. Substantial research engineering to wrap modded-nanogpt with an active-token-selection layer.

---

## Hybrid / combination directions

These compose two methods from above. Listed because the agents called out specific combinations that neither method alone would deliver.

### H1. Chunker preprocessor + FWP-delta on the surprise stream
A1 + A2. Automatizer absorbs the predictable bytes; the chunker is a delta-rule FWP over the surprise stream (constant-state, no KV cache, time-local writes). Combines surprise-gating's "skip predictable" energy win with FWP's "no KV cache" memory win.

### H2. Forward-Forward + delta-rule fast weights
A3 + A2. Layer-local FF updates between layers, fast-weight memory within each layer. Removes the two backprop-required quantities (chain across depth, and KV cache across time) simultaneously. **Research-paper territory** — no published version exists.

### H3. Random-projection feature lift + Hebbian associator
B3 with a fixed random nonlinear projection as the feature stack. Cheapest possible "no backprop ever" pipeline: random projection → ReLU → linear Hebbian update on the projected features → 256-way output. Bigram-strength baseline plus capacity from random features. Cheap to test (hours), bounds the floor.

### H4. PM-pretrained byte embedding + transformer
B8 used as embedding pretraining only. Pretrain factorial-code embedding offline under PM (cost amortized across all submissions); ship a transformer that uses the frozen pretrained embedding. Reduces gradient computation by removing the embedding from the optimizer.

### H5. LWTA inside modded-nanogpt + chunker preprocessor
B1 + A1. LWTA reduces optimizer-state traffic by 1/k; chunker reduces step count by skipping predictable bytes. Stacked savings on the existing baseline.

---

## Diagnostic experiments (cheap before committing)

These are the "first hour" tests that decide whether a Tier-A/B direction is worth the multi-day engineering.

### D1. Empirical surprise rate on Wikitext (gates A1)
Train a small transformer (e.g. 2-layer 128-d) for 30 s on Wikitext bytes. Compute per-byte `−log p(true_byte)` distribution; report the cumulative fraction of bytes below threshold τ for τ ∈ {0.01, 0.05, 0.1, 0.3}. If <50% of bytes are "surprises" at τ=0.1 after warm-up, A1 is worth the port. If >70%, drop A1.

### D2. NBB on character bigrams (gates B9) — RUN, FAILED
Run 2026-05-17 on CPU. Best non-overflow accuracy 0.014 (η/λ=4.0); η/λ=10.0 reaches 0.196 only by float32 overflow to "always predict space." Root cause: dissipative substance dynamics under stochastic targets, `E[ΔW/W] = p·η − λ` cannot be balanced across bytes with varying modal-byte probability. See `experiments/nbb_bigram/REPORT.md`. NBB demoted to Tier C.

### D3. FWP capacity at byte vocab (gates A2)
Implement the delta-rule fast-weight write at d_k=d_v=128 with W_fast reset every 256 bytes. On a 10K-byte Wikitext window, measure how many recent bytes can be recalled by key (analog of the 1992 FWP capacity experiment). If recall holds for ≥ 32 keys, A2 has enough capacity. If it collapses past 8 keys, you need hierarchical fast weights (research project).

### D4. LWTA accuracy degradation (gates B1)
Swap LWTA-k=2 into the modded-nanogpt MLP blocks; train one full submission. If val acc stays ≥ 0.70, measure joule delta. If acc drops below 0.70, try k=2 only in MLP layer 1, or restrict to the wider blocks. ~2 hours.

### D5. FF candidate-scan throughput (gates A3)
Implement just the FF inference path: forward 256 candidate next-bytes through a small FF stack at d=512, L=3, batch=8K. Measure wall-clock per output-byte-predicted. If > 1 ms per prediction, the 60K val chars eat 60 s — fine. If > 10 ms, the eval blows the harness budget; need to batch within prediction differently or drop A3.

---

## Open research questions

Questions that would shape the design even before implementation.

1. **What is the per-byte information rate of NBB's binary reward vs cross-entropy's log-prob?** Theoretical answer is ~1 bit vs ~8 bits per byte; empirical answer (with substance redistribution amplifying signal across ticks) is unknown.

2. **Does an NBB-style local rule have a stable equilibrium under a stochastic target distribution?** XOR has deterministic targets; English bytes do not. The dissipative dynamics may starve all non-modal-byte connections, collapsing to unigram. Worth a small analytical investigation.

3. **What's the optimal W_fast reset cadence for FWP on byte streams?** No published guidance. Candidates: never (capacity saturation), every paragraph boundary (`\n\n`), every 256 bytes (matches typical context), every byte (degenerates to no memory).

4. **Can multi-subset WTA architectures (capsule-style: many small WTA groups) carry distributed representations under local learning rules?** The NBB paper notes "two 2-unit hidden subsets" as a follow-up but never demonstrated it. Modern capsule and MoE literature has tools that could apply.

5. **Is there a hybrid loss that gives FF (or NBB, or RBM) cross-entropy's high information rate while preserving local updates?** E.g., a "per-layer per-byte goodness" defined as `−log P(correct_byte | layer_activations)` evaluated with a tiny frozen readout. Would lose strict layer-independence but might recover the 8-bits/byte signal.

6. **Does the chunker's surprise rate decrease monotonically through training, or plateau at the entropy floor?** Determines whether chunker depth (chunker-of-chunker) is worth implementing. Wikitext byte entropy is ~5 bits/byte → at least 5/8 ≈ 60% of bytes carry information; chunker at one level may only get a 1.6× compression, not the 10× the toy demo achieves.

7. **What does GPU energy-per-FLOP look like for WTA + scalar-update kernels vs tensor-core matmuls?** The repo's framing assumes "fewer FLOPs = less energy" but the A100 has very different efficiency at large dense matmuls vs sparse scalar ops. This number determines whether *any* of the Tier-A/B methods actually beat the baseline on joules.

8. **Combining methods: which pairs commute?** H1 (chunker + FWP) and H2 (FF + FWP) look independent. H3 (random-proj + Hebbian) and H4 (PM-pretrain + transformer) attack different stages. H5 (LWTA + chunker) is purely additive. Is there a 3-method combination that doesn't conflict on the activation/update path?

---

## Bottom line

If forced to one week of work: **D1 → A1 (if D1 passes) → A2 in parallel → B1 as Day-5 zero-effort experiment**. The chunker is the biggest genuine energy bet; FWP-delta is the cleanest "different paradigm" with a known scaling family; LWTA is the lowest-hanging fruit. If forced to one *day*: **B1 (LWTA in modded-nanogpt)** — 2 hours of work for a known-direction sanity result.

Honest expectation: no method here is `pip install && submit`, and out-of-the-box none will beat 0.7374 acc. The bet is purely on the energy axis. If the entire 1980s-2020s alternative-learning catalog cannot beat a modern transformer on energy at this scale, that is itself a publishable finding.
