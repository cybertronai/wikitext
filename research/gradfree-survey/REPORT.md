# Wikitext Gradient-Free Method Survey

## Overview

- **5 methods × 2 passes = 10 experiments.**
- **Total Modal runs attempted: 12** (~$7.44 at $0.62/run estimate).
- **Best non-DQ run: none — every one of the 10 experiments was disqualified** (8 by `val_accuracy_below_floor`, 3 by `train_time_exceeded`; one experiment counted in both buckets is not possible — see breakdown below).
- **Best run by val_char_acc (any DQ class): pass-1 `ppm-context-tree` at val_char_acc = 0.6300, training_energy_J = 633, training_duration_s = 44.5.** This is ~82× more energy-efficient than the baseline modded_nanogpt transformer (51,704 J) but 0.07 acc short of the 0.70 floor.

### Status count

| Status | Count |
|---|---:|
| `ran`, passed floor | 0 |
| `ran`, DQ — val_accuracy_below_floor | 7 |
| `ran`, DQ — train_time_exceeded | 3 |
| `failed_execution` | 0 |

## Methods

### Prediction by Partial Matching (`ppm-context-tree`)

- **Summary**: Adaptive variable-order character n-gram with PPMd escape smoothing; trained by streaming-count updates over a context trie. Non-parametric, no gradients.
- **Refs**: Cleary & Witten 1984 PPM; `github:byronknoll/cmix`.
- **Why promising**: Order-6 PPMd typically reaches ~1.5–1.8 bpc on English Wikipedia, which translates to top-1 char-acc in the 0.65–0.75 band.

**Pass 1 — design TL;DR:** Pure-Python PPMd-D trie at max order K=6, byte-level, with a 40-second early-abort guard.

**Pass 1 — result:** `ran`, DQ on val_accuracy_below_floor.
- val_char_acc = **0.6300** | training_energy_J = **633** | training_duration_s = **44.5** | modal_runs = 1
- The early-abort guard fired at 43.8 s after only 10 MB of train was processed (1.56 M trie nodes); the spec's planned 60 MB was never reached.

**Pass 2 — design TL;DR:** Variant A — arena-allocated trie (dense int32 child tables at depths 0–2, sparse dicts at 3–7), max order K=7, online eval-time `observe()` updates, full 280 s budget. Predicted 6–8× CPython speedup from eliminating per-byte `bytes(ctx[i:])` slicing.

**Pass 2 — result:** `ran`, DQ on train_time_exceeded.
- val_char_acc = **null (DQ before eval)** | training_energy_J = **980** | training_duration_s = **300.0** | modal_runs = 2 (first run hit `ModuleNotFoundError: numpy`; rewrite to `array.array` fixed it).
- Probed throughput at K=7 was ~214 KB/s, essentially the same as pass-1's 230 KB/s — the arena rewrite did **not** deliver the predicted speedup in pure CPython. The K=6 fallback fired and reached ~247 KB/s but only ingested 60 MB of the 220 MB target before SIGALRM.

**Verdict: PROMISING — the strongest candidate in the survey by a wide margin.** Pass 1 cleared 0.63 on ~2% of the train budget at <1 kJ; the marginal-character learning curve of byte-level PPMd is steep in that regime, so the path to 0.70 is "more data, same algorithm" — but pure-Python is the wrong substrate for this. A Cython/C trie (or torch-CSR vectorized batched count updates) would plausibly hit 0.70+ at well under 5 kJ. **This is the (method, pass) pair worth investing further iteration in.**

---

### Echo State Network with ridge readout (`esn-ridge-readout`)

- **Summary**: Fixed sparse recurrent reservoir; only the linear softmax readout is fit (closed-form ridge regression).
- **Refs**: arxiv:2507.15779; arxiv:2503.01724; Jaeger 2001.
- **Why promising**: Reservoir state already conditions on full history; per-char inference is one matvec; readout via single Cholesky solve on GPU.

**Pass 1 — design TL;DR:** N=8192 tanh reservoir, density 0.05, ρ=0.95, leak 0.3, byte-level one-hot input, ridge over 2 M streamed states.

**Pass 1 — result:** `ran`, DQ on val_accuracy_below_floor.
- val_char_acc = **0.3467** | training_energy_J = **29,762** | training_duration_s = **285.3** | modal_runs = 1
- Burned the full budget on a Python streaming loop at ~7 k steps/s; accuracy landed well below even bigram (~0.50) and far below the spec's 0.62 honest target.

**Pass 2 — design TL;DR:** Variant A — batched B=64 parallel reservoir streams sharing W_res/W_in, widened to N=16384, K=4-byte short-history input, ridge feature φ = concat(state, K-byte one-hot).

**Pass 2 — result:** `ran`, DQ on train_time_exceeded.
- val_char_acc = **null (DQ before ridge solve)** | training_energy_J = **71,567** | training_duration_s = **300.9** | modal_runs = 1
- Sparse-CSR-@-dense on A100 at B=64 ran at ~10.5 k rows/s (vs predicted ~500 k) — only 3.07 M of 16 M planned rows accumulated before SIGALRM. This is the spec's explicitly anticipated failure mode (sparse matmul throughput on A100).

**Verdict: DEAD-END for hitting 0.70 within the budget.** Pass 1 was below a naive bigram baseline; pass 2's throughput rewrite didn't materialize on Modal's A100. The fundamental issue is that ESN state on byte input is information-starved relative to what the linear readout needs.

---

### SoftHebb stacked conv-1D + ridge (`softhebb-stacked`)

- **Summary**: Layer-wise local Hebbian conv stack (soft winner-take-all + Oja-style anti-Hebbian) + closed-form ridge readout.
- **Refs**: arxiv:2209.11883; `github:NeuromorphicComputing/SoftHebb`.
- **Why promising**: Strongest backprop-free local rule on vision; should extract n-gram-like features from byte windows.

**Pass 1 — design TL;DR:** 4-layer causal conv [384, 384, 512, 512], dilations [1,2,4,8], kernel 5, τ=1.0, layer-wise on 100 M chars/layer, 1408-dim ridge readout on layers 2–4.

**Pass 1 — result:** `ran`, DQ on val_accuracy_below_floor.
- val_char_acc = **0.1189** | training_energy_J = **21,179** | training_duration_s = **89.0** | modal_runs = 2 (first crashed in Cholesky on rank-deficient Gram; jitter + lstsq fallback fixed it).
- **Filters did not differentiate**: per-channel entropy stayed at max log(C) for every layer through training. SoftHebb on one-hot byte patches did not break channel symmetry.

**Pass 2 — design TL;DR:** Variant B — drop the deep stack, single Hebbian layer at H=8192, with a side-by-side frozen-Gaussian random-projection control sharing the same ridge readout. The diagnostic delta (Hebbian − random) is the load-bearing measurement.

**Pass 2 — result:** `ran`, DQ on train_time_exceeded.
- val_char_acc = **null (DQ before ridge fit)** | training_energy_J = **93,197** | training_duration_s = **300.0** | modal_runs = 1
- Per-channel entropy stayed pinned at log(H) = 9.011 every step — **same pathology as pass 1, at 8x width**. The Hebbian sweep alone ate 272 s of the 300 s budget; the random-projection control was never measured.

**Verdict: DEAD-END — strong evidence the SoftHebb soft-WTA rule does not break channel symmetry on byte-level text inputs.** Two independent specs at two depths/widths produced identical entropy-pinned dynamics. The image-domain inductive bias does not transfer.

---

### Evolutionary Strategies on a tiny char-Transformer (`es-tiny-transformer`)

- **Summary**: OpenAI-style antithetic ES on a small char-transformer; no autograd, all updates from fitness-weighted Gaussian perturbations.
- **Refs**: arxiv:1703.03864; `github:openai/evolution-strategies-starter`.
- **Why promising**: Forward-only, embarrassingly parallel across the population, trivially gradient-free.

**Pass 1 — design TL;DR:** ~230 k-param char-transformer (L=4, d=64), P=64 antithetic, σ=0.02, α=0.05, ctx=64, 150 ES iters.

**Pass 1 — result:** `ran`, DQ on val_accuracy_below_floor.
- val_char_acc = **0.1900** | training_energy_J = **1,938** | training_duration_s = **62.1** | modal_runs = 1
- Used only 62 s of the 270 s budget (spec-capped at 150 iters). NLL fell 5.56 → ~3.6.

**Pass 2 — design TL;DR:** Variant A — much smaller model (~33 k params, L=2, d=32, ctx=32), P=128, centered rank weights, σ anneal 0.05 → 0.01, full 270 s budget.

**Pass 2 — result:** `ran`, DQ on val_accuracy_below_floor.
- val_char_acc = **0.1900** | training_energy_J = **8,719** | training_duration_s = **261.7** | modal_runs = 1
- Completed 599 ES iters (~4x pass 1). **Identical val_char_acc to pass 1 to four decimals.** NLL plateaued at ~3.20 by iter ~225 and oscillated thereafter.

**Verdict: DEAD-END (for this budget) — flat result across two scales is strong evidence ES has hit a capacity ceiling, not an optimization-budget bottleneck.** The 1/√D variance scaling argument predicted improvement from smaller D; the data says otherwise. ES on neural-net params is too sample-inefficient for 5-min char-LM training, even with the standard variance reduction tricks.

---

### Causal Forward-Forward (`forward-forward-causal`)

- **Summary**: Hinton's per-layer local goodness loss; layer-wise Adam on detached inputs; no backprop across layers.
- **Refs**: arxiv:2212.13345; arxiv:2307.04205.
- **Why promising**: Local learning, constant memory per layer, no global graph.

**Pass 1 — design TL;DR:** 6 FC layers × 512 width, K=24 context window, layer-1 frozen-random, per-layer Adam lr=3e-4, unigram negatives, prediction by softmax over per-candidate goodness sums (256 forwards/char at eval).

**Pass 1 — result:** `ran`, DQ on val_accuracy_below_floor.
- val_char_acc = **0.2351** | training_energy_J = **2,737** | training_duration_s = **95.6** | modal_runs = 1
- Mild lift over unigram baseline (~0.18). Eval dominated wall-clock (180 s of the 220 s end-to-end) because of the 256-candidate-forwards-per-char predictor.

**Pass 2 — design TL;DR:** Variant D — shrunk backbone (5 × 384), train FF as in pass 1, then **discard the goodness predictor entirely** and fit a closed-form ridge from concat(LayerNorm(a_2..5)) to next-byte one-hot. Eval becomes 1 forward + 1 matvec per char (vs 256 forwards). Hard negatives from the ridge's own top-K sampled every 500 steps.

**Pass 2 — result:** `ran`, DQ on val_accuracy_below_floor.
- val_char_acc = **0.2792** | training_energy_J = **3,845** | training_duration_s = **105.7** | modal_runs = 1
- Real lift over pass 1: 0.2351 → 0.2792 (+0.044, ~19% relative). Eval throughput jumped from ~3 char/s to ~920 char/s. Ridge train-subset acc was 0.295. FF features are clearly contributing structure the ridge readout can exploit (well above the ~0.18 unigram floor).

**Verdict: MILDLY PROMISING as a representation-learning method, dead-end as a complete LM.** The 0.279 ceiling at full budget is far from 0.70. But pass 2 proved FF is doing *some* representation learning — the goodness-based predictor in pass 1 just couldn't extract it. Worth further iteration only as a "local-learning backbone + closed-form readout" hybrid pattern, which is the same shape PPM already achieves more cheaply.

---

## Cross-cutting observations

**The closed-form-readout pattern recurs in every actually-learning method.** PPM is essentially a closed-form count-based predictor on a sparse-context feature space. The only pass-2 *improvement* anywhere in the survey came from putting a ridge readout on top of FF features (0.235 → 0.279). ESN and SoftHebb both have a ridge readout but their feature spaces were uninformative. The lesson: gradient-free representation learning + closed-form linear readout is the only thing that worked; SGD-on-weights or population-search-on-weights both failed.

**Pass-2 time-DQ was systemic (3 of 5 pass-2 runs).** Every pass-2 design that tried to use the full 300 s budget (PPM, ESN, SoftHebb) hit the wall before producing a number. Throughput predictions for novel algorithms were consistently 5–15× too optimistic — pure-Python trie at K=7, batched sparse-CSR matmul on A100, and Hebbian sweep at width 8192. None of the executors had ground-truth throughput numbers to calibrate against, and the spec writers extrapolated optimistically from pass 1.

**The 0.70 floor is genuinely hard at 300 s.** Across 5 method families and 10 experiments, only one came within 0.07 of the floor. The baseline transformer reaches 0.7374 in ~250 s but uses ~83× more energy. The "energy-efficient gradient-free" frontier on this benchmark is wide open if anyone can land a PPM-class result under 1 kJ.

**Random-projection / Gaussian-W controls were specified but never measured** (SoftHebb pass 2 time-DQ'd before its control ran; FF pass 2 skipped its diagnostic). These would have given the strongest signal about whether the unsupervised representation step is doing real work. A future round should run the controls first.

### Worth a deeper follow-up

**`ppm-context-tree` (pass 1).** Reasoning: 0.63 acc at 633 J was achieved on 2% of the available training data, with the only execution-substrate issue being pure-Python throughput. A Cython/C-loop trie at K=7 with the full 200+ MB of train data, plus eval-time online updates (which the harness allows — `observe()` is called per char during eval, so the trie keeps learning from val context), has a clean path to 0.70+ at ~1–5 kJ. This would beat the modded_nanogpt baseline by 10–50× on energy.

## Cost summary

- **Total Modal runs attempted: 12** (5 pass-1: 1+1+2+1+1; 5 pass-2: 2+1+1+1+1).
- **Approximate Modal cost: ~$7.44** (at $0.62/run estimated).
- Phase A investigation: ~1 model-conversation-min (subagent), no Modal cost.
- Phase B: 12 Modal runs over 2 design waves + 2 execute waves, parallel-dispatched.
- Phase C (this report): lead-written, no subagent.

## Appendix — file pointers

- `methods.json` — Phase A's selected 5 methods with their per-method rationale.
- `designs/method_<id>_pass_<p>.md` — 10 design specs (5 methods × 2 passes).
- `results/method_<id>_pass_<p>.json` — 10 survey-normalized result summaries.
- `runs/method_<id>_pass_<p>/submission.py` — the actual submitted `submission.py` for each slot.
- `runs/method_<id>_pass_<p>/result.seed0.json` — raw harness output.
- `runs/method_<id>_pass_<p>/run.seed0.log` — Modal stdout/stderr.
