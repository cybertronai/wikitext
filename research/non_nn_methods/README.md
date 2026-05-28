# Non-Neural, Non-Backprop Universal Approximators — Portfolio Index

Breadth-first portfolio for the `~/wikitext` benchmark (A100-80GB, 300 s wall-clock,
val char-acc >= 0.70, ranked by NVML joules, baseline modded_nanogpt at 51.7 kJ).

**Selection filter.** No multilayer perceptron/transformer/CNN/RNN/SSM trained by SGD+backprop.
Methods that *use* a backbone net trained by a non-backprop rule (DFA, predictive coding) are
flagged as "borderline" with the relation to backprop spelled out.

**Hardware filter.** Each spec includes a roofline back-of-envelope. A100 ridge ~156 FLOPs/byte
(312 TF FP16 / 2 TB/s HBM). Methods landing left of the ridge are bandwidth-bound and
will burn time/joules moving parameters; methods landing right of the ridge are compute-bound
and can saturate Tensor Cores.

**What is NOT in this portfolio (already attempted, see `research/`):** PPM (best 0.63 at 633 J),
PAQ mixer, n-gram Kneser-Ney (GPU/CPU), chunker (Phase 1), LWTA drop-ins,
FWP delta-rule, Forward-Forward (`forward-forward-deep/`), Echo-State-Network + ridge,
SoftHebb stacked conv, evolution strategies on a tiny transformer, MambaByte,
pointer-sentinel, hyena.

---

## Verdict table

| ID  | Method                                                              | Family                       | Compute / BW    | Verdict |
|-----|---------------------------------------------------------------------|------------------------------|-----------------|---------|
| 01  | [Uniform MPS Born-machine LM](spec_01_uniform_mps_born_machine.md)  | tensor network               | compute-bound   | **Tier B** |
| 02  | [RFF + closed-form ridge LM](spec_02_rff_closed_form_ridge.md)      | random features / kernel     | compute-bound   | **Tier A** |
| 03  | [Polynomial / TensorSketch + ridge](spec_03_polynomial_tensorsketch_ridge.md) | polynomial kernel    | compute-bound   | **Tier A** |
| 04  | [Falkon Nyström kernel ridge LM](spec_04_falkon_nystrom_kernel_ridge.md) | kernel ridge / Nyström   | compute-bound   | **Tier B** |
| 05  | [Hyperdimensional / VSA n-gram LM](spec_05_hyperdimensional_vsa_lm.md) | HDC / VSA                 | bandwidth-bound | **Tier B** |
| 06  | [Sum-Product Network / Probabilistic Circuit LM](spec_06_sum_product_network_lm.md) | SPN / PC      | bandwidth-bound | **Tier C** |
| 07  | [Context-Tree Weighting LM](spec_07_context_tree_weighting.md)      | Bayesian mixture compressor  | bandwidth-bound | **Tier A** |
| 08  | [Dynamic Markov Compression LM](spec_08_dynamic_markov_compression.md) | bit-level FSM / count    | bandwidth-bound | **Tier C** |
| 09  | [Gradient Boosting (XGBoost) next-byte](spec_09_gradient_boosting_xgboost.md) | tree ensemble       | mixed           | **Tier B** |
| 10  | [CMA-ES on a tiny LM](spec_10_cma_es_tiny_lm.md)                    | evolutionary, covariance     | compute-bound   | **Tier C** |
| 11  | [Direct Feedback Alignment LM](spec_11_direct_feedback_alignment.md) | local error signals (borderline backprop) | compute-bound | **Tier A** |
| 12  | [Predictive Coding LM with local Hebbian updates](spec_12_predictive_coding_local.md) | PCN local rules    | compute-bound | **Tier B** |
| 13  | [Cascade-correlation constructive LM](spec_13_cascade_correlation.md) | constructive / greedy      | compute-bound   | **Tier C** |

**Recommended order of execution** (cheap fast-failure first):

1. **CTW** (spec_07) — pure counting, sub-1 kJ feasibility check; if it clears 0.70 it is the cheapest
   submission in the leaderboard's history. **Should be the very first thing run.**
2. **RFF + closed-form ridge** (spec_02) — single Cholesky on A100, claim-verification for the
   "closed-form readout is enough" hypothesis that recurs in the gradfree-survey REPORT.
3. **Polynomial / TensorSketch + ridge** (spec_03) — direct memory-movement competitor to attention;
   pure dense matmul; verifies the polynomial-kernel UAT story at LM scale.
4. **DFA on small transformer** (spec_11) — local-error-signal baseline competing with modded_nanogpt
   on the same arithmetic intensity, with the gradient-free hypothesis tested explicitly.
5. **uMPS Born machine** (spec_01) — capability demo; first uMPS on real-language LM benchmark in the
   literature, plausible 0.55–0.70 with bond dimension 256–512 in 300 s.
6. **XGBoost next-byte** (spec_09) — sanity check on whether tree ensembles can clear 0.70 at all on text.
7. **Falkon** (spec_04) — paradigm-A kernel-machine-replaces-model on byte n-gram features.
8. **HDC/VSA** (spec_05), **Predictive Coding** (spec_12), **SPN** (spec_06), **DMC** (spec_08),
   **CMA-ES** (spec_10), **Cascade-correlation** (spec_13) — lower priority / specific demos.

---

## Cross-cutting principles applied

- **Stochasticity filter** (`finding_kernel_stochasticity_filter.md`): hard-WTA methods are flagged.
  None of the proposed methods rely on a single dissipative-substance rule.
- **Tokenization invariance** (`feedback_tokenization_invariance.md`): no method here proposes BPE
  as a rescue from char-level stochasticity.
- **Char-level scoring as universal interface** (`feedback_research_framing.md`): a few methods
  (uMPS, SPN, gradient boosting) condition on a fixed window and translate to per-char predict()
  via the documented byte-marginalization wrapper.
- **Closed-form readout pattern** from `gradfree-survey/REPORT.md`: methods 2, 3, 4, 7, 11, 12
  all rely on a closed-form or per-layer-closed-form output head — the one pattern that has
  shown movement in the survey so far.
