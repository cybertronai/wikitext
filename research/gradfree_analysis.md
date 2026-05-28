# What the existing results tell us about gradient-free char-LM

## The empirical landscape (passing submissions only)

| Submission | E (kJ) | acc | What learns? | "How gradient-free?" |
|---|---:|---:|---|---|
| `subset_70_mkn` | **1.06** | 0.7031 | Counts + Modified Kneser-Ney smoothing | **fully gradient-free** |
| `gpu_ngram_w31_k11` | **1.33** | 0.7050 | Counts + KN | **fully gradient-free** |
| `gpu_ngram_o14_xorfix` | 3.44 | 0.7184 | Counts + KN | **fully gradient-free** |
| `paq_mixer_v3` | 3.58 | 0.7048 | Counts + tiny SGD mixer (~880 params) | mostly gradient-free |
| `chunker_phase1_v1` | 5.92 | 0.7057 | KN + NN trained on surprise positions | hybrid |
| `lwta_k4_alpha_065` | 13.17 | **0.7382** | NN(LWTA, k=4) + KN, α=0.65 blend | hybrid + 1/k sparse grads |
| `alpha_06` | 14.73 | **0.7405** | NN + KN, α=0.60 blend | hybrid |
| `hebbian_fw_block_v2` | **22.20** | 0.7010 | 4 SGD blocks + 1 Hebbian fast-weight block | partially gradient-free |
| `mha_alpha00` | 34.92 | 0.7306 | 4-layer Muon transformer | full SGD |
| `hopfield_layer` | 40.16 | 0.7293 | 4-layer Muon + frozen Hopfield retrieval | full SGD |
| `mono_forward_v2` | 46.25 | **0.7346** | Per-block CE on probe heads | layer-local SGD |
| `lwta_k4` | 46.22 | 0.7238 | LWTA-k=4 MLP | full SGD (1/k flow) |
| `ff_pretrain_then_sgd` | 48.19 | 0.7293 | FF pre-train (provably zero signal) + SGD | full SGD (FF was dead) |
| `rf_mlp_block2` | 48.91 | 0.7345 | Random features in MLP + SGD | full SGD |
| `modded_nanogpt` | 51.70 | 0.7374 | 6-layer Muon transformer | full SGD (baseline) |

## What DQ'd or failed — and the structural lessons

- **DFA (`dfa_v1`)** → **0.0 acc.** Direct Feedback Alignment with fixed random feedback matrices cannot transmit usable signal through a 6-layer transformer on bytes. Launay-Poli's recipe does not transfer to byte-LM.
- **NoProp (`noprop_etf_v2`)** → wall-clock DQ. Diffusion-style local denoising as a *replacement* head burns the budget on the inference chain.
- **Pure Forward-Forward (`ff_pretrain_then_sgd`)** → **g_pos ≡ g_neg through 300 steps**. Random-byte-corrupt negatives are numerical noise at byte windows; FF stage updated nothing. Hinton's plausible-negatives requirement does not survive shortcutting.
- **SoftHebb stacked / single-layer** → per-channel entropy stayed pinned at `log(H)` — soft-WTA Hebbian could not break channel symmetry on one-hot byte input.
- **ESN ridge readout, kernel-ridge (RFF / Nystrom / TensorSketch / Performer)** → 0.30–0.59 acc. **Confirmed "Paradigm-A representation ceiling" ≈ 0.37**: fixed featurizer + linear readout cannot carry byte n-gram structure.
- **CTW (`ctw_d24`)** → 0.475 acc. Variable-order escape-probability arithmetic alone underperforms simple KN at the same memory.
- **uMPS / MERA / TTN / TT-HMM** → spec'd but untried. The HMM/TN family has consistently been **mis-specified** in this portfolio (AR-causality violation in MERA, fabricated TT-HMM citations, partition-function bug in uMPS for streaming).
- **NBB (Schmidhuber 1989 bucket-brigade)** → demoted, structural failure. *E[ΔW/W] = p·η − λ* has no stable equilibrium under stochastic targets.
- **Evolutionary Strategies (`es-tiny-transformer`)** → 0.19 acc at 1.9 kJ. ES on 230k-param transformer cannot get off the unigram floor in the budget.
- **PPM (pure Python `ppm-context-tree`)** → 0.63 acc with full budget. CPython is the wrong substrate; the algorithm itself is competitive (this is the `paq_mixer` family in disguise).

## Three load-bearing structural findings

1. **Stochasticity filter.** Any rule that uses "one-shot WTA" or "dissipative substance under multiplicative updates" structurally fails on byte targets (English ≈ 1.3 bits/char; top-1 is unstable). NBB proved this analytically; SoftHebb confirmed it empirically. This kills a whole family: pure Hebbian-WTA, classical bucket-brigade, single-head competitive learning.

2. **Paradigm-A ceiling ≈ 0.37.** Frozen features + closed-form linear readout cannot reach the 0.70 floor regardless of width. This kills kernel ridge, random projections + ridge, ESN + ridge as **standalone** mechanisms.

3. **n-gram counting owns the low-J band.** At **1.06 kJ / 0.703**, `subset_70_mkn` is **~50× more efficient than the baseline** and is the empirical floor that any gradient-free neural method must beat to be relevant. Currently *no* fully gradient-free neural submission does.

## What the Pareto says

The energy-vs-accuracy front is **bimodal**:

- **Low-J corner (1–15 kJ):** owned by n-grams + tiny mixers. Best is `alpha_06` (14.7 kJ / 0.7405) — n-gram + small SGD NN blend.
- **Mid-band (22 kJ):** `hebbian_fw_block_v2` is the **only partially-gradient-free deep method** that passes the floor — barely (0.7010), but the energy is striking.
- **High-band (35–50 kJ):** transformer territory. Depth-reduction (6→4 Muon layers) was the actual mechanism behind the "Hopfield PASS" that motivated half the portfolio.

The gap between *fully* gradient-free n-grams (1 kJ, 0.70) and *partially* gradient-free `hebbian_fw_block_v2` (22 kJ, 0.70) is **20×**. That gap is the territory where novel methods must compete.

---

# Novel directions worth pursuing

Filtered against the three structural findings above. Ordered by **expected information × scaling story**, not by "how likely to top the leaderboard."

## Tier 1 — strongest theoretical grounding × untried at byte-LM

### N1. Predictive Coding Networks (PCN) for byte LM
Spec exists: `research/non_nn_methods/spec_12_predictive_coding_local.md`. **Never executed.** Whittington & Bogacz (2017) and Millidge (2022) prove PCN approximates backprop with **strictly local** updates `ΔW_l = -η · e_l · x_{l+1}^T · g'(.)`, where `e_l = x_l - g(W_l x_{l+1})` is the per-layer prediction error. Two crucial properties: (a) the update is *driven by an expectation of error*, not a one-shot WTA — escapes the stochasticity filter; (b) iterative inference is the same energy descent that Hopfield uses, but the layer parameters update *during* inference (not after). The repo has nothing in this family; the closest neighbor (NoProp) is a special-case denoising version. This is the highest-information experiment available.

### N2. Cascade-Correlation byte LM with closed-form output refits
Spec exists: `spec_13_cascade_correlation.md`. **Never executed.** Fahlman-Lebiere 1990: grow network one hidden unit at a time; train each new unit to maximize `cov(unit_output, residual_error)`; freeze it; refit output by ridge regression. No chain across units. Failure mode (residual-correlation weakening) is bounded and diagnosable per-unit. Strong claim if it works: the first **constructively-grown** byte-LM that clears 0.70.

### N3. "Mono-Forward all the way down" — ridge probe heads at every block, zero global gradient
The single most surprising existing result is `mono_forward_v2` at **0.7346 / 46.2 kJ** — layer-local CE on per-block probe heads. Today the *blocks themselves* still see SGD via the probe-head gradient. **The natural ablation: replace every probe head's training with closed-form ridge against the next-byte target on detached previous-block features.** No SGD anywhere. This is a strict test of whether layer-local *byte-target* supervision (which escapes the stochasticity filter) can drive depth. Cheapest of the three Tier-1 designs to implement.

## Tier 2 — Pareto extension

### N4. PAQ8-scale context mixing (50–100 orders + bit-level mixer)
`paq_mixer_v3` showed 3.58 kJ / 0.7048 with **7 orders + 22-feature mixer**. CMIX/PAQ8 use hundreds of context models with bit-level mixing and hit ~0.12 bpc on enwik8 (≈ 0.78–0.80 char-acc territory). The mechanism is fully GPU-parallelizable now (one `torch.unique` per order). This is the direct Pareto extension of the n-gram corner and the most likely path to a **sub-5-kJ / 0.74+ submission**, displacing `alpha_06`. Not novel mechanism-wise but novel at this benchmark scale.

### N5. Hedge / Vovk aggregation of n-gram experts with regret-bounded mixing
A principled cousin of N4: maintain N expert predictors (each a context model), aggregate predictions via multiplicative-weight updates (Cesa-Bianchi & Lugosi's universal-portfolio analogue). The experts never see gradients; only the simplex over them updates. Provides a regret bound that PAQ8 lacks. Novel framing of an existing-paradigm Pareto direction.

## Tier 3 — capability demos (clearance uncertain, but high-information if attempted)

### N6. Equilibrium Propagation on a Hopfield energy
Already listed as #9 in `research/outer_aggressive_gradfree/` portfolio with LOW plausibility. The two-phase clamped/free relaxation is a known time-cap risk. *But* if it clears the floor, it's the first EP byte-LM number.

### N7. Test-time-training inner loop with no outer training
Already #3 in the outer-aggressive portfolio. Cheapest "is fast-weight reading alone enough?" falsification. Useful as a tight failure-mode test even if it fails.

### N8. uMPS-Born with log-norm carry (proper streaming AR inference)
Spec `experiment_11_umps_born_dmrg.md` exists but is unsubmitted; the prior cross-check flagged a partition-function bug for streaming AR. The fix (transfer-matrix dominant eigenvector for streaming, not learned R) is mechanical. **The one remaining tensor-network experiment that is not mis-specified.** Worth one Modal run as a definitive "can multilinear closed-form clear 0.70?" answer.

## How the new directions interact with what we've learned

| Lesson | N1 PCN | N2 CC | N3 Mono-FW-all | N4 PAQ8 | N6 EP | N8 uMPS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Survives stochasticity filter? | ✓ (error is an expectation) | ✓ (covariance) | ✓ (CE on byte target) | ✓ (counts) | ✓ (energy gradient) | ✓ (Born marginals) |
| Escapes Paradigm-A ceiling? | ✓ (features adapt) | ✓ (features grow) | ✓ | n/a (no features) | ✓ | partial (D-bound) |
| Has a "scale with data" story? | strong | weak (unit count caps) | strong | strong | weak (inner solve) | medium (D-bound) |
| Cost to first signal | medium | medium | **low** | **low** | high | medium |
| Repo gap | total | total | partial | partial | total | total |

## Recommended sequencing

1. **N3 (Mono-Forward all-the-way-down)** first — cheapest, builds directly on the only working layer-local result, isolates whether closed-form per-block supervision is enough.
2. **N4 (PAQ8 at GPU scale)** in parallel — owns the Pareto, low risk, displaces the current low-J leader.
3. **N1 (PCN)** as the high-information bet — best chance of producing a genuinely new gradient-free mechanism at byte-LM scale.
4. **N2 (Cascade-Correlation)** and **N8 (uMPS-Born)** as capability demos to close out two of the catalogued specs that have been sitting unrun for the longest.
5. **N6 EP, N7 TTT-Hebbian** only if the Tier-1 experiments produce a positive signal and we want to map the family more completely.

The single highest-leverage shift in the portfolio is the recognition that **layer-local supervision against the next-byte target** (mono_forward, and by extension PCN, and N3) is the only deep-learning local rule that has shown signal so far — every Hebbian / WTA / contrastive-goodness rule has failed structurally. That's the axis worth doubling down on.
