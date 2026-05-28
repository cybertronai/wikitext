# Outer-Aggressive, Fully Gradient-Free Char-LM — Portfolio Summary

This portfolio explores **mechanistically distinct** alternatives to the current
`hopfield_layer` / `hebbian_fw_block*` submissions. Those submissions are
paradigm-B hybrids: a transformer body trained by AdamW+Muon with a single
Hopfield or Schlag-style fast-weight block grafted in. The user's mandate is
the opposite end of the spectrum — outer products / Hebbian / energy-descent
as the **dominant** learning mechanism, with **no chain-rule training anywhere**.

## Constraint reminder (from `project_wikitext_constraints.md`)
- WikiText-103 char-LM, val char-acc ≥ 0.70 on first 60K val chars
- A100-80GB, 300 s wall-clock
- Char-level scoring API (`predict() -> dict[str, float]`, argmax over 256 bytes)
- Ranked by NVML joules; baseline modded-nanogpt at 51.7 kJ; hopfield_layer at 40.2 kJ; hebbian_fw_v2 at 22.2 kJ
- Stochasticity filter (`finding_kernel_stochasticity_filter.md`): predict() must
  emit a *modal-byte-favouring* distribution, not a one-shot recalled key
- Paradigm-A representation ceiling at ~0.37 (`finding_kernel_round1_results.md`):
  fixed feature extractor + closed-form readout can NOT clear 0.70 alone

## Ranking notes

Three orthogonal axes:

- **novelty** = mechanistic distance from `hopfield_layer` + `hebbian_fw_block_v2`
- **plausibility** = realistic shot at clearing 0.70 char-acc in 300 s on A100
- **scaling story** = does it look like it would still work / scale at 100×
  params or 100× tokens?

| # | File | Mechanism | Novelty | Plaus | Scaling |
|---|---|---|---|---|---|
| 1 | `01_krotov_dam_energy_inference.md` | Krotov dense associative memory, energy-descent inference, Hebbian writes | **HIGH** | LOW | high |
| 2 | `02_srwm_local_delta_no_outer_sgd.md` | Irie SRWM with the slow projections replaced by online delta-rule | **HIGH** | MED | very high |
| 3 | `03_ttt_hebbian_inner_loop_no_outer.md` | Test-Time-Training layers with Hebbian inner loop and random outer features | **HIGH** | MED | very high |
| 4 | `04_storkey_stack_pca_projections.md` | Stack of Storkey/Hebbian associative memories with PCA/CCA-trained projections, fully closed-form | MED | LOW–MED | medium |
| 5 | `05_universal_hopfield_memory_lm.md` | Millidge UHN factorization (similarity-separation-projection) with Hebbian write + RFF separation + KRR projection | **HIGH** | MED | high |
| 6 | `06_noprop_sequence_lm.md` | NoProp adapted to next-byte LM: per-block diffusion-denoising on label embedding, no backprop across blocks | MED | MED–HIGH | high |
| 7 | `07_ff_seq_pos_neg_recall.md` | Forward-Forward over (context, next-byte) pairs with goodness = associative recall quality | MED | LOW | medium |
| 8 | `08_streaming_memorizing_lm.md` | Memorizing-Transformers idiom as the *whole* model: Hebbian write, kNN read, RFF features, ridge readout | MED | MED | high |
| 9 | `09_eqprop_modern_hopfield_energy.md` | Equilibrium Propagation on a continuous Hopfield energy so attention emerges from energy gradient and weights update locally | **HIGH** | LOW | medium |
| 10 | `10_sdm_kanerva_as_lm.md` | Sparse Distributed Memory (Kanerva) as the entire LM — random address hyperplanes, Hebbian content writes | MED | LOW–MED | medium |

## Suggested execution order (information-gain / cost)

The portfolio is breadth-first by design. Order suggestion below balances cheap
fast-failure tests against high-value claim verifications.

1. **#3 TTT-Hebbian inner loop** — cheapest first experiment (no outer training
   loop at all); falsifies the "fast-weight reads alone suffice" hypothesis fast.
2. **#6 NoProp-seq** — recent, well-documented, only paper claiming gradient-free
   training that competes with backprop on classification; LM adaptation is
   open. Highest expected information per joule.
3. **#2 SRWM with delta slow weights** — the most direct attack on the user's
   stated goal ("eliminate backprop on q/k/v/proj").
4. **#8 Streaming-memorizing-LM** — direct ablation of the `hopfield_layer`
   win: same retrieval mechanism, *no* gradient on projections.
5. **#5 Universal Hopfield Memory factorization** — clean test of the
   "similarity-separation-projection" decomposition.
6. **#1 Krotov DAM (energy-descent inference)** — capability-demo; if the
   benchmark gate is clearable by pure energy descent at all, this finds out.
7. **#4 Storkey + PCA stack** — cheap baseline; calibrates how far closed-form
   non-NN methods can be pushed.
8. **#10 SDM as LM** — VSA-adjacent, capability-demo.
9. **#7 FF-seq** — already known to be hard (FF causal LM ceiling at 0.279 per
   `finding_gradfree_family_verdicts.md`); included here as
   refined-mechanism variant.
10. **#9 EqProp-Hopfield** — most novel + most uncertain; the iterative
     inner-solve is a known time-cap risk per round-1 patterns.

## Cross-portfolio themes

- All 10 designs replace BOTH the "build random memory bank" hack of the
  existing Hopfield submission AND the SGD-trained projections of the existing
  Hebbian submissions.
- 7 of 10 (#1, 2, 3, 5, 6, 8, 10) provide a **streaming-natural** prediction
  rule, matching the `CharModel.observe()` API without a separate inference path.
- 3 of 10 (#3, 5, 6) include a closed-form readout for the *prediction head* —
  the one place where Paradigm-A's ceiling can be partially escaped because the
  feature side is itself learned (by Hebbian/energy/diffusion).
- 4 of 10 (#1, 4, 7, 9) are *capability demos* — clearing 0.70 is uncertain;
  the value is in mapping what gradient-free outer-aggressive mechanisms can
  reach at all.
- None of the 10 propose "transformer body + one Hebbian block" — every
  design either eliminates the transformer entirely or keeps a *frozen*
  random-feature backbone with no gradient flow.

## What I considered and rejected

- **Hopfield Boosting (Hofmann 2024)** — designed for OOD detection on a
  pretrained backbone, not LM-from-scratch. The "boosting" loop is over labelled
  in/out-of-distribution pairs; no obvious next-byte analog.
- **HopCPT (Hopfield Conformal Prediction over Time)** — calibration tool, not
  a generative model.
- **Modern Hopfield as attention drop-in (Ramsauer 2020 single-update)** — that
  IS the existing `hopfield_layer` submission; mechanistically identical.
- **Pure CMA-ES / NES on the FW block** — `finding_gradfree_family_verdicts`
  marks ES on 1M-param transformer dead; ES on the smaller FW projections in
  300 s is still hopeless at this dimensionality.
- **Outerproduct-only Hebbian without any normalization** — equivalent to v1
  Schlag without the sum-norm fix, mathematically degenerate.
- **Symmetric Hopfield with Storkey rule trained on raw bytes** — covered by #4
  but rejected as standalone direction; capacity is ~n/√(2 ln n) and won't
  beat unigram.
- **Echo State Networks + ridge** — exists in `reference_method_shortlist.md`;
  paradigm-A, hits 0.37 ceiling per round-1.
- **Spiking Hebbian (SoftHebb) standalone** — already refuted per
  `finding_gradfree_family_verdicts`.
- **Predictive Coding networks as full LM** — per-step inner iteration loop
  caps wall-clock; refuted in same memory file.
- **Memorizing-Transformers + backprop projections** — that's a hybrid; doesn't
  match the gradient-free mandate. (#8 is the genuinely outer-aggressive variant.)
