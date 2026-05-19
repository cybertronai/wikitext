# P2-A — Random Projection Floor

**Phase.** FF investigation Phase 2 (diagnostics). **Axis varied.** None — the FF rule is *removed* entirely. **Purpose.** Establish the "random projection + ridge readout" floor; everything FF does later must beat this.

## 1. Hypothesis
The pass-2 FF stack reaches 0.279 val char-acc. If a stack of **frozen-random Gaussian layers** with the *same* ridge readout reaches ≈ 0.279 as well, FF is contributing zero representational value — only the ridge is doing the work. This is the single most load-bearing missing measurement from the prior survey: it tells us whether the entire FF investigation has a representational hypothesis to chase or is shadowboxing the readout.

## 2. Model
- **Backbone.** Identical to pass-2 (`FFStack`): 1 input layer + 4 hidden layers, ReLU, no bias, L2 normalisation between layers. Width 384.
- **Key change.** **All 5 layers are frozen-random** (Kaiming init, then `requires_grad_(False)` on every layer). No FF training loop. No optimiser. No negative sampling.
- **Readout.** Ridge regression — identical to pass-2: closed-form solve of `W = (Phi^T Phi + λI)^-1 Phi^T Y` on `Phi = concat(LN(a_2..a_5))` (dim 1536), `Y = one-hot next byte`, λ = 1.0, N_fit = 80000.

## 3. Training procedure
1. **Init** (~1 s). Random init of 5 FC layers, all frozen.
2. **Skip FF phase.** No 14k-step training loop. Go directly to ridge fit.
3. **Ridge fit** (~30 s). Same as pass-2 — sample 80000 (context, byte) pairs, extract phi via frozen stack, solve normal equations on GPU.
4. **Eval** (~80 s). Same `FFRidgeCharModel` as pass-2.

## 4. Hyperparameters
- All architecture HPs match pass-2 exactly (width 384, K=24, 5 layers).
- N_fit = 80000, λ = 1.0.
- SEED honoured for layer init + ridge sample selection.

## 5. Expected wall time (A100-80GB)
- Init + ridge fit + eval: ~110 s total (no FF phase). Comfortable headroom.

## 6. Success criterion
This is a **diagnostic**, not a candidate submission. The number we want is the val char-acc, not a pass/fail.
- **Reading 1 — random ≈ pass-2 (within 0.02):** FF is not learning useful features at this width/depth/budget. The investigation pivots toward architecture and capacity (Phases 4 and 7) rather than rule variants (Phase 3). Phase 3 priors weaken.
- **Reading 2 — random < pass-2 by ≥ 0.04:** FF *is* adding signal. Phase 3 rule variants are worth running.
- **Reading 3 — random > pass-2:** the ridge dominates and FF actively interferes. Concerning; would re-frame Phase 4 around "what backbone helps the ridge."

## 7. Failure modes anticipated
- **Random projections happen to be very good.** Likely outcome at width 384 / depth 5 is val acc in the 0.20–0.27 range, depending on the LN behaviour. Either way the comparison is informative.
- **Ridge underfits 1536 → 256.** Same risk as pass-2; mitigated by λ = 1.0.

## 8. What we will NOT do
- NOT touch the FF rule (rule variants are Phase 3).
- NOT change the readout (readout variants are Phase 6).
- NOT change architecture or capacity (those are Phases 4 and 7).

---
This run takes ~$0.62 of Modal and produces the most important single number in the diagnostic phase.
