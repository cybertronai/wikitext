# P2-B — Layer-wise Ridge Probes

**Phase.** FF investigation Phase 2 (diagnostics). **Axis varied.** Readout (multiple, per-layer) — used as a probe. **Purpose.** Reveal whether FF actually builds hierarchy: does each successive layer carry more next-char signal than the one before, or does the stack plateau?

## 1. Hypothesis
Pass-2's ridge readout concatenates layers 2–5 and gets 0.279. That collapses any hierarchy into a single number. If we fit **one ridge readout per layer** (Phi_l = LN(a_l), 5 separate solves) and probe their accuracy individually, we learn:
- Whether layer 1 (frozen random) alone matches or exceeds the FF-trained layers (a stronger version of P2-A).
- Whether accuracy monotonically increases with depth (FF is doing what it's supposed to) or plateaus / decreases (FF features at higher layers degrade for next-char prediction).
This pattern dictates Phase 4 priors: a strongly-monotone hierarchy implies "deeper helps"; a plateau implies "don't bother with depth, work on width or backbone."

## 2. Model
- **Backbone.** Identical to pass-2 (5×384 FC FF stack, L2-LN between layers, layer 1 frozen-random, layers 2–5 trained by FF with sum-of-squares goodness + logistic loss + hard-neg refresh).
- **Training rule.** Identical to pass-2 — same 14k FF round-robin steps, same hard-neg sampling.
- **Readout — KEY CHANGE.** Fit **5 independent ridge readouts**, one per layer:
  - `W_l = solve(Phi_l^T Phi_l + λI, Phi_l^T Y)` for l = 1..5, Phi_l = LN(a_l) (dim 384), Y = one-hot byte.
  - λ = 1.0, N_fit = 80000 (same as pass-2).
- **Submission CharModel.** Use the **best-by-train-set-acc** single-layer readout for the gated 60K val accuracy (so the submission has a well-defined predict()). Print all 5 per-layer accuracies on the first 20K val chars in the run log (diagnostic) — this is the deliverable.

## 3. Training procedure
1. **FF phase** (~55 s). Identical to pass-2.
2. **Per-layer ridge fits** (~30 s total). Extract Phi_l for each l on 80000 samples, solve 5 separate normal equations. Total cost ≈ same as pass-2's single fit (forward pass dominates; the matrix solves are trivial).
3. **Diagnostic eval** (~5 s). Run all 5 readouts on a 20K-char chunk of the val stream, print the 5 numbers.
4. **Submission eval** (~60–80 s). Pick the readout with the best train-set-acc; use it as the active readout in `FFRidgeCharModel`.

## 4. Hyperparameters
- All FF HPs match pass-2 exactly.
- 5 per-layer readouts, λ = 1.0 each.
- SEED honoured throughout.

## 5. Expected wall time (A100-80GB)
- FF + 5 ridge fits + diagnostic + full eval: ~200 s. Comfortable headroom.

## 6. Success criterion
**Diagnostic, not a record candidate.** The five per-layer numbers are the artifact. Patterns we interpret:
- **Monotone-increasing across layers 1..5:** FF is building hierarchy. Phase 4 prioritises deeper backbones.
- **Layer 1 (random) ≈ best layer:** FF training is wasted. Pivot Phase 3 toward different rules or skip to Phase 4 with frozen-random + better backbone.
- **Late layers degrade vs middle (e.g. peak at layer 3):** FF over-fits to its local objective at depth. Phase 5 negative-quality work becomes higher priority.
- **All five within 0.02 of each other:** the ridge is doing all the work regardless of layer. Aligned with P2-A; rule choice unlikely to help.

The submission's val acc itself is secondary, but expected ≥ pass-2 (best-layer readout ≥ best-feature-set readout from pass-2 in many setups).

## 7. Failure modes anticipated
- **Best single-layer < concat-of-layers:** if pass-2 was 0.279 and best single-layer is 0.24, the submission will report below pass-2. That's expected and not a failure of the diagnostic — the diagnostic value is the 5 numbers.
- **Per-layer features collinear within a layer:** unlikely at width 384, but if any solve is ill-conditioned bump λ to 10.0 for that layer.

## 8. What we will NOT do
- NOT change the FF training rule (that's Phase 3).
- NOT add per-layer ridges *as an ensemble* — that's Phase 6 P6-2. Here we use them as **probes** to read off hierarchy.
