# P2-C — Width 1024

**Phase.** FF investigation Phase 2 (diagnostics). **Axis varied.** K (capacity, width). **Purpose.** Measure the slope of val char-acc vs FF stack width. Informs whether Phase 7's "push to budget" should prioritise wider layers over other axes.

## 1. Hypothesis
Pass-2 was 5×384, val 0.279. If widening to 1024 (~2.7×) gives ≤ 0.03 lift, width is not the bottleneck — Phase 7 should put effort elsewhere. If it gives ≥ 0.05 lift, width is a load-bearing axis and Phase 7 should chase ≥ 1024 with all reasonable budget cuts elsewhere.

## 2. Model
- **Backbone.** 5 FC layers, **width 1024** (was 384). Layer 1 frozen random, layers 2–5 trained by FF.
- **Training rule.** Identical to pass-2 (sum-of-sq goodness, logistic loss, hard-neg refresh).
- **Readout.** Pass-2 ridge: concat(LN(a_2..a_5)), feature dim 4×1024 = 4096 (was 1536). Closed-form solve with λ = 1.0.
- **Step count — KEY CUT.** Reduce N_STEPS to ~5000 (was 14000) so the training stays inside ~250 s with a 2.7× wider stack. Per-step cost scales ~width², so 14k × (1024/384)² ≈ 100k step-equivalents — clearly infeasible. 5k at width 1024 is roughly equivalent compute to 14k at width 384's hot-path (forward+backward).

## 3. Training procedure
1. **FF phase** (~180 s estimated). 5000 round-robin steps at B=256, width 1024. Hard-neg refresh every 250 steps (scaled from 500/14000).
2. **Ridge fit** (~30 s). N_fit = 80000, feature dim 4096, λ = 1.0. The (4096, 4096) matrix is fine on A100.
3. **Eval** (~80 s). One forward per char through wider stack — slightly slower than pass-2 but still well-batched.

## 4. Hyperparameters
- L = 5 FF layers, **WIDTH = 1024** (vs 384 in pass-2).
- K = 24, theta = 2.0, per-layer Adam lr = 3e-4.
- B = 256, **N_STEPS = 5000** (vs 14000).
- Hard-neg every **250** steps (vs 500), 50% replacement, top-K=5.
- N_fit = 80000, λ = 1.0.
- Total params ≈ 5.0 M (vs pass-2's ~1.1 M).
- SEED honoured.

## 5. Expected wall time (A100-80GB)
- FF training: 5000 steps × ~30 ms (width 1024, 2 fwd/2 bwd per step across layers 2..5) ≈ 150 s, plus ~20 s hard-neg refits → ~170 s.
- Ridge fit: ~30 s (4096² solve ≈ 5 s).
- Eval: ~80 s.
- **Total: ~280 s.** Tight but inside budget. If wall_clock_guard fires we accept partial training and the eval still runs; result is still informative for the diagnostic.

## 6. Success criterion
**Diagnostic.** The number we want is val char-acc(width=1024) relative to pass-2's 0.279.
- **Slope ≥ 0.05** (val ≥ 0.33): width is load-bearing. Phase 7 P7-1..3 prioritise width 1024 and try 2048.
- **Slope 0.02–0.05** (val 0.30–0.33): mild lift, comparable to other axes. Width is one of several knobs.
- **Slope ≤ 0.02** (val < 0.30): width is not the bottleneck. Phase 7 should focus on context length, rule, or backbone instead.

## 7. Failure modes anticipated
- **Run DQs on wall time:** if N_STEPS = 5000 still overruns, the partial-train eval still gives a useful number — interpret as "width 1024 at lower step count vs pass-2 width 384 at full step count." Note the DQ in the report but use the data.
- **Ridge solve too slow at D=4096:** unlikely on A100 (a 4096² Cholesky/solve is < 5 s); if it is, drop to a partial fit on 40000 samples.
- **OOM:** unlikely at width 1024 / B=256 on A100-80GB. ~5M params × 4 bytes × Adam state ≈ negligible.

## 8. What we will NOT do
- NOT change K, theta, the FF rule, or the readout.
- NOT widen layer 1 (it's frozen random; widening just makes more random features, costs compute, no upside without retraining).
- NOT add layers (depth scaling is implicit in Phase 4).
