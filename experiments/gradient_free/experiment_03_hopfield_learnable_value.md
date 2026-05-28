# Experiment 03: Hopfield with Learnable V_mem (Frozen Keys, Trained Values)

## Hypothesis
Letting V_mem be a learnable parameter (while keeping K_mem frozen) closes most of the −0.008 accuracy gap between `hopfield_layer` (0.7293) and `modded_nanogpt` (0.7374) at marginal energy cost. Rationale: K_mem defines a fixed kernel-feature lookup, V_mem is the readout — making V_mem learnable is the strict generalization, and the gradient-free aspect (the kernel of frozen keys) is preserved.

## Motivation
In exp 11, both K_mem and V_mem were frozen. K_mem must be frozen for the kernel-feature interpretation to hold (the "memory" of patterns). But V_mem is just the readout dictionary — there's no reason to freeze it. This experiment tests the partial-freeze A/B: does most of Hopfield's value come from the frozen-key retrieval, with the frozen V_mem being a residual restriction?

This is the cleanest A/B in the portfolio — it sharpens *what* the gradient-free component is contributing.

## Method
Same arch as `hopfield_layer`. Change: register `V_mem` as `nn.Parameter` instead of buffer. K_mem stays a buffer. Add `model.hopfield.V_mem` to the Muon optimizer's 2D-param list (it's (M, d) and benefits from spectral norm).

## Memory-Movement Analysis
- V_mem becomes a parameter: at M=4096, d=384 → 1.5M extra params, +4 MB optimizer state, +4 MB gradient. Negligible.
- Forward unchanged. Backward: gradient through `attn @ V_mem` is just `attn.T @ grad_out` — one extra (M, B·T) × (B·T, d) matmul per step. ~M/T × per-attn cost, ~0.6 GFLOP — sub-1% of per-step compute.
- Total per-step compute delta: ~+1%. Total energy delta predicted: +1–2%.

## Setup
- Same as `hopfield_layer`.
- Configurations:
  - A: `V_mem` learnable, K_mem frozen (this experiment).
  - B (already done): both frozen — baseline = `hopfield_layer`.
  - C (for completeness): K_mem learnable, V_mem frozen. Tests whether the "memory" actually needs to be frozen.

## Procedure
1. `cp -r submissions/hopfield_layer submissions/hopfield_learn_V`
2. In `HopfieldLayer.__init__`, change:
```python
self.register_buffer("V_mem", torch.zeros(M, d))
```
to
```python
self.V_mem = nn.Parameter(torch.zeros(M, d))
```
3. Init V_mem via `_init_hopfield_memory` as before (it'll be overwritten by `.copy_`).
4. In the optimizer setup, the existing `block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2]` excludes V_mem because it's `model.hopfield.V_mem`. Add it explicitly:
```python
hopfield_2d = [model.hopfield.V_mem]
block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2] + hopfield_2d
```
5. Train. Submit.
6. Mirror for variant C (K_mem learnable, V_mem frozen).

## Success Criteria
- **Strong**: learn-V matches `modded_nanogpt` acc (≥0.737) at energy < 45 kJ → the gradient-free contribution is fully from frozen-key retrieval.
- **Pass**: learn-V improves on `hopfield_layer` by ≥0.005 acc at energy ≤ 42 kJ.
- **Diagnostic value (regardless)**: comparing A vs C tells us which side of the kernel (features vs readout) carries the gradient-free win.

## Failure Modes & Diagnostics
- V_mem explodes under Muon: V_mem is M × d which is much larger than block params; Muon's spectral norm step may apply differently. Diagnostic: log V_mem singular-value spectrum every 500 steps; if top singular value > 10× init, fall back to AdamW for V_mem only.
- Trained V_mem just memorizes a few targets and ignores the rest of K_mem: log row-norms of V_mem; if 90% of rows have ‖V_mem[i]‖ < 0.01·max, V_mem has collapsed — try L2-reg on V_mem.

## Estimated Cost
2 Modal runs (variant A, variant C) × 10 min ≈ $0.85.

## References
- `experiments/kernel_methods/result_11.md`
- Ramsauer 2020. Note: their original Hopfield-as-attention identity treats the keys/values *both* as either parameters or stored patterns — separating them was not their focus, so this ablation has not (to our knowledge) been published.
