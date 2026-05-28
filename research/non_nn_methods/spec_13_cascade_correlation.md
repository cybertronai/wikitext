# Spec 13 — Cascade-correlation constructive LM

## 1. Method & mechanism

Cascade-Correlation (Fahlman & Lebiere 1990) is a constructive learning algorithm:
start with a minimal network (input + output, no hidden units); fit the output
weights by closed-form linear regression on the residuals; then *add* one hidden
unit at a time, training each new unit to maximize the *correlation* between its
output and the current residual error. New units feed into all later units
(cascade structure) — old weights are *frozen* after the unit is added.

Algorithm per added unit:
1. Sample N candidate units with random init.
2. For each, gradient-ascend on covariance(unit_output, residual_error) over
   training samples — *only the new unit's weights move*, not any other layer.
3. Add the winner to the network; refit output weights by linear regression.

For char-LM: input = K-byte context window; output = 256-d softmax; hidden units
added one at a time to fit residuals on the cross-entropy loss.

## 2. Why not a neural network / not backprop

The network is an MLP architecture, but training is **strictly layer-by-layer
greedy** with closed-form output refits. The per-unit gradient ascent is a *local*
single-layer optimization, not backpropagated through a stack — by construction
there is no backward pass through the prior layers (they're frozen).

This is at the boundary of the "no backprop" filter. The per-candidate unit training
uses gradient ascent on a 1-layer covariance criterion — gradient-based but not
chain-rule across the stack. Spelling out for the user's filter: no end-to-end
backprop; each unit's training is a 1-layer problem.

## 3. Universal approximation status

The constructive class is UAT in the limit of infinite added units (Fahlman & Lebiere
1990): given enough hidden units, a cascade-correlation net can approximate any
continuous function on a compact domain. Convergence rate is greedy and may be slow.

## 4. Discrete categorical fit

Standard softmax output head — refit by closed-form linear regression on residuals
after each unit add. 256-d categorical, soft scores. Stochasticity-safe.

## 5. Autoregressive applicability

Cascade-correlation has been used for sequence prediction in *Recurrent Cascade-
Correlation* (Fahlman 1991), with known limitations (Giles et al. 1995) showing it
cannot represent certain finite-state automata. For char-LM at byte level, using
the *feedforward* cascade with a sliding-window input is the natural setup.

**No published byte-LM result for cascade-correlation.** Capability demo.

## 6. Roofline analysis

Adding N=20 units in 300 s budget:
- Per unit: covariance maximization on (B*T)-sample set with d-dim input ~= 1-layer
  MLP training, 100 steps of SGD, batch B*T=32K samples, d=256 -> 100 * 32K * 256 =
  8e8 FLOPs.
- Output refit: B*T x H matrix solve, H grows from 0 to 20 — total cost across 20
  units sum_{H=0..20} (B*T)*H^2 ~ 1e10 FLOPs.
- Total: 20 * 8e8 + 1e10 = 2.6e10 FLOPs — fast.

Inference: forward pass through K-byte input + 20 hidden units = 256*K + 20 * 256
ops per char = 5K + 5K = 10K ops per char ≪ 60K * 1e4 = 6e8 FLOPs eval = trivial.

**Compute-bound, but utilization will be poor** because each unit's per-iter matmul
is tiny (d=256). Expected ~5% A100 peak.

Memory: a few MB of weights. Far from saturating HBM.

## 7. Top references

1. Fahlman, Lebiere 1990, "The Cascade-Correlation Learning Architecture", NeurIPS.
   <https://papers.nips.cc/paper/207-the-cascade-correlation-learning-architecture>
   *Cascade-correlation original.*
2. Fahlman 1991, "The Recurrent Cascade-Correlation Architecture", NeurIPS.
   *Recurrent variant.*
3. Giles, Chen, Sun, Chen, Lee, Goudreau 1995, "Constructive learning of recurrent
   neural networks: limitations of recurrent cascade correlation and a simple
   solution", IEEE Trans. NN. <https://ieeexplore.ieee.org/document/392247/>
   *Documented limitations of RCC.*
4. Lahnajärvi, Lehtokangas, Saarinen 2002, "Evaluation of constructive neural
   networks with cascaded architectures", Neurocomputing.
   *Modern comparison.*

## 8. Limitations / failure modes

- **20-unit cap is far below typical CC ablation sizes** (100-1000). Even with
  closed-form refits, the function class at 20 hidden units is shallow / narrow.
  Expected char-acc 0.35–0.55.
- **No published text-LM result.**
- **Per-unit gradient ascent is small-batch SGD** — modest GPU utilization.
- **Constructive methods are notoriously slow** to reach competitive performance;
  the modern reading is that they are out-classed by depth-fixed gradient methods.
- The **20 candidate units per add** introduces a stochastic search element similar
  to ES but at a much smaller scale.

## 9. Experiment spec

**Setup.**
- Input: K=16 byte context one-hot, flattened to 4096-dim.
- Output: 256-byte softmax.
- Add up to 20 cascaded units, each trained for 50 SGD steps of covariance
  maximization with batch 1024.
- Pool of 8 random candidates per add, pick the best by training-set covariance.
- Output refits by closed-form ridge after each unit.

**Implementation.**
- Custom training loop in PyTorch — no off-the-shelf CC library is GPU-mature.
- Frozen-weight checkpoint per added unit; output weights re-solved each cycle.

**CharModel translation.**
- `predict()`: forward pass through input + all added units → softmax. ~10K ops / char.
- `observe(c)`: append to buffer; no learning during eval (closed-form after training).

**Energy budget.** 60–120 s training, 5–15 kJ.

**Char-acc ceiling estimate.** 0.35–0.55. Likely DQ on 0.70.

## 10. Verdict — **Tier C**

Historical lineage and breadth coverage only. Cascade-correlation is the canonical
constructive / non-fixed-depth method but is widely understood as out-classed.
Run only if a "constructive vs. fixed-depth" comparison data point is wanted; do
not expect a leaderboard entry.
