# Spec 08 — Dynamic Markov Compression (DMC) LM

## 1. Method & mechanism

Dynamic Markov Compression (Cormack & Horspool 1987) is a bit-level adaptive
compressor. The model is a finite-state machine where each state has two transitions
(to a "0 successor" and a "1 successor") with counts (n0, n1). The predicted next-bit
distribution is (n0 + 1) / (n0 + n1 + 2). After each observation, the counts of the
active state are incremented. The FSM is *grown* by a state-splitting rule: when both
the transition count and the incoming count cross thresholds, the state is cloned and
the splitting transition is rerouted, allowing the FSM to learn richer context.

For byte-level char-LM: bit-by-bit DMC, then take the product of 8 bit-predictions per
output byte to form a 256-byte distribution.

## 2. Why not a neural network / not backprop

A growable Markov FSM with closed-form Laplace-rule updates. No layers, no gradients,
no SGD. The "training" is pure counting + a state-cloning rule.

## 3. Universal approximation status

**Empirical.** DMC is in the same family as CTW and PPM — finite-context-mixture
adaptive predictors. It does not come with the clean Bayesian-optimality guarantee
of CTW, but it has comparable empirical compression ratio (Cormack & Horspool 1987;
Bell, Cleary, Witten 1990 book). The state-splitting heuristic does not have a
known UAT-style result; the model class is "tree-source FSMs reachable by the
splitting rule from the initial FSM."

## 4. Discrete categorical fit

Same as CTW (spec_07): bit-conditionals multiplied to give a 256-byte distribution.
Soft output, no stochasticity-filter risk.

## 5. Autoregressive applicability

Native. DMC is literally an online autoregressive byte/bit predictor.

## 6. Roofline analysis

Same profile as CTW: bandwidth-bound pointer-chasing. The FSM grows during training,
so allocation overhead can be significant. ~10-30 MB FSM at convergence on text. Per
bit: 1 transition + count update + possible split test = ~20 ops + 1 cache line
streamed.

Total compute for N=5e8 bytes: ~1e11 ops. HBM: ~5e10 bytes streamed.
Arithmetic intensity: ~2 ops/byte. **Deeply bandwidth-bound, GPU-unfriendly.**

CPU implementation is the natural substrate. On A100 the GPU sits idle most of the
time waiting for memory.

## 7. Top references

1. Cormack, Horspool 1987, "Data Compression Using Dynamic Markov Modelling", Comp. J.
   <https://webhome.cs.uvic.ca/~nigelh/Publications/DMC.pdf>
   *DMC original.*
2. Bell, Cleary, Witten 1990, "Text Compression", Prentice Hall.
   *Comparative analysis of DMC vs PPM vs LZ.*
3. Bunton 1996, "A Characterization of the Dynamic Markov Compression FSM Family",
   UW Tech Report. <https://dada.cs.washington.edu/research/tr/1994/11/UW-CSE-94-11-03.pdf>
   *Theoretical characterization of DMC's model class.*

## 8. Limitations / failure modes

- **Strictly weaker than CTW in expected redundancy** under the universal-source
  framework (no closed-form bound).
- **No GPU-native implementation.** Bandwidth-bound throughout.
- **Memory growth** during training is unbounded without a node-cap; rare-context
  pruning is non-canonical and impacts the model's monotone-convergence story.
- **No published modern-hardware DMC result on WikiText-103.**
- **Predicted to underperform CTW** at equal compute budget. Run only as a "different
  family of adaptive Markov FSM" comparison data point.

## 9. Experiment spec

**Setup.**
- Bit-level DMC with state-cloning thresholds (MIN_CNT1=2, MIN_CNT2=2, default
  Cormack-Horspool values).
- Max FSM size: 5e7 states (~600 MB at 12 bytes/state). Prune by LRU if exceeded.
- CPU implementation via Cython or pre-compiled C extension; A100 used only for the
  NVML energy measurement.

**CharModel translation.** Identical to CTW (spec_07).

**Energy budget.** Training: 60–120 s on the train set (CPU-bound on a single thread).
A100 idle while CPU works: ~50 W * 90 s = 4.5 kJ baseline-subtracted = 0 (the idle
subtraction zeroes it out unless we set the meter aggressively). Net energy: dominated
by the GPU package idle, sub-3 kJ.

**Char-acc ceiling estimate.** ~0.60–0.70, lower than CTW. Likely DQ on 0.70.

## 10. Verdict — **Tier C**

Distinct family from CTW but strictly weaker on expected redundancy. Use only as a
comparison point if CTW lands. Otherwise skip.
