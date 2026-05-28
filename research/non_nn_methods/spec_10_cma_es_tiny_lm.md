# Spec 10 — Covariance Matrix Adaptation Evolution Strategy (CMA-ES) on a tiny LM

## 1. Method & mechanism

CMA-ES (Hansen & Ostermeier 2001) is a second-order black-box optimization method that
maintains a multivariate Gaussian search distribution N(mean, sigma^2 * C) over
parameter space and adapts both the mean and the full covariance C from the fitness
ranking of sampled offspring. Update rule: weighted recombination of best half of
population for mean; rank-mu + rank-one updates for C; cumulative step-size adaptation
for sigma.

For char-LM: parameterize a *tiny* LM (e.g., a 2-layer 32-dim transformer or a 1-layer
gated linear unit, ~10K parameters) and let CMA-ES search over its parameter vector.
Per iteration: sample population of N=10 + 3 ln(d) offspring, evaluate fitness =
negative log-likelihood on a fixed validation subset, rank, update.

The Sep-CMA-ES variant restricts C to diagonal — feasible at 10K-50K params.
Full-covariance CMA-ES is infeasible above ~1000 params (matrix decomposition cost
O(d^3)).

## 2. Why not a neural network / not backprop

CMA-ES is the strongest classical gradient-free optimizer. It does NOT use
backpropagation — only forward passes of the model on the fitness data. The
underlying model (a tiny transformer) is a neural network architecture but trained
without backprop.

**This is a borderline case.** The architecture is a transformer; only the
training algorithm is non-backprop. Under the user's filter, "borderline ok if the
user can clearly identify 'this is not standard backprop'" — CMA-ES qualifies and
this spec is included for the diversity of the optimizer family. Note: the
existing `gradfree-survey/es-tiny-transformer` is the simpler **OpenAI-ES** (rank
weighting + diagonal Gaussian) and DQ'd at val=0.19. CMA-ES is a strict
**second-order** improvement — the question is whether the covariance adaptation
overcomes ES's sample-inefficiency.

## 3. Universal approximation status

The optimizer is universal in the sense that CMA-ES converges to a local optimum
of any continuous fitness with positive probability (Auger & Hansen 2011 theoretical
analysis). The *model class* (a tiny transformer) is a universal approximator with
sufficient width / depth. The composite system thus has UAT in principle — but the
practical question is sample efficiency.

## 4. Discrete categorical fit

The model is a transformer with a 256-way softmax output head. Fitness is per-batch
mean cross-entropy. Soft outputs; no stochasticity-filter risk.

## 5. Autoregressive applicability

Standard transformer LM — handles AR by construction. The CharModel translation is
identical to modded_nanogpt's: forward pass, softmax output.

CMA-ES has not been published as a training method for autoregressive byte-LM. The
closest is the OpenAI-ES + tiny transformer in the gradfree-survey (DQ at 0.19).
**Novel application** of CMA-ES specifically.

## 6. Roofline analysis

Per CMA-ES iteration with population P, model with d parameters, fitness over B
batches of T tokens:
- Forward passes: P * B * T model FLOPs.
- Rank + covariance update: O(d^2) for Sep-CMA-ES (diagonal); O(d^3) for full CMA-ES.

For d=10K, P=20, B=32, T=512:
- Fitness forwards: 20 * 32 * 512 * 10K ops/token * 2 (model FLOPs per param) =
  6.5e9 ops per iter * tiny-model-per-token-cost.
- Tiny-model per-token: ~2 * d ~ 20K FLOPs. Total per iter: 6.7e9 + 1e8 = ~7e9 FLOPs.
- 300 s budget → ~3e11 forward FLOPs available → ~40 CMA-ES iterations on this tiny
  model.

This is *insufficient* for CMA-ES to converge — the published OpenAI-ES baseline did
~600 iters and still DQ'd. CMA-ES is somewhat faster per iter (covariance adapts to
the landscape) but the sample budget is still tight.

**Compute-bound** (the model evaluations are dense matmul). But the model is *too
small* to saturate Tensor Cores — each per-token matmul is too small (32-dim hidden).
Expected utilization 5-10% of peak.

## 7. Top references

1. Hansen, Ostermeier 2001, "Completely Derandomized Self-Adaptation in Evolution
   Strategies", Evol. Comput. <https://www.lri.fr/~hansen/cmaartic.pdf>
   *CMA-ES original.*
2. Hansen 2016, "The CMA Evolution Strategy: A Tutorial". <https://arxiv.org/abs/1604.00772>
   *Modern tutorial; includes Sep-CMA-ES variant.*
3. Salimans, Ho, Chen, Sidor, Sutskever 2017, "Evolution Strategies as a Scalable
   Alternative to Reinforcement Learning". <https://arxiv.org/abs/1703.03864>
   *OpenAI-ES on neural nets.*
4. Ros, Hansen 2008, "A Simple Modification in CMA-ES Achieving Linear Time and
   Space Complexity", PPSN. *Sep-CMA-ES.*
5. Auger, Hansen 2011, "Theory of Evolution Strategies: A New Perspective".
   *Convergence theory.*

## 8. Limitations / failure modes

- **Sample inefficiency.** ~40 iterations is far below CMA-ES's published
  convergence regime on similar-scale problems.
- **Covariance maintenance cost.** Sep-CMA-ES is O(d) per iter; full CMA-ES is
  O(d^3) — infeasible above d=1000.
- **Tiny model capacity ceiling.** 10K params is roughly bigram + position-aware
  trigram capacity. Char-acc ceiling estimate 0.30–0.50 — much like the existing
  ES submission.
- **Existing OpenAI-ES result strongly predicts failure.** CMA-ES improves
  optimization efficiency by ~10x vs OpenAI-ES on well-studied benchmarks but the
  problem here is fundamentally model-capacity-limited, not optimization-limited.
- **The `gradfree-survey/REPORT.md` conclusion specifically calls ES "DEAD-END":**
  "1/√D variance scaling argument predicted improvement from smaller D; the data
  says otherwise."

## 9. Experiment spec

**Setup.**
- Architecture: 2-layer 32-dim transformer with 4 heads, ctx=64, ~12K params total.
- Optimizer: Sep-CMA-ES (`cma` Python package, `CMAEvolutionStrategy` with
  `'CMA_diagonal': True`).
- Population: P = 4 + floor(3 * ln(12000)) = 32.
- Fitness: NLL on 32-sequence batch of 64 tokens per fitness call.
- ~30-50 generations in 270 s budget (allowing 30 s for setup/eval).

**CharModel translation.** Standard transformer; forward pass on byte token
sequence; softmax output. Identical wrapper to modded_nanogpt.

**Energy budget.** ~5-10 kJ training (model is tiny, GPU underutilized — significant
idle subtraction).

**Char-acc ceiling estimate.** 0.30–0.45 — similar to OpenAI-ES baseline.

## 10. Verdict — **Tier C**

The prior ES experiment failed (0.19 val); CMA-ES is a strict optimizer improvement
but the failure mode in `gradfree-survey/REPORT.md` was "capacity ceiling, not
optimization-budget bottleneck." Same failure expected. Run only as a comparison
data point if a definitive ES-family conclusion is wanted (e.g., to formally close
"no ES variant clears 0.70 in 300 s on this benchmark"). **Not recommended for
energy-frontier exploration.**
