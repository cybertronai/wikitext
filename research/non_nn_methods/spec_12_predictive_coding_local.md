# Spec 12 — Predictive Coding Network with local Hebbian updates

## 1. Method & mechanism

A Predictive Coding Network (PCN; Rao & Ballard 1999; Whittington & Bogacz 2017) is a
hierarchical generative model whose layers each predict the activity of the layer
below. The dynamics minimize a sum of *layer-local* prediction errors:

    F = sum_l ||x_l - g(W_l x_{l+1})||^2 / (2 sigma_l^2)

At inference (and during training), layer activities x_l relax to a fixed point that
minimizes F given clamped input. **Weight updates** are then *local Hebbian*:

    Delta W_l = - eta * (x_l - g(W_l x_{l+1})) * x_{l+1}^T * g'(W_l x_{l+1})

This update uses only quantities accessible at layer l + its neighboring layer's
activity at the relaxed fixed point — no chain rule through the rest of the network.

For char-LM: condition the bottom layer x_0 on the K-byte context window, clamp the
top layer to a "predict next byte" representation, and read out the predicted next
byte from a learned projection of x_1 (the layer above input).

## 2. Why not a neural network / not backprop

The architecture is a neural network (stack of dense layers), but training updates
each weight using **only the activities of the two layers it connects**. No backward
pass through the entire stack is performed. Whittington & Bogacz 2017 prove
predictive coding *approximates* backprop in a specific limit but the actual computed
gradient is local Hebbian.

**Borderline backprop, like DFA (spec_11).** The user's filter explicitly accepts
predictive coding networks.

## 3. Universal approximation status

The architecture is a multi-layer feedforward / hierarchical generative model and
inherits UAT from MLP UAT (Cybenko 1989). Training convergence is local-minimum
only; same caveats as SGD on MLPs.

## 4. Discrete categorical fit

Top layer is a 256-d categorical readout — soft, no stochasticity risk. The relaxation
dynamics give a posterior over the next-byte representation, which projects onto a
softmax over 256 classes.

## 5. Autoregressive applicability

PCN has been demonstrated for sequence modeling (Salvatori et al. 2022, "Reverse
Differentiation via Predictive Coding") on small toy NLP tasks. **No published
byte-level WikiText result.** AR adaptation: feed K-byte context window as bottom-
layer clamp; relax; read top-layer next-byte prediction.

## 6. Roofline analysis

Per training step, the iterative relaxation is the dominant cost. With T_relax = 10
relaxation iterations per training sample:

- Per iter: 2L matmuls (forward predict + backward error pass) = 2 * L * B * T * d^2
  FLOPs.
- Per training step: T_relax * 2L * B * T * d^2 FLOPs.

For L=6, d=256, B=32, T=64, T_relax=10:
- Per step: 10 * 12 * 32 * 64 * 256^2 = 1.6e10 FLOPs.

This is ~3x more FLOPs per step than backprop on the same architecture (backprop
is forward + 1 backward). However:
- Per-iter matmul is dense, Tensor-Core-friendly.
- Activation memory: only need current relaxed state, *not* a backward-pass
  stash — memory is O(L*d) instead of O(L*d*T).
- HBM traffic per step: ~similar to backprop forward (we re-read the same data per
  relax iter); arithmetic intensity ~300 FLOPs/byte. **Compute-bound.**

Net energy estimate: 3x FLOPs at same intensity → 3x energy → **~150 kJ**. Worse
than backprop on energy. **The interest is not energy efficiency; it is the
"can a local-rule trained NN clear 0.70 on byte LM?" capability claim.**

## 7. Top references

1. Rao, Ballard 1999, "Predictive coding in the visual cortex: a functional
   interpretation of some extra-classical receptive-field effects", Nat. Neurosci.
   *Original PCN.*
2. Whittington, Bogacz 2017, "An Approximation of the Error Backpropagation Algorithm
   in a Predictive Coding Network with Local Hebbian Synaptic Plasticity", Neural Comput.
   <https://pubmed.ncbi.nlm.nih.gov/28333583/>
   *Proves PCN approximates backprop in the small-error limit.*
3. Millidge, Tschantz, Buckley 2020, "Predictive Coding Approximates Backprop along
   Arbitrary Computation Graphs", arxiv:2006.04182.
   *Generalization of W&B 2017 to arbitrary architectures.*
4. Salvatori, Song, Hong, Sha, Frieder, Xu, Bogacz, Tang, Lukasiewicz 2022,
   "Learning on Arbitrary Graph Topologies via Predictive Coding".
   <https://arxiv.org/abs/2201.13180>
   *Shows PCN training works on transformer-like graphs.*
5. Salvatori, Pinchetti, Millidge, Song, Bogacz, Lukasiewicz 2023,
   "Reverse Differentiation via Predictive Coding", AAAI.
   <https://ojs.aaai.org/index.php/AAAI/article/view/26136>
   *Most recent PCN training results; small toy NLP among the tasks.*

## 8. Limitations / failure modes

- **Relaxation cost.** T_relax iterations multiply the per-step cost. Reducing
  T_relax to 1-2 (a near-instant forward) loses much of the PCN advantage but
  is what's needed at the 300 s budget.
- **No published WikiText-103 byte PCN result.** Expected char-acc 0.40–0.55 based
  on small-NLP precedents.
- **Energy worse than backprop** — runs counter to the project's energy-frontier
  framing. The justification is mechanistic novelty.
- **Stability** of the relaxation dynamics needs careful tuning of layer-wise
  precisions sigma_l.

## 9. Experiment spec

**Setup.**
- 6-layer fully-connected predictive coding stack, d=256 hidden, K=16 byte context.
- T_relax = 4 inference iterations (reduced from typical 10 for budget).
- Hebbian update lr = 1e-3 with linear warmup.
- Cross-entropy at the top layer projected to 256-byte softmax.
- bf16 mixed precision; relaxation in fp32.

**Implementation.**
- Reference: `nat-rg/predictive-coding-tutorials` for the layer-local update code.
- Replace the standard backward pass with iterative relaxation + Hebbian update.

**CharModel translation.**
- `predict()`: clamp K-byte context to bottom layer, relax for 4 iters, project top
  layer to 256-byte softmax. ~10 ms / char.
- `observe(c)`: update buffer; per-eval-char Hebbian updates allowed (CharModel
  contract allows learning during eval).

**Energy budget.** ~100–200 kJ training. **Will not be energy-competitive.**

**Char-acc ceiling estimate.** 0.40–0.55. Likely DQ on 0.70.

## 10. Verdict — **Tier B**

Mechanism novelty: first PCN result on byte-level WikiText. Energy story is poor.
Includes for the breadth of "near-backprop alternatives" (DFA, PCN, target prop,
equilibrium prop). Run only as a calibrated comparison data point against DFA
(spec_11). The PCN-vs-DFA contrast is interesting because they make opposite
choices: PCN keeps a local objective but pays an iterated-relaxation cost; DFA
keeps a single forward+backward pass but with a non-chain-rule backward.
