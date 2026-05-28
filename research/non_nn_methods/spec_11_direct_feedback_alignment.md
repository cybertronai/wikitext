# Spec 11 — Direct Feedback Alignment (DFA) char-LM

## 1. Method & mechanism

Direct Feedback Alignment (Nøkland 2016) replaces the chain-rule backward pass with
a fixed random projection of the output-layer error directly to every hidden layer:

    Forward (standard):
        h_l = sigma(W_l h_{l-1} + b_l)

    Backward (DFA, NOT backprop):
        e = (output - target)              # error at output
        delta_l = (B_l e) * sigma'(h_l)     # B_l is a FIXED random matrix per layer
        dW_l = delta_l h_{l-1}^T            # local outer-product update
        dB_l: never updated.

Key property: each layer's weight update needs only its own input + a top-down random
projection of the output error. **The full activation stack is never needed for the
backward pass.** This is precisely the memory-movement story driving the project
framing (see `project_wikitext_task.md`: "backprop is inefficient in commute-to-compute
because it requires fetching all activations for each gradient add").

For char-LM: a stack of L feedforward layers (or a small transformer) trained by
DFA on the 256-way softmax cross-entropy.

## 2. Why not a neural network / not backprop

The architecture *is* a neural network — fully connected layers or small transformer
blocks. The training is *not* standard backprop:

- No chain rule. The "gradient" at layer l is the output error pushed through a
  fixed random projection B_l.
- The activation stack does not need to be held in memory for the backward pass.
- Provably distinct gradient direction (Nøkland 2016 Lemma 1: the DFA pseudo-gradient
  becomes aligned with the true backprop gradient as training progresses — "feedback
  alignment").
- **Borderline backprop**: the user's filter explicitly accepts "target propagation,
  equilibrium propagation, direct feedback alignment, ..., FF, Hebbian/STDP" with
  the relation to backprop spelled out. DFA is the canonical near-backprop
  alternative that retains layer-stacked function-class expressivity.

## 3. Universal approximation status

The model class (MLP / transformer) is UAT by Cybenko 1989 / Hornik 1991. DFA
training is empirically known to reach SGD-comparable accuracy on MLP-friendly
tasks (Nøkland 2016: MNIST + CIFAR within ~1-3% of SGD; Refinetti et al. 2021:
DFA matches backprop for small two-layer nets, gap grows with depth).

For transformer training, DFA has been shown to work but with a measurable gap
to backprop on language-modeling benchmarks (Launay et al. 2020).

## 4. Discrete categorical fit

Standard 256-way softmax output head. Soft scores, no stochasticity risk.

## 5. Autoregressive applicability

Yes — DFA has been published on transformers + LM benchmarks (Launay, Poli,
Krzakala 2020 "Direct Feedback Alignment Scales to Modern Deep Learning Tasks
and Architectures"). Their transformer-LM on WikiText reached within ~5% of
backprop perplexity. **Not yet shown on the modded_nanogpt scale / 300s budget.**

## 6. Roofline analysis

DFA's central memory-movement claim: the *backward pass* arithmetic intensity is
strictly higher than backprop's because the activation stack is not read back.

Per training step, for L layers of width d, batch B, sequence T:
- **Forward**: same as backprop. B*T*L*d^2 FLOPs, ~B*T*L*d bytes for activations.
- **Backward (DFA)**: per layer, one outer product dW_l = delta_l h_{l-1}^T. The
  delta_l is computed from one matmul of error with B_l (random feedback): B*T*d*V
  FLOPs (V=256), and the outer product is B*T*d^2 FLOPs. **No need to read the
  upstream-layer gradient.**
- **Backward (backprop, for contrast)**: per layer, two matmuls (input grad +
  weight grad) plus a read of the activation stack from HBM.

Quantitative on modded_nanogpt's architecture (L=6, d=384, B=32, T=1024):
- DFA backward per step: 6 * (32 * 1024 * 384 * 256) ~ 6 * 3.2e9 = 1.9e10 FLOPs.
- Backprop backward per step: 6 * (2 * 32 * 1024 * 384^2) ~ 6 * 9.7e9 = 5.8e10 FLOPs.
- DFA reads ~half the activation bytes that backprop does.

**Arithmetic intensity of DFA backward: ~250 FLOPs/byte** (the per-layer outer
product is dense matmul of d x d, B*T-major); **compute-bound.**

Forward + DFA: net ~30-50% FLOP reduction vs backprop, and a strict reduction in
HBM traffic. The energy story is concrete: at A100's energy-per-FLOP ratio,
**a ~30% reduction in training FLOPs translates to a ~30% reduction in joules
all else equal.** Direct path to beating 51.7 kJ.

## 7. Top references

1. Nøkland 2016, "Direct Feedback Alignment Provides Learning in Deep Neural
   Networks", NeurIPS. <https://arxiv.org/abs/1609.01596>
   *DFA original.*
2. Launay, Poli, Krzakala, Boniface 2020, "Direct Feedback Alignment Scales to
   Modern Deep Learning Tasks and Architectures", NeurIPS.
   <https://arxiv.org/abs/2006.12878>
   *DFA on transformers + LM at WikiText scale.*
3. Refinetti, d'Ascoli, Ohana, Goldt 2021, "Align, then memorise: the dynamics
   of learning with feedback alignment", ICML. <https://arxiv.org/abs/2011.12428>
   *Theoretical analysis; quantifies the depth-dependent gap.*
4. Frenkel, Lefebvre, Bol 2021, "Learning without feedback: Fixed random
   learning signals allow for feedforward training of deep neural networks", Frontiers.
   <https://www.frontiersin.org/articles/10.3389/fnins.2021.629892>
   *DRTP — DFA generalized to per-step random targets; more memory-efficient.*
5. Han, Bohte, Roerdink 2024 review, "Local Learning Rules for Spike-Based and
   Other Biologically Plausible Networks". *Survey including DFA placement.*

## 8. Limitations / failure modes

- **Depth-dependent gap to backprop.** Refinetti 2021 shows the DFA-vs-backprop
  gap grows with depth; for L=6 modded_nanogpt this should be manageable but for
  deeper nets the gap can be 10-20%.
- **Transformer attention layers and layer-norm are not trivially DFA-compatible.**
  Launay 2020 reports that attention requires careful per-component DFA recipes;
  some skip-connections need to be treated specially.
- **Random feedback matrices** consume parameters: an L * d * V projection per
  layer. For L=6, d=384, V=256: 6 * 384 * 256 * 4 bytes = 2.4 MB — negligible.
- **No published WikiText-103 char-level (byte) DFA result** at 300 s budget. The
  Launay 2020 result is word-level WikiText with 1+ day budgets.
- **Optimization stability.** DFA needs lower learning rates than backprop;
  inappropriate LR can cause training divergence.

## 9. Experiment spec

**Setup.**
- Architecture: same as `submissions/modded_nanogpt` (6-layer 384-dim transformer)
  but with DFA replacing the backward pass.
- Random feedback matrices B_l: sampled once at init from N(0, 1/sqrt(d)),
  frozen.
- Layer norm: keep, but only apply DFA to the linear/attention parameters
  (LN gain/bias updated by closed-form running statistics — non-gradient).
- Optimizer: SGD with momentum 0.9, lr=3e-3 (lower than backprop's typical lr,
  per Launay 2020 recipe).
- Steps target: same wall-clock budget as modded_nanogpt's 2150 steps.

**Implementation.**
- Implement DFA backward as a custom `torch.autograd.Function` that overrides
  the backward to do the fixed-projection rule instead of chain rule.
- Use the same data pipeline / batch shape / bf16 mixed precision as the
  modded_nanogpt submission so this is an apples-to-apples comparison.

**CharModel translation.** Identical to modded_nanogpt (forward pass + softmax).

**Energy budget.** Per the ~30% FLOPs reduction estimate, **~35 kJ** target
(vs 51.7 kJ baseline). This is the most concrete "beat baseline" candidate in
the portfolio.

**Char-acc ceiling estimate.** 0.65–0.73 — Launay 2020's WikiText (word-level)
result was ~5% below backprop. If the gap is similar at byte-level, this clears
the 0.70 floor comfortably.

## 10. Verdict — **Tier A — most direct beat-baseline candidate**

DFA is the most credible "competing with modded_nanogpt on its own ground" entry
in this portfolio: same architecture, ~30% FLOPs reduction, published precedent
on LM tasks at within-5% of backprop accuracy. The novelty here is the
**byte-level WikiText 300 s** demonstration — not yet in the literature.

Run after CTW (spec_07, cheap fast-failure) and RFF (spec_02, cheap kernel
baseline).
