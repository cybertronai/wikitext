# 07 · Forward-Forward Sequence Model with Associative-Recall Goodness

## Mechanism

Hinton's Forward-Forward 2022 (arXiv 2212.13345) trains layers locally
using positive and negative examples and a per-layer "goodness" score. Each
layer independently learns to push `goodness(positive) > θ > goodness(negative)`.

The standard FF construction for classification labels: positive examples
embed the correct class label in the input; negative examples embed a wrong
label. The layer's goodness (typically `||h||²`) discriminates between them.

**This direction reframes the FF positive/negative pair as an
associative-recall problem.** For char-LM:

- **Positive example:** `(context window of length T, next byte = byte_y)`
  concatenated as input — i.e. an example where the byte_y is the *correct*
  continuation of context.
- **Negative example:** `(context window, next byte = byte_y')` where
  byte_y' is sampled from the *unconditional* unigram (or a stronger
  bigram) baseline. Negative examples are continuations that don't match
  context.
- **Goodness function:** `g_ℓ(h_ℓ) = h_ℓ^T M_ℓ h_ℓ` where M_ℓ is a small
  associative-memory matrix stored *Hebbian-style* at this layer from
  positive examples only. High g_ℓ means h_ℓ matches well-stored positive
  patterns.

Training:

1. **Layer initialization:** M_ℓ = 0.
2. **Per training batch:**
   - Compute h_ℓ = σ(W_ℓ h_{ℓ-1}) for positive and negative examples
     separately.
   - Hebbian write to M_ℓ:  `M_ℓ += (1/B) Σ_+ h_+ h_+^T - (1/B) Σ_- h_- h_-^T`.
   - W_ℓ is updated via the FF local rule on the per-layer logistic loss
     `log σ(g_ℓ(h_+) - g_ℓ(h_-) - θ)` — but with the goodness defined as
     above, **the gradient of g_ℓ wrt W_ℓ is itself an outer product**.
     So the W update is also Hebbian:
       ∂g/∂h = (M_ℓ + M_ℓ^T) h
       W_ℓ_new = W_ℓ + η · (rate) · h_{ℓ-1} · ((M_ℓ + M_ℓ^T) h_ℓ)^T
   This *is* the FF update rule, written as an outer product.

**No global backward pass.** Each layer's W_ℓ is updated by a single outer
product per batch using only its own activations and its own M_ℓ. M_ℓ
itself is updated by pure Hebbian.

Prediction: at inference, for each candidate next byte b ∈ {0..255}, form
`x = [context, b]`, propagate through all layers, sum the goodness across
layers, normalize via softmax to get a 256-d distribution.

This is the **256-way generative discriminator** form of FF — standard FF
practice for classification.

## Seed papers

- Hinton, *The Forward-Forward Algorithm: Some Preliminary Investigations*,
  arXiv 2212.13345 (2022). Establishes goodness-based local training.
- Ororbia, Mali, *The Predictive Forward-Forward Algorithm*, arXiv
  2301.01452 (2023). FF + predictive coding for sequence prediction.
- Zhao et al., *Mono-Forward: Backpropagation-Free Algorithm for Efficient
  Neural Network Training Harnessing Local Errors*, arXiv 2501.09238 (2025).
  Most recent improvement on the FF paradigm.
- DeeperForward (ICLR 2025) referenced from FF literature — multi-layer
  scaling of FF.
- See also `finding_gradfree_family_verdicts.md`: prior FF-causal-LM
  attempt plateaued at 0.279.

## Why it could work here (modulo the prior FF-LM failure)

- The previously failed FF-LM attempt (per memory) used Hinton's *original*
  per-byte sliding goodness, not the associative-recall variant proposed
  here. The associative-recall framing is mechanistically distinct.
- The 256-way inference loop is expensive (256× per-byte forward) but
  trivially parallelizable: batch all 256 candidates and one forward pass
  emits all goodnesses.
- M_ℓ stays small (d × d per layer), so HBM traffic is bounded.
- The W updates are *literally* one outer product per batch per layer.
  Tensor Cores will saturate.

## Threshold of plausibility

The prior FF-LM attempt's 0.279 ceiling is a strong negative prior. The
research framing memory marks "FF as a full LM" as DEAD. This experiment
is included not because it's likely to clear 0.70 but because the
associative-recall goodness *is* a different mechanism that hasn't been
tested at char-LM.

Realistically: 0.30–0.40. **Most likely a confirming-the-ceiling result.**
The value is the post-mortem ablation: did the associative-recall
goodness improve over the original FF formulation, even if neither cleared
the gate?

If by chance it clears 0.50 it would be the first FF-paradigm result at
char-LM, which is publishable on its own.

## Failure modes

- **Reasserts the FF-LM 0.279 ceiling.** Most likely.
- **256-way inference is too slow.** At 60 K val bytes × 256 candidates ×
  4-layer forward = 60 M layer-forwards. With per-forward cost ~50 K flops
  → 3 × 10^12 flops at inference, ~10 s. Tight but feasible.
- **Stochasticity filter:** the 256-way goodness softmax is a soft
  distribution by construction, so the filter passes.
- **Negative-example sampling matters.** Uniform negatives are weakest;
  unigram-frequency negatives are stronger; bigram-frequency negatives
  are strongest. Sweep these.
- **Symmetric M_ℓ vs asymmetric.** Hinton's FF assumes scalar goodness
  that doesn't distinguish forward/backward pattern. Allow M_ℓ ≠ M_ℓ^T
  for richer representations.

What would falsify it: 4-layer FF-seq with associative-recall goodness,
optimal negative sampling, val acc ≤ 0.30 → confirms the FF-LM ceiling is
not specific to Hinton's goodness, it's intrinsic to the paradigm. Mark
permanently dead.

## Smallest first experiment

`ff_seq_assocrec_v1`:

1. **Frozen byte embedding:** 256 → 128-d Gaussian random, fixed.
2. **Window feature:** last 64 bytes flattened, projected by frozen RFF
   to d_h = 256.
3. **Number of FF layers:** 1, 2, 4 (sweep).
4. **Per layer ℓ:**
   - W_ℓ: (d_h, d_h), initialized random Gaussian.
   - M_ℓ: (d_h, d_h), zeros.
5. **Training pass over 100 K positive examples + 100 K negatives:**
   - For each (context, byte_y) positive: h = σ(W · [feature; embed(byte_y)]);
     M_ℓ += (1/N) h h^T.
   - For each (context, byte_y') negative (sampled from unigram): same,
     but subtract: M_ℓ -= (1/N) h h^T.
   - Periodically (every 5K examples), refresh W via the FF outer-product
     update.
6. **Inference (`predict`):**
   - For current window, compute feature φ.
   - For each candidate byte b: form x_b = [φ; embed(b)], compute
     h_ℓ = σ(W_ℓ h_{ℓ-1}) for all layers, sum goodness over layers.
   - Softmax over 256 candidates → distribution.
7. **No streaming state needed.**

Sweep: layer count, negative sampling distribution (uniform / unigram /
bigram), η (W update rate), θ (goodness threshold).

## Memory-movement analysis

Train: 200 K examples × (4 layers × (d_h, d_h) reads + writes) = 5 × 10^10
flops + 2 GB HBM traffic. ~5 s.

Inference: 60 K bytes × 256 candidates × 4 layers × d_h² = 6 × 10^12 flops.
Practical wall-clock 10–30 s if batched.

Total: ~30–60 s wall-clock, ~5 kJ.

## References

- Hinton FF 2022: <https://arxiv.org/abs/2212.13345>
- Ororbia & Mali, Predictive FF 2023: <https://arxiv.org/abs/2301.01452>
- Mono-Forward 2025: <https://arxiv.org/abs/2501.09238>
- DeeperForward (ICLR 2025 — see <https://proceedings.iclr.cc/paper_files/paper/2025/file/7dd309df03d37643b96f5048b44da798-Paper-Conference.pdf>)
