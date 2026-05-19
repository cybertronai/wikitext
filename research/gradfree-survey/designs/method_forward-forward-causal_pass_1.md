# Causal Forward-Forward Char-LM (pass 1)

## 1. Hypothesis
A 6-layer fully-connected stack trained by Hinton's Forward-Forward rule on causal char windows will learn enough local distributional structure to score above unigram baseline, but very unlikely to reach 0.70. We are testing whether the goodness-as-likelihood-surrogate signal carries usable LM gradient when the "label" being scored is a 1-of-256 next-byte slot. Honest expectation: 0.30-0.50. The value of running this is family coverage (local learning, no end-to-end backprop) and quantifying *how much* worse FF is than cross-entropy on the same compute -- a clean negative result is informative.

## 2. Model
- Input encoding: rolling K=24 char one-hot window concatenated with a 1-hot candidate next-byte -> input dim = (24 + 1) * 256 = 6400. Concat (not interleaved) so the candidate slot is always at the end.
- 6 fully-connected layers, width 512 each. Activation: ReLU. No biases (standard FF). No residuals.
- Between layers, layer-normalize the activation vector along the feature axis (divide by L2 norm, no learned scale) before feeding to the next layer. This strips magnitude info -- the only signal carried up is direction.
- Goodness per layer: G_l(x) = sum_i a_l[i]^2 over the 512 ReLU outputs of layer l.
- Predictor head: total goodness G(x) = sum_{l=2..6} G_l(x) (skip layer 1 per Hinton's recipe -- its goodness leaks too directly from input norm). At eval, batch all 256 candidate next-bytes for the same context into a single B=256 forward pass; softmax over G across candidates yields the 256-byte distribution.
- Yes, we batch the 256 candidates per query as a single B=256 minibatch (one forward through the stack per char to predict).

## 3. Training procedure
- Positive sample: real (context_24, true_next_byte) pair, both one-hot, concatenated to 6400-dim.
- Negative sample: same context_24, with next-byte drawn from the **unigram** distribution over training bytes, rejected and resampled if it equals the true byte. Unigram (not uniform) so the model can't trivially learn "frequent byte = positive".
- Per step (round-robin schedule -- all layers updated per minibatch):
  1. Sample a minibatch of B=256 (context, true_byte) pairs. Build x_pos (B, 6400) and x_neg (B, 6400).
  2. Forward through layer 1 with **no grad** on the input -> a1_pos, a1_neg. LayerNorm both. (Layer 1 weights are frozen at init; per Hinton we don't train layer 1.)
  3. For l = 2..6:
     - With grad enabled only on layer l's weights, compute a_l_pos = ReLU(W_l @ a_{l-1}_pos_normed), a_l_neg = ReLU(W_l @ a_{l-1}_neg_normed). The input a_{l-1}_*_normed is detached.
     - G_pos = (a_l_pos ** 2).sum(dim=-1); G_neg = (a_l_neg ** 2).sum(dim=-1).
     - L_l = softplus(theta - G_pos).mean() + softplus(G_neg - theta).mean().
     - opt_l.zero_grad(); L_l.backward(); opt_l.step(). (Each layer has its own Adam optimizer; gradient only touches W_l because input is detached.)
     - Detach a_l_pos, a_l_neg, then LayerNorm them as input to layer l+1.
- Schedule: 8000 round-robin steps total (each step touches all 5 trained layers). Negatives regenerated every step.

## 4. Hyperparameters
- Layers L = 6 (layer 1 frozen-random projection, layers 2-6 trained).
- Width = 512, ReLU, no bias.
- K = 24, input dim 6400.
- theta = 2.0 (goodness threshold; tunable in [1, 5] if collapse observed).
- Per-layer optimizer: Adam, lr=3e-4, betas=(0.9, 0.99), no weight decay. One optimizer per trained layer (5 optimizers).
- Batch size B = 256.
- n_steps = 8000 round-robin steps.
- Negative sampler: training-corpus unigram, rejection-sampled to exclude true byte.
- SEED honored via os.environ["SEED"] for init, minibatch indexing, and negative sampling.

## 5. Expected wall time (A100-80GB)
- Per training step: 2 (pos+neg) * 5 trained layers * one (B=256, 512x512) matmul + activation + goodness sum ~= 10 * (256 * 512 * 512) = ~6.7e8 FLOPs forward + ~3x for local backward = ~2e9 FLOPs/step. At 100 TFLOP/s sustained bf16 -> ~20 us/step ignoring overhead. Realistic with Python overhead: ~3 ms/step * 8000 steps = ~24 s training.
- Eval cost is the dominant risk. 60K predict() calls, each a B=256 forward through 6 layers of 512-wide FC = 256 * 6 * 512 * 512 = ~4e8 FLOPs per char. Wall: ~2-4 ms per predict() on A100 (matmul is tiny, Python launch overhead dominates). 60K * 3 ms = **180 s**. observe() is cheap (just shifts the rolling window).
- Encoding + setup + first cuda init: ~15 s.
- Total: 24 + 180 + 15 = **~220 s**, under 300 s with margin. If eval blows up, we cut to K=16 or width=384 (the dominant cost is eval, not training).

## 6. Success criterion
- Honest target: val_char_acc >= 0.25 on first 60K val chars (well above the ~0.18 unigram baseline; demonstrates FF carries *some* LM signal at char level).
- Stretch: 0.40. Bar-clear (0.70): not expected; if hit, that's a publishable surprise.
- Energy: budget < 25 kJ (eval-dominated; FF training itself is cheap).
- Primary deliverable is the joules-per-acc datapoint vs. the cross-entropy baselines, even if acc is low.

## 7. Failure modes anticipated
- **Eval too slow**: 256-hypothesis batching per char is the bottleneck. If wall > 280 s, fall back to K=16 + width=256 (4x cheaper eval) before considering Reducing layers.
- **Goodness collapse**: all examples produce the same goodness (positive == negative). Diagnostic: track G_pos - G_neg gap each 200 steps; if < 0.1 by step 1000, raise theta to 4.0 or lower lr to 1e-4.
- **LayerNorm wiping signal**: stripping magnitude may leave too little for deeper layers to discriminate. Diagnostic: per-layer val acc using only that layer's goodness. If layer 6 < layer 2, depth isn't helping -- report and stop.
- **Unigram negatives too easy**: model just learns byte-frequency. Mitigation: every 1000 steps, regenerate 50% of negatives by sampling from the model's own predict() distribution at that context (self-negatives, Hinton's later refinement).
- **No public FF char-LM precedent**: we may discover the method simply doesn't transfer. That is a valid finding for the survey.

## 8. What we will NOT do
- NOT use end-to-end backprop. The local backward through *one* layer (with input detached) is allowed and explicitly defined as the FF gradient rule.
- NOT pretrain any component (no embedding init from word2vec / random projection trained beforehand). Layer 1 is frozen at random init.
- NOT mix in a cross-entropy auxiliary loss or a linear classifier head. Goodness is the only score.
- NOT use a recurrent or attention layer. FC stack only.
- NOT use BatchNorm or residual connections. Only the prescribed L2 layer-normalization between layers.
- NOT use Hinton's "negative pass via top-down feedback" variant -- too complex for one spec. Stick with externally-generated negatives.
- NOT exceed K=24 context. FF goodness over long contexts is uncharted; we stay near the Gandhi & Gala scale.

---

Layer width: 512 (all 6 FC layers; layer 1 frozen-random, layers 2-6 trained by local FF rule).
Success criterion: val_char_acc >= 0.25 honest / 0.40 stretch / 0.70 unlikely upside; energy budget < 25 kJ.
