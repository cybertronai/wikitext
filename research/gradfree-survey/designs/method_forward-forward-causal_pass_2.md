# Causal Forward-Forward + Cascaded Ridge Readout (pass 2)

Direction: **D** -- train an FF stack as in pass 1 (faster, narrower) and discard the goodness-as-likelihood predictor. Instead, fit a closed-form ridge regression from the concatenation of all FF layer activations (on the context only, no candidate-byte slot) to the next-byte one-hot. Goodness is used purely as a *representation-learning* objective; prediction is a separate, gradient-free linear readout.

## 1. Hypothesis
Pass-1's goodness softmax over 256 candidates is a weak score: it ranks bytes by how much each blows up activation norm, not by next-char likelihood. The FF stack *does* learn distributional structure (0.235 > 0.18 unigram), but the readout throws most of it away. A ridge regression from concatenated layer activations -- fit in closed form, no SGD, gradient-free across the stack -- should extract substantially more signal. Honest target: **val_char_acc 0.35-0.45**, a 1.5-2x lift over pass 1. Stretch 0.55. Bar (0.70) still unlikely in 5 min but no longer absurd.

## 2. Model
- **FF backbone** (representation, trained by local Forward-Forward rule):
  - Input: rolling K=24 char one-hot window concatenated with a 1-hot candidate next-byte -> dim 6400. Same as pass 1 (the candidate slot is needed during FF training so positive/negative pairs exist).
  - 5 FC layers, width 384, ReLU, no bias. (Was 6x512 in pass 1; we shrink to free wall time for the longer schedule and the readout-time forward passes.)
  - L2 layer-norm between layers (no learned scale), identical to pass 1.
  - Layer 1 frozen-random; layers 2-5 trained by FF.
- **Ridge readout** (predictor, gradient-free closed-form):
  - At feature-extraction time the input is the K=24 context concatenated with an **all-zeros candidate slot** (dim 6400). We do NOT iterate over 256 candidates here -- one forward per char.
  - Feature vector phi(ctx) = concat( LayerNorm(a_2), LayerNorm(a_3), LayerNorm(a_4), LayerNorm(a_5) ) -> dim 4*384 = 1536. Skip layer 1 (frozen random, low value) and skip the raw input.
  - Targets: 256-dim one-hot of the true next byte.
  - Solve W = (Phi^T Phi + lambda I)^(-1) Phi^T Y in float32 on GPU. Predict by argmax over W^T phi(ctx) at eval.

## 3. Training procedure
1. **FF phase (round-robin local updates, ~120 s)**
   - Identical loop to pass 1 but with the harder-negatives refinement: every 500 steps, regenerate 50% of the negative batch by sampling next-byte from the *model's own current ridge-readout distribution* (re-fit on a 20K-sample cache). For steps before the first re-fit (step 500), use unigram negatives.
   - 14000 round-robin steps at B=256 (vs 8000 in pass 1; smaller layers compensate for more steps).
2. **Ridge fit phase (closed form, ~30 s)**
   - Sample N_fit = 80000 (context, true_byte) pairs from training stream.
   - Forward each through the FF stack with zero-candidate input -> Phi (80000, 1536).
   - Build Y (80000, 256) one-hot. Solve normal equations on GPU: W = solve(Phi^T Phi + lambda I, Phi^T Y), lambda = 1.0.
3. **Eval phase (one forward per char, ~80 s)**
   - For each of 60K val chars: forward context-only (B=1, zero candidate slot) through FF -> phi. logits = W^T phi. predict = argmax. **One forward per char, not 256.** This is the major speedup that funds everything else.

## 4. Hyperparameters
- L = 5 FF layers, width 384, layer 1 frozen.
- K = 24, input dim 6400.
- theta = 2.0, per-layer Adam lr=3e-4, betas=(0.9,0.99), no weight decay. 4 per-layer optimizers.
- Batch B = 256. n_steps = 14000.
- Hard-negative refresh every 500 steps after step 500; 50% replacement from ridge-readout top-K (K=5) excluding true byte.
- Ridge: N_fit = 80000, feature dim 1536, target dim 256, lambda = 1.0, solved in float32 on GPU.
- SEED honored for init, minibatch indexing, negative sampling, and ridge sample selection.

## 5. Expected wall time (A100-80GB)
- FF training: 14000 steps * ~3 ms = ~42 s (smaller layers, no eval-in-loop). Add ~10 s for periodic ridge re-fits (mini, N=20000) -> ~55 s.
- Ridge fit: 80000 forwards (B=512 batched -> 156 batches) at ~2 ms = ~0.3 s feature extraction; matrix solve on (1536,1536) trivial -> total ~5 s. Call it 15 s with overhead.
- Eval: 60000 single-context forwards. At B=256 batching across consecutive chars (we can batch up to 256 stream-positions per forward since each is independent given its window) -> 60000/256 = 235 batches * ~3 ms = ~1 s of GPU. Realistically with Python overhead and the per-char window updates: ~60-80 s.
- Setup + cuda init + encoding: ~20 s.
- **Total: ~55 + 15 + 80 + 20 = ~170 s.** Comfortably under 300 s. Budget headroom (~130 s) is reserved for the eval-batching overhead if it goes worse than projected.

## 6. Success criterion
- **Honest target: val_char_acc >= 0.35** on first 60K val chars (1.5x pass-1's 0.235; clears that the FF representation is non-trivial).
- Stretch: 0.45. Bar-clear (0.70): not expected.
- Energy: budget < 6 kJ (eval is ~250x cheaper than pass 1's 256-candidate-per-char scheme; pass 1 spent 2.7 kJ in 96 s, this run is ~170 s but at lower GPU utilization during eval -> ~5 kJ projection).
- Primary deliverable: does decoupling representation (FF) from prediction (ridge) recover the signal that goodness-softmax loses?

## 7. Failure modes anticipated
- **Ridge underfits**: 1536 features may be too few to span 256 classes well. Mitigation: if val < 0.30 at end, also concatenate quadratic features (elementwise a_l * a_{l+1}) for layers 2-3 -- still closed form, adds ~150K features. Skip if wall budget tight.
- **Feature collinearity blows up the solve**: lambda=1.0 should handle it; bump to 10.0 if numeric warnings appear.
- **Hard-negative loop unstable**: ridge readout used to generate negatives is itself learned from FF features -> feedback. Mitigation: cap hard-negative fraction at 50%, keep 50% unigram for ballast. If goodness gap collapses, fall back to pure unigram negatives for remainder.
- **Eval batching across stream positions breaks causality bookkeeping**: each forward needs the window ending at position t. Pre-build all 60K windows as one (60000, 6400) tensor upfront; then chunked B=256 forwards are trivially correct.
- **FF backbone learns nothing useful for a linear readout**: possible if local goodness optimum is orthogonal to next-char-prediction subspace. Diagnostic: also fit a ridge readout from a *random-projection* baseline (same arch, all layers frozen-random). If FF ridge ~= random-projection ridge, FF added zero representational value -- a clean negative result worth reporting.

## 8. What we will NOT do
- NOT use end-to-end backprop. FF rule is local-per-layer (one detached input, one layer's grad). Ridge is closed-form -- no gradient at all.
- NOT pretrain any component. Layer 1 frozen-random at init; layers 2-5 trained from scratch by FF; ridge fit on FF-extracted features only.
- NOT mix cross-entropy or any other end-to-end loss into FF training. Goodness is the only training signal for the backbone.
- NOT use attention, recurrence, residuals, BatchNorm. FC stack with L2 layer-norm only.
- NOT use Hinton top-down feedback negatives. External negatives only (unigram + hard ridge-top-K).
- NOT exceed K=24 context.
- NOT touch the val stream during ridge fit (fit only on training-stream chars).

---

Layer width: 384 FF (5 layers, layer 1 frozen-random; ridge readout 1536 -> 256, lambda=1.0).
Success criterion: val_char_acc >= 0.35 honest / 0.45 stretch / 0.70 unlikely upside; energy budget < 6 kJ; wall < 300 s.
