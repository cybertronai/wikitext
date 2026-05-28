# 06 · NoProp-Seq — Per-Block Diffusion-Denoising for Next-Byte Prediction

## Mechanism

NoProp (Li, Teh, Pascanu et al. 2025, arXiv 2503.24322) trains each network
block as an *independent* denoising autoencoder against a fixed
representation of the target label: layer ℓ learns to predict a known
noised version `z_t^ℓ = α_ℓ · embed(label) + σ_ℓ · ε`. There is no
backprop across blocks; each block has its own local optimizer step on its
own local denoising loss. The "global" prediction is the final block
outputting the clean label embedding, decoded via a fixed dictionary.

**NoProp-Seq:** adapt this to next-byte autoregressive LM. Each "block" is a
single per-layer transformation that takes (current_state, noised_target,
context_features) → cleaner state. The label embedding is the *next byte's*
embedding (256-d random Gaussian, fixed at init). For each (window, byte_y)
training pair:

1. Sample noise level α_ℓ for each layer ℓ.
2. Form z^ℓ = α_ℓ · e(byte_y) + √(1−α²_ℓ) · ε.
3. Each layer ℓ learns to predict ẑ^ℓ = f_ℓ(window_features, z^ℓ).
4. Loss per layer: ‖f_ℓ - e(byte_y)‖² + denoising-score regularizer.
5. **No autograd across layers.** Each f_ℓ's parameters are updated by:
   - the local denoising loss (Adam on that block ONLY), OR
   - a **closed-form ridge regression** on (input_features, e(byte_y))
     pairs sampled during the epoch.

If the latter is used, *no SGD anywhere*. If the former, SGD is local to
each block but no global gradient flows.

To keep this within the user's "fully gradient-free" mandate, we adopt the
**closed-form ridge variant**: each block does a single Cholesky solve per
epoch on its accumulated (feature, target) pairs. No optimizer, no backprop.

Inference: each block applies its learned function to a denoising chain
starting from pure noise, producing a clean next-byte embedding. The final
softmax over the byte dictionary gives the distribution.

## Seed papers

- Li, Teh, Pascanu, *NoProp: Training Neural Networks without Full
  Back-propagation or Full Forward-propagation*, arXiv 2503.24322 (2025).
  Establishes the local diffusion-denoising training paradigm.
- Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS
  2020 (arXiv 2006.11239). The denoising mathematics being adapted.
- Hoogeboom et al., *Argmax flows and multinomial diffusion*, NeurIPS 2021
  (arXiv 2102.05379). Discrete-target diffusion that may apply directly to
  bytes.

## Why it could work here

- **Per-block-independent training is embarrassingly parallel.** All blocks
  can train simultaneously on the same data stream, each with its own
  (k, target) pairs.
- **No "deep credit assignment" problem.** Each block has a self-contained
  target. The signal it gets is exactly as strong as the one byte's
  embedding minus noise.
- **Closed-form ridge variant = ZERO optimizer state, zero gradient
  steps.** Pure linear algebra. Wall-clock dominated by data loading.
- **NoProp claims parity with backprop on CIFAR-10 / CIFAR-100.** If even
  half of that quality transfers to byte-level LM, we'd land near 0.55–0.65.
  If it generalizes well, 0.70 is reachable.

## Threshold of plausibility

NoProp's CIFAR-100 result (parity with backprop on a 9-block CNN with
~76% acc) was on a fixed-label task. Char-LM has 60K examples that all
share the same 256-byte dictionary, but with *different conditional
distributions per context*. The conditional structure is what makes NoProp's
fixed-label-embedding assumption fragile here: the "target" of denoising is
not a deterministic class label but a *distribution over bytes*.

Mitigation: replace the label embedding with the *expected* next-byte
embedding under the conditional distribution at this context. Estimate the
conditional via k-NN on a calibration set (cheap RFF + lookup). The target
of denoising becomes a soft byte embedding rather than a hard one. This
keeps the stochasticity filter happy.

The unknown: whether the per-block denoising objective gives strong enough
local signal to learn task-adapted representations *without any
cross-layer gradient flow*. The CIFAR result is encouraging; the LM
adaptation is open.

Estimate: 0.50–0.65. Possibly the highest-probability "novel" candidate to
clear 0.70 in the entire portfolio. **Highest expected information per
joule** in my judgment.

## Failure modes

- **The "label embedding" target is wrong for stochastic context.** Modal
  byte under conditional distribution must replace deterministic byte_y.
  If not done correctly, training collapses (NBB-class failure).
- **Each block's ridge solve has to be re-done as data accumulates.** Use
  a streaming Cholesky update (Higham 2002) rather than re-solving from
  scratch. O(d²) per update.
- **Multi-block stack does not improve over single-block.** This would
  confirm the suspicion that NoProp's CIFAR result is mostly
  per-block-individual-projection power and there's no compositional gain.
  Important to test 1, 2, 4 blocks separately.
- **The denoising chain at inference is iterative.** This could blow the
  300 s cap if too long. Use 3-step DDIM-style sampling rather than full
  100-step DDPM.
- **Wall-clock dominated by data loading.** Once everything is in HBM,
  each block's ridge solve is ~1 s. Move all train bytes to GPU once.

What would falsify it: 4-block NoProp-Seq with d = 256, ridge readout, 3-step
inference, val acc ≤ 0.45 → block-independent denoising does not produce
sufficient credit for byte-level LM. Reject.

## Smallest first experiment

`noprop_seq_v1`:

1. **Embeddings:** byte_id → R^d, d = 256. **Frozen random Gaussian.**
2. **Window feature:** byte window (256 bytes) → R^512 via frozen RFF.
3. **Number of blocks:** 1, 2, 4 (sweep).
4. **For each block ℓ:**
   - Sample noise level α_ℓ (fixed at init): α_1 = 0.9, α_2 = 0.5, α_4 = 0.1.
   - Compute noised target z_ℓ = α_ℓ · embed(byte_y) + √(1-α²_ℓ) · ε.
   - Stack (feature, z_ℓ) → X ∈ R^{(512+256)}.
   - Sample 100 K (X, embed(byte_y)) pairs from train.
   - Solve closed-form ridge: W_ℓ = (X^T X + λI)^{-1} X^T Y.
   - Store W_ℓ.
5. **Inference (`predict`):**
   - Compute feature φ from current window.
   - Initialize z = pure noise.
   - For ℓ = 1 → L: z = W_ℓ · [φ; z]
   - Final z is the predicted byte embedding.
   - Distribution: softmax of `z @ embedding_table^T` (256 dot products).
6. **No streaming state needed.** `observe()` just updates the rolling
   window. Stateless beyond the byte history.

Sweep: L ∈ {1, 2, 4}, λ ∈ {1e-4, 1e-3, 1e-2}, d ∈ {128, 256, 512},
ctx_len ∈ {64, 128, 256}.

## Memory-movement analysis

Train: 4 ridge solves on (1024+, d) matrices = 4 × O((512+256)²) ≈ 10^9 flops
total. Sub-second on A100.

Inference: 4 matmuls of (1, 768) × (768, 256) per byte × 60 K bytes = 200 M
flops. < 1 s.

Param count: 4 blocks × (768 × 256) ≈ 800 K floats. < 4 MB.

Total: wall-clock < 60 s, energy ~3-6 kJ. Sub-second train and
sub-millisecond per-byte inference if implemented right.

## References

- NoProp paper: <https://arxiv.org/abs/2503.24322>
- DDPM: <https://arxiv.org/abs/2006.11239>
- Multinomial diffusion: <https://arxiv.org/abs/2102.05379>
