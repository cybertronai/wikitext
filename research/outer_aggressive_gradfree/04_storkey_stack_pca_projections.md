# 04 · Storkey Associative Memory Stack with Closed-Form (PCA/CCA) Projections

## Mechanism

Build the LM as a *stack* of classical Hopfield-Storkey associative memories
(Storkey 1997). Each layer ℓ holds a square weight matrix W^ℓ ∈ R^{d×d}
written incrementally by the Storkey rule:

    W^ℓ_{new} = W^ℓ_{old} + (1/n)(x x^T - x h^T - h x^T),
    h = W^ℓ_{old} x

where x is the d-dim binary pattern (sign of the projected hidden state) at
that layer. The Storkey rule has 1.7× the capacity of vanilla Hebbian and
is still strictly local + incremental.

The between-layer projections are trained by **CCA / PCA on activations**,
NOT by SGD:

- Bottom projection P_0: closed-form PCA on byte-window one-hot inputs over
  a 100 K-byte calibration set. Captures byte-bigram-trigram structure.
- Middle projections P_ℓ (ℓ ≥ 1): closed-form CCA between layer-ℓ activations
  and the next-byte one-hot targets, again over 100 K calibration tokens.
  This is the Belilovsky-Eidnes 2019 / Nøkland-Eidnes 2019 *layerwise greedy
  CCA-trained projections* idea, but with Storkey associative memories
  between them instead of standard ReLU MLPs.

Readout: after the top Storkey settles, the converged state is mapped to
256-d byte logits by a **closed-form multiclass ridge regression** on
(top-state, byte) pairs.

**No SGD anywhere.** All trainable matrices come from PCA / CCA / Storkey
/ ridge — all closed-form.

## Seed papers

- Storkey, *Increasing the capacity of a Hopfield network without
  sacrificing functionality*, ICANN 1997. Local incremental rule with
  ~1.7× Hebbian capacity.
- Belilovsky, Eidnes, Solbrig et al. 2019, *Greedy Layerwise Learning Can
  Scale to ImageNet*, ICML 2019. Closed-form / layer-local training beats
  expectations.
- Nøkland & Eidnes, *Training Neural Networks with Local Error Signals*,
  ICML 2019. CCA-objective for layer training.
- Cho & Saul, *Kernel Methods for Deep Learning*, NIPS 2009. Foundation for
  arc-cosine kernel + closed-form readouts that DON'T break under depth.

## Why it could work here

- **Entirely closed-form.** No iterative optimization at all. Train pass
  is 4–6 PCA/CCA computations + a few outer-product writes + one ridge solve.
  Wall-clock should be under 60 s.
- **Storkey's incremental + local property** means the FW updates can run
  on the GPU as a simple sequence of bmm calls; the chunkwise-parallel form
  is straightforward (same WY-Householder trick as DeltaNet).
- **PCA / CCA preserve modal structure.** A learned projection that
  maximizes correlation with next-byte targets is exactly what beats the
  paradigm-A ceiling on representation. Crucially this is *not random* — it
  IS task-adapted, just via a closed-form objective.
- **Calibration-set CCA fits in a few seconds on A100** with covariance
  matrices of size d² ≤ 65 K.

## Threshold of plausibility

The paradigm-A ceiling at 0.37 was hit by methods with frozen/random feature
extractors. CCA-projections are *task-adapted* features and therefore
shouldn't have that ceiling — they're closer to the Cho-Saul deep-kernel
regime.

The question is whether a 4-layer Storkey stack with CCA-trained
projections has enough representational depth to clear 0.70. Belilovsky 2019
showed layer-local training can match end-to-end backprop on ImageNet
classification, but with 6+ blocks of ReLU CNN. With Storkey associative
memories — which have storage capacity ~n/√(2 log n) ≈ 80 patterns at
d = 256 — the representational power per layer is limited.

Best estimate: 0.45–0.60. Unlikely to clear 0.70 because Storkey's
discrete sign-attractor is awkward for soft per-byte probability emission.

This is therefore a **capability demo** + **calibration tool** ("how far
can fully closed-form non-NN training go on this benchmark?"). If it
beats the existing kernel_methods sweep's ~0.36 ceiling by margin, that's
a notable result on its own.

## Failure modes

- **Storkey rule requires bipolar patterns** (±1). Quantizing real-valued
  hidden states to ±1 throws away most information. Mitigation: a soft
  Storkey variant using `tanh(x)` instead of `sign(x)`, accepting a small
  capacity hit.
- **CCA calibration set has stochasticity.** Each context maps to *many*
  next-byte labels with different frequencies. CCA on (context, modal_byte)
  pairs is more stable than CCA on (context, sampled_byte). Use modal-byte
  CCA per byte-equivalence-class (cluster contexts by trigram suffix first).
- **Sign-flip iteration during settle.** Hopfield/Storkey converge to a
  fixed point; iteration depth affects expressivity. Mitigation: single-step
  read (à la Ramsauer high-β) and let the high-d feature space + softmax
  do the work.
- **The "stochasticity filter" applies** — single-WTA Storkey read would
  catastrophically fail. We MUST use a softmax over Storkey energy
  per-byte rather than an argmax of converged state. This is the same fix
  applied in #01.
- **n/√(2 ln n) capacity exhausted at small d.** With d = 256 and the
  effective number of distinct contexts in 540 MB ~ 10^8, the Storkey
  matrix is *vastly* undercapacity. Mitigation: don't store all contexts;
  store only modal-byte representatives per context-cluster. The
  Belilovsky CCA pipeline produces a natural cluster index.

What would falsify it: 4-layer Storkey with d = 512, CCA-trained
projections, ridge readout — val acc ≤ 0.45. That would tell us that
closed-form non-NN methods cannot do byte-level LM, period. Reject.

## Smallest first experiment

`storkey_stack_cca_v1`:

1. **PCA layer:** sample 200 K (256-byte window) examples from train.
   One-hot encode (one-hots of byte_id pair per offset), flatten to a
   256·8 = 2048-dim vector. PCA → d = 512 components. Project all bytes
   through.
2. **Storkey layer 0:** at each token, accumulate
   `W^0 += (sign(x) sign(x)^T - sign(x) h^T - h sign(x)^T) / N` where
   h = W^0 sign(x). Use a chunkwise parallel form (the same WY scan
   already in `hebbian_fw_block_v2` works with minor sign changes).
3. **CCA bridge to layer 1:** sample (layer-0 settled state, next-byte
   one-hot) pairs over 100 K tokens, compute closed-form CCA, take top-d
   canonical directions as P_1.
4. **Repeat steps 2-3 for layers 1, 2, 3.**
5. **Ridge readout:** sample (top-state, byte one-hot) pairs, closed-form
   ridge with λ = 1e-3. Cap output via softmax(scale · ridge_logit).

Train phase: 4 PCA/CCA + 4 Storkey passes + 1 ridge ≈ 30–60 s wall-clock.
No iteration. No backprop.

## Memory-movement analysis

PCA / CCA: covariance matrices of size d² = 0.25 M floats per layer.
Eigendecomposition is O(d³); at d = 512 that's 10^8 flops — negligible.
Storkey writes: one (d, d) outer product per chunk of 64 tokens, 200 K
tokens / 64 = 3 K chunks, each (32, d, d) write = 8 M floats. Total
HBM write: 24 G floats × 4 bytes = 100 GB — actually significant on a
GPU with 2 TB/s bandwidth → ~50 ms.

Inference: one CCA-projection (d²) plus one Storkey-energy softmax read
per byte. With 60 K val bytes: 4 × d² × 60 K = 60 G flops total → ~0.1 s.

**Total energy budget: ≤ 5 kJ if it converges at all.** This is the
cheapest credible submission in the portfolio.

## References

- Storkey, ICANN 1997: <https://homepages.inf.ed.ac.uk/amos/learning.html>
  (also <https://link.springer.com/chapter/10.1007/BFb0020196>)
- Belilovsky, Eidnes et al., ICML 2019: <https://arxiv.org/abs/1812.11446>
- Nøkland & Eidnes, ICML 2019: <https://arxiv.org/abs/1901.06656>
- Cho & Saul, NIPS 2009 (deep kernels): <https://papers.nips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html>
