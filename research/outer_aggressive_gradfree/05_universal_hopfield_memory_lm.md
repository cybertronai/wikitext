# 05 · Universal Hopfield Memory LM — Similarity-Separation-Projection

## Mechanism

Millidge, Salvatori, Song et al. 2022's *Universal Hopfield Networks*
(arXiv 2202.04557, ICML 2022) decompose every Hopfield-style associative
memory model into three operations applied in sequence to a query q against
a stored set {(k_i, v_i)}:

1. **Similarity:** s_i = sim(q, k_i)        — e.g. dot product, Euclidean,
                                                Manhattan
2. **Separation:** σ_i = sep(s_i)            — softmax, identity, max
3. **Projection:** o = Σ_i σ_i v_i           — linear combination

Different choices recover the classical Hopfield network, SDM, modern
continuous Hopfield, etc. **The key insight for this experiment:** the
three components are mechanistically independent and can be implemented
with three completely different gradient-free methods.

Proposed instantiation for char-LM:

- **Similarity = RFF kernel.** Both q and k_i are projected by a frozen RFF
  feature map, so similarity is approximately the Gaussian kernel between
  byte-window embeddings.
- **Separation = softmax with adaptive temperature.** Closed-form
  temperature β = 1/σ̂(s) where σ̂ is the std of similarities over a
  calibration set (computed once, never updated).
- **Projection = closed-form ridge regression** on (k_i, v_i) pairs, where
  v_i = onehot(next_byte_i). The ridge gives a learned-but-closed-form
  combiner over a fixed support set.

The (k, v) memory is grown by **online Hebbian outer-product accumulation**
on the train stream — every byte adds one (encoded_window, onehot_next_byte)
pair. At the end of training, the entire model is:

    predict(window):  q = φ_RFF(window);  s = q^T K_mem;  σ = softmax(β s);
                       logits = σ V_mem (ridge-corrected by P)

where P is a fixed (V, V) precision matrix from the ridge solve. The whole
prediction is two matmuls and a softmax. No state to carry, no recurrence.

**No gradient flows.** Memory is built by additive outer products. Readout
is fitted by one matrix inverse.

## Seed papers

- Millidge, Salvatori, Song, Lukasiewicz, Bogacz, *Universal Hopfield
  Networks*, ICML 2022 (arXiv 2202.04557). The three-operation decomposition.
- Ramsauer et al., ICLR 2021 (arXiv 2008.02217). Modern continuous Hopfield;
  the softmax-separation case is identical to modern attention.
- Kanerva, *Sparse Distributed Memory*, MIT Press 1988. The threshold
  similarity + linear separation case.
- Schaeffer 2022 blog summary of UHN: <http://rylanschaeffer.github.io/blog_posts/2022-09-08-Universal-Hopfield-Networks.html>

## Why it could work here

- **Compositional gradient-free design.** Each of the three operations is
  independently swappable. If the experiment fails, the post-mortem will
  cleanly identify which component is responsible.
- **Closed-form everywhere.** No iterative training. No optimizer.
- **Mathematically equivalent to the existing `hopfield_layer` submission**
  *minus* the surrounding transformer body. If the surrounding transformer
  was the source of the 0.729 acc and the Hopfield contributed nothing,
  this experiment will reveal that immediately. **Direct ablation of the
  current winner.**
- **Memory grows online**, so the V_mem bank at the end of training
  contains O(540 M) entries — but with online streaming Hebbian
  accumulation, only the *consolidated* outer-product accumulator is kept
  (size d × 256), not the full N entries. This is the key compute win
  over a Memorizing-Transformers-style explicit kNN cache.

## Threshold of plausibility

The 0.37 paradigm-A ceiling applies to fixed-feature + closed-form-readout
methods. UHN with RFF similarity + softmax separation + ridge projection
IS such a method. **Most likely it hits the same ceiling.**

What might push it above 0.37: the *separation* step with task-adaptive
β acts like learned mixing, and the *projection* step with ridge correction
is more powerful than a plain linear readout. With enough storage capacity
(M = 1 M consolidated patterns at d = 512) and a carefully tuned RFF
bandwidth, **plausibly 0.50–0.60.**

Clearing 0.70 requires representation learning, which UHN does not do.
This is a **capability demo** for the UHN-decomposition framing, valuable
as the cleanest possible test of "is task-adapted similarity + soft
read + closed-form projection enough to escape paradigm-A?"

## Failure modes

- **0.37 paradigm-A ceiling reasserts itself.** The most likely outcome.
- **RFF bandwidth is wrong.** RBF kernel on RFF-projected byte windows
  has no natural σ; calibrate via median-pairwise-distance heuristic on
  a sample.
- **Online Hebbian accumulator drift.** As patterns are added one by one,
  the (k, v) outer-product matrix grows; without normalization it diverges.
  Mitigation: store running mean of `k_i v_i^T` instead of sum, dividing
  by counter at end of training.
- **Stochasticity:** the softmax-separation + ridge-projection combination
  emits a soft 256-d distribution, so the stochasticity filter passes.
- **Ridge solve numerical instability** at large M / V. Use Cholesky on
  the regularized Gram matrix; floor λ at 1e-3.

What would falsify it: M = 256 K consolidated patterns, d = 512, β tuned,
ridge λ tuned — val acc ≤ 0.40 → confirmed paradigm-A ceiling. Reject.

What it could verify: with the *same* mechanism as `hopfield_layer` but
*no* transformer body, val acc = 0.65 → that hybrid's win is mostly the
Hopfield component, and `hopfield_layer` is over-spending on its
transformer.

## Smallest first experiment

`uhn_lm_v1`:

1. **Two frozen random projections** — `R_k: byte_window → R^d_feat` and
   `R_v: byte → R^d_v`. d_feat = 512, d_v = 64.
2. **Build (K, V) bank by streaming Hebbian accumulation:**
   for each token in 5 M-byte sample of train stream:
     k_t = R_k(window_t); v_t = onehot(byte_{t+1})  ∈ R^256
     K_acc += k_t k_t^T / N        # (d_feat, d_feat)
     KV_acc += k_t v_t^T / N        # (d_feat, 256)
   Final outputs are just `K_acc` and `KV_acc`.
3. **Closed-form Ridge on (k, v):** solve W = (K_acc + λI)^{-1} KV_acc.
   W: (d_feat, 256). One Cholesky.
4. **Calibrate β:** on a held-out 10 K tokens, compute median similarity
   scale s̄; set β = 1/s̄.
5. **Inference (`predict`):** for current window, q_t = R_k(window_t);
   logits = β · q_t · W; softmax. One matmul of size 1 × 512 × 256 per
   byte. < 1 ms per byte.
6. **Sweep:** d_feat ∈ {256, 512, 1024}, λ ∈ {1e-4, 1e-3, 1e-2}, ctx_len ∈
   {32, 64, 128}. 9 submissions.

Note: this is mathematically a kernel ridge regression on RFF features
mapping windows to byte distributions. We *expect* it to land near
`rff_ridge_v1` (0.364 acc, 2.5 kJ). The novelty is the framing as a
UHN instantiation and the comparison to `hopfield_layer` to localize
what the latter is buying.

## Memory-movement analysis

Train: stream M = 5 M tokens through (d_feat) projection. Accumulator
updates: 2 × d_feat² flops per token = 5 × 10^11 flops total. A100 fp16
~312 TF → < 5 s pure compute. Single Cholesky on (d_feat) matrix:
O(d_feat³) ≈ 10^8 flops. Total train: well under 30 s.

Inference: 1 matmul of 1 × 512 × 256 per byte × 60 K bytes = 8 M flops
= sub-ms. Practically instant.

Total submission: < 60 s, < 5 kJ expected.

## References

- Millidge et al., ICML 2022: <https://arxiv.org/abs/2202.04557>
- Ramsauer et al., ICLR 2021: <https://arxiv.org/abs/2008.02217>
- Rylan Schaeffer summary blog:
  <http://rylanschaeffer.github.io/blog_posts/2022-09-08-Universal-Hopfield-Networks.html>
- Kanerva 1988, *Sparse Distributed Memory* (MIT Press).
