# 08 · Streaming Memorizing-LM — Hebbian Write, kNN Read, RFF Features, Ridge Readout

## Mechanism

Wu, Rabe, Hutchins, Szegedy 2022 (*Memorizing Transformers*, ICLR 2022,
arXiv 2203.08913) inserted a non-differentiable external kNN-cache attention
layer into a transformer. Approximate kNN lookup over millions of past
(key, value) pairs improved long-context LM. The keys/values come from a
transformer block — i.e. they require backprop.

**This direction strips backprop entirely.** The "Memorizing-LM" is just the
external memory + a fixed feature extractor + a closed-form readout. There
is NO transformer body.

Architecture:

1. **Frozen feature extractor:** byte_window → R^d via two stacked frozen
   RFF projections with ReLU between (arc-cosine kernel, depth-2). d = 512.
2. **External memory M:** a streaming buffer of (k_i, v_i) pairs where k_i
   is the projected window features at training position i and v_i is the
   *embedding* of byte_{i+1}. The buffer grows as training progresses; old
   entries are dropped FIFO when capacity (e.g. M_max = 1 M entries) is hit.
3. **Hebbian-consolidated covariance:** alongside the explicit buffer, we
   maintain `K_acc = Σ k_i k_i^T` and `KV_acc = Σ k_i v_i^T` updated by
   Hebbian outer-product writes (this consolidates the FIFO buffer into a
   single (d, d) and (d, V) matrix).
4. **Hybrid read:** given query q at inference, return the *mixture* of
   (a) explicit kNN over the buffer with top-K = 32 nearest neighbors, and
   (b) closed-form ridge prediction using K_acc, KV_acc.
5. **Readout:** logits = mix_weight · kNN_logits + (1 - mix_weight) · ridge_logits.
   mix_weight is tuned on a 1K-byte calibration sample.

**No gradient. No backprop. No SGD anywhere.** Train is one pass over
the stream with one Hebbian outer-product per token.

## Seed papers

- Wu, Rabe, Hutchins, Szegedy, *Memorizing Transformers*, ICLR 2022
  (arXiv 2203.08913). Establishes kNN-augmented attention with
  non-differentiable memory.
- Cho & Saul, *Kernel Methods for Deep Learning*, NIPS 2009. Arc-cosine
  kernel = two-layer ReLU network → closed-form features.
- Rahimi & Recht, *Random Features for Large-Scale Kernel Machines*, NIPS
  2007. The RFF construction.
- Khandelwal et al., *Generalization through Memorization: Nearest Neighbor
  Language Models (kNN-LM)*, ICLR 2020 (arXiv 1911.00172). The kNN-LM
  reading mechanism we re-use.

## Why it could work here

- **Direct ablation of the `hopfield_layer` winning recipe.** The Hopfield
  submission's memory bank is *frozen at init*; here it grows online via
  Hebbian writes during the training pass. If our setup converges to the
  same accuracy, the Hopfield's frozen bank is wasteful and we save energy.
- **No transformer body to train.** Avoids the entire AdamW+Muon overhead.
- **kNN at inference is fast on A100** with FAISS-style flat L2 / cosine
  index. At M = 1 M and d = 512, top-K = 32 search per byte is ~1 ms.
- **The arc-cosine kernel** gives provable function-class universality for
  the RFF projections — they aren't just noise, they're a learned-free
  kernel that has been shown to handle text features.
- **Memory bank covers the entire 540 MB train stream**, not just 4096
  windows. Capacity for context-specific patterns is orders of magnitude
  larger than `hopfield_layer`.

## Threshold of plausibility

The kNN-LM line of work showed that frozen-feature kNN over a large memory
*adds 5–10% to a strong neural LM* on WikiText. Standalone kNN-LM (without
the neural LM partner) is far worse — they reported 50%+ ceiling at val
perplexity, well below a strong NN.

For char-LM at 0.70 acc, the question is whether the RFF + arc-cosine
features (paradigm-A in framing) carry enough representation to beat the
0.37 ceiling. The combination of kNN (locally adaptive) + ridge (global
linear projection) is more expressive than either alone, possibly enough
to push to 0.50–0.60.

But clearing 0.70 likely requires representation learning, which arc-cosine
RFF does not provide. This is on the same ceiling line as #5.

Estimate: 0.45–0.60. **Capability demo** for "online Hebbian kNN-LM as
standalone model".

## Failure modes

- **kNN dominance at inference returns memorized exact bytes**, not modal
  bytes — stochasticity-filter failure. Mitigation: temperature softmax
  over top-K rather than top-1; smoothing toward the consolidated K_acc
  ridge prediction.
- **Memory consolidation drift.** As K_acc grows linearly with N, the
  ridge matrix changes after every batch. Use streaming Cholesky update.
- **FAISS / kNN index rebuild cost** at each insertion: index entries get
  added incrementally during training pass. With M = 1 M and FAISS IVF
  index, this is ~30 s additional wall-clock. Acceptable.
- **Arc-cosine kernel may not capture byte-level local structure** (it's
  a smooth kernel with no positional bias). Mitigation: feature includes
  byte position information via Fourier features (RoPE-style).

What would falsify it: M = 1 M, d = 512, arc-cosine RFF, mix kNN + ridge,
val acc ≤ 0.45 → online Hebbian + kNN cannot escape paradigm-A. Reject.

What it would verify: val acc near `hopfield_layer`'s 0.73, with no
transformer body → confirms that the Hopfield submission's win was
nearly entirely the Hopfield component (and we save energy by dropping
the transformer).

## Smallest first experiment

`stream_memorize_v1`:

1. **Frozen arc-cosine RFF:** two stacked random Gaussian projections with
   ReLU between, projecting byte_window → R^512. ctx_len = 128.
2. **Train pass over 5 M tokens:**
   - Per token: k_t = features(window_t), v_t = embed_byte(byte_{t+1}) (256-d).
   - Append (k_t, v_t) to FAISS IVF index.
   - Accumulate K_acc += k_t k_t^T / N, KV_acc += k_t v_t^T / N.
3. **Closed-form ridge solve:** W = (K_acc + λI)^{-1} KV_acc.
4. **Calibrate mix_weight ∈ [0, 1] on 10 K held-out tokens.**
5. **Inference (`predict`):**
   - q = features(window).
   - logits_kNN = softmax-mix of v_i for top-32 nearest k_i in index.
   - logits_ridge = W · q.
   - logits = mix · logits_kNN + (1 - mix) · logits_ridge.
   - softmax → 256-d distribution.
6. **Sweep:** M_max ∈ {256K, 1M, 5M}, top-K ∈ {16, 32, 128}, d ∈ {256, 512,
   1024}, ctx_len ∈ {64, 128, 256}.

## Memory-movement analysis

Train: 5 M tokens × 2 × d_window² flops + accumulator updates ≈ 5 × 10^11
flops total. < 5 s pure compute. FAISS index build: ~30 s.

Inference: per byte, 1 × (1, d) × (d, 256) ridge eval = 250 K flops, +
FAISS lookup ~ 1 ms. 60 K bytes total: 15 G flops + 60 s FAISS time =
~70 s total. Tight against the 300 s cap; safer to use brute-force kNN on
GPU (matmul + topk) — at M = 1 M, d = 512: (60 K, 512) × (1 M, 512)^T =
3 × 10^13 flops, ~ 100 s. Still feasible.

Energy: estimate 10-20 kJ. Comparable to `hebbian_fw_v2`. The novelty
isn't energy; it's the all-gradient-free training of the entire model.

## References

- Memorizing Transformers, ICLR 2022: <https://arxiv.org/abs/2203.08913>
- kNN-LM, ICLR 2020: <https://arxiv.org/abs/1911.00172>
- Cho & Saul, NIPS 2009 (arc-cosine):
  <https://papers.nips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html>
- Rahimi & Recht, NIPS 2007 (RFF):
  <https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf>
