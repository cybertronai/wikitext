# Spec 05 — Hyperdimensional Computing (VSA) n-gram LM

## 1. Method & mechanism

Hyperdimensional Computing (Kanerva 2009) / Vector Symbolic Architectures (Plate 1995,
Gayler 2003) operate on D-dimensional (typically D = 1024 to 10000) bipolar or complex
vectors. Three operations: **binding** (elementwise multiplication or circular
convolution) — composes a role with a filler; **bundling** (elementwise sum +
threshold/sign) — superposes multiple bindings into one vector; **permutation** — encodes
position.

For a char-LM:

    # Build memory M by streaming train: for each context (c_{t-K+1}, ..., c_t) -> next byte c_{t+1}
    pos_i = random_hv() for i in range(K)         # K position hypervectors, frozen
    byte_v = random_hv() for v in range(256)       # 256 byte hypervectors, frozen
    M[c_{t+1}] += bind(pos_1, byte_v[c_{t-K+1}]) * bind(pos_2, byte_v[c_{t-K+2}]) * ...
                  # bundled context hypervector accumulated per next-byte class

At inference, build the query hypervector q = bundle(bind(pos_i, byte_v[c_{t-K+i}])) for
the current context, then argmax over v in 0..255 of <q, M[v]>. Output: a hard winner
over 256 bytes — survives the stochasticity filter only if M[v] aggregates "modal
prediction per context" rather than "memorize each pair."

## 2. Why not a neural network / not backprop

Pure additive accumulation into class prototypes — no layers, no gradients, no SGD.
The "binding" and "bundling" operations are deterministic elementwise ops; the
"training" step is a sum of bound hypervectors per class. This is the gold standard
of one-pass, gradient-free learning.

## 3. Universal approximation status

**Empirical.** HDC has no clean UAT theorem in the kernel/MLP sense. Plate 1995
established that HRR vectors form a *compositional* representation system with
bounded similarity distortion. For classification tasks, HDC has been shown to
match k-NN performance on many datasets (Imani et al. 2017, Gallant 2013).

The relevant theoretical hook: a high-dim random projection of categorical features
+ class-prototype averaging is the "Centroid Classifier" — a well-known weak
baseline. Universal approximation is *not* expected; HDC is a fast retrieval-style
classifier, not a function-class-universal learner.

## 4. Discrete categorical fit

256 class prototypes M[0], ..., M[255]. Score = inner product. Argmax for predict().
Hard-WTA over 256 classes — **at risk under the stochasticity filter** for
contexts where multiple next-bytes are roughly equiprobable. Mitigation: emit a
soft distribution via softmax(scores / tau) instead of argmax; the CharModel API
accepts either.

## 5. Autoregressive applicability

Standard sliding-window K. Has been used for ranking-style next-symbol prediction
in HDC literature (Kleyko 2018) at toy alphabet sizes. **Novel application** for
byte-level WikiText.

## 6. Roofline analysis

For D=10000, K=16, V=256:

**Training (one pass over N=5e6 contexts):**
- Per-context: K-1 bindings (K-1 * D ops) + 1 bundle into M[v] = K * D ops.
- Total: N * K * D = 5e6 * 16 * 1e4 = 8e11 ops, single-pass.
- HBM traffic: M (256 * D * 4 bytes = 10 MB) + streaming contexts (5e6 * 16 = 80 MB).
- Arithmetic intensity: 8e11 ops / 1e8 bytes ~= 8000 ops/byte.

The ops here are not Tensor-Core-friendly matmuls; they are elementwise products
into per-class buffers. Effective compute peaks at A100's ~9 TF FP32 elementwise
throughput, not 312 TF Tensor Core throughput. **Compute-bound on the elementwise
unit, not the Tensor Cores.** ~0.1 s wall on A100.

**Inference (per-char):**
- Build query: K * D = 160K ops.
- Score against 256 class prototypes: 256 * D = 2.5M ops.
- Total per char: 2.66M ops. **Memory-bound**: every char reads the 10 MB M into
  cache. 60K chars * 10 MB = 600 GB streamed = ~0.3 s purely on HBM throughput.

Overall **bandwidth-bound** for inference; compute time is negligible.

## 7. Top references

1. Kanerva 2009, "Hyperdimensional Computing: An Introduction to Computing in
   Distributed Representation with High-Dimensional Random Vectors", Cogn. Comput.
   <https://link.springer.com/article/10.1007/s12559-009-9009-8>
   *Modern HDC framing.*
2. Plate 1995, "Holographic Reduced Representations". <https://www.cs.toronto.edu/~plate/papers/PlateIEEE.pdf>
   *HRR original; circular-convolution binding.*
3. Kleyko, Osipov, Wiklund 2018, "A Comparison of Vector Symbolic Architectures",
   Artif. Intell. Rev. <https://arxiv.org/abs/2001.11797>
   *Includes sequence/temporal benchmarks.*
4. Alam, Raff, Holt 2023, "Recasting Self-Attention with Holographic Reduced
   Representations" (ICML). <https://arxiv.org/abs/2305.19534>
   *Hrrformer; HRR for sequence modeling at modern scale; "up to 370x faster to train".*
5. Imani, Salamat, Khaleghi, Samragh, Koushanfar, Rosing 2017, "QuantHD: A Quantization
   Framework for Hyperdimensional Computing". *Compute-efficient HDC benchmarks.*

## 8. Limitations / failure modes

- **Capacity is set by D.** D=10000 is the published high-fidelity setting; at this
  D, the centroid classifier can hold ~D/log(V) distinguishable patterns. For text
  with N=5e6 contexts and effective entropy 5 bits/byte, the prototypes will saturate
  long before convergence. Mitigation: per-class weighting by inverse frequency, or
  a fresh-Hebbian rule (Plate's "cleanup memory" trick).
- **Hard-WTA at output** — same risk as NBB if M[v] is naive sum-of-contexts. The
  cure is to bundle *only the modal next byte's bindings*, i.e. for each unique
  context cluster, accumulate into the modal-byte's M[v]. This is non-trivial
  without a clustering pass.
- **Bandwidth-bound inference** — does not exploit Tensor Cores. 60K char eval
  fits in ~1 s; not a budget blocker.
- **No published byte-level WikiText result for HDC** — this is a capability demo
  with realistic ceiling ~0.40–0.55 char-acc. Unlikely to clear 0.70.

## 9. Experiment spec

**Setup.**
- D=10000, K=16, V=256, bipolar {-1, +1} encoding.
- Permutation-based positional encoding: pos_i = roll(seed_v, i) for a single seed
  hypervector seed_v (Kanerva's "binding by permutation" reduces memory vs K random
  pos_i's). Saves 16x parameter cost.
- Bipolar bind = elementwise multiply. Bundle = sum + sign.

**Training.**
1. Initialize 256 byte hypervectors and 1 seed hypervector (frozen).
2. Stream 5e6 byte positions; for each, build the bound-and-bundled context
   hypervector and add to M[next_byte].
3. After streaming, normalize: M[v] ← sign(M[v]) for bipolar, or M[v] /= ||M[v]||
   for L2-normalized.

**CharModel translation.**
- `predict()`: build query hypervector for current K-byte window; dot product
  against 256 normalized prototypes; argmax (or softmax/tau for soft).
- `observe(c)`: append c to ring buffer.
- `reset()`: clear ring buffer.

**Energy budget.** Training: <2 s wall, <500 J. Inference: ~1 s for 60K chars,
~250 J. **Smallest energy submission in the portfolio if it clears 0.70.**

**Char-acc ceiling estimate.** 0.30–0.50. The HDC centroid classifier is a known
weak baseline; the surprise would be reaching anywhere near 0.70.

## 10. Verdict — **Tier B (capability demo)**

Worth running because the energy is tiny (cost is dominated by NVML idle baseline)
and the mechanism is genuinely different in kind from anything in the leaderboard.
Realistic ceiling 0.40–0.55 means it likely DQs. The value: published number for
the first time on byte-level WikiText. Submit as a capability claim, not a
record-class entry.
