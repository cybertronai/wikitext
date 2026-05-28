# Spec 03 — Polynomial / TensorSketch random features + closed-form ridge

## 1. Method & mechanism

Same scaffold as spec_02 (closed-form ridge on a fixed feature map) but with a
**polynomial kernel** rather than RBF. The polynomial kernel k(x, y) = (x^T y + c)^p
has the appealing property of *exactly* representing all interactions up to order p of
the input features, which on byte n-gram contexts directly captures multi-character
interactions like "th" → "e" or "ing" → " ".

Two implementation choices:

1. **TensorSketch (Pham & Pagh 2013):** approximate the polynomial kernel by computing
   m-dim CountSketches of x and chaining FFT-based circular convolution to produce a
   m-dim sketch of the degree-p tensor x ⊗ x ⊗ ... ⊗ x (p times). O(p * (d + m log m))
   per sample.

2. **Explicit degree-p monomials with random hashing (Pham/Pagh / Avron 2014):** explicit
   sparse construction of degree-2 features hashed to m dimensions, avoiding the
   exponential explosion of dim^p monomials.

Closed-form ridge maps the m-dim sketch to a 256-class score vector — identical fit
procedure to spec_02.

## 2. Why not a neural network / not backprop

Same lineage as spec_02. The polynomial kernel feature map is a deterministic random
projection followed by a count-sketch / FFT chain. No SGD anywhere in the pipeline;
ridge is closed-form.

A degree-2 polynomial kernel on byte features is *known to be equivalent* to a 2-layer
MLP with quadratic activation (Cho & Saul 2009 arc-cosine kernel, k=1). We are using
this as a *fixed* feature map, not as a network to train — no backprop.

## 3. Universal approximation status

**Proven up to order p:** the polynomial kernel of degree p is a universal kernel on a
compact subset of R^d for p sufficiently large. For finite p, the function class is the
RKHS of all polynomials of degree <= p in the input features. By Stone-Weierstrass,
taking p large enough approximates any continuous function on a compact input set to
any desired accuracy.

For byte-context inputs (256*K-dim one-hot), the relevant question is whether the
expressivity at moderate p (p=3 or 4) captures the n-gram structure of English well
enough. The answer from compression theory is yes — PPM order-4 already reaches ~0.63
char-acc, and a degree-3 polynomial on a one-hot byte window encodes all length-3
character interactions.

## 4. Discrete categorical fit

Identical to spec_02 — output is a 256-vector of real scores. Soft scoring; argmax for
CharModel.predict().

## 5. Autoregressive applicability

Identical to spec_02 — sliding window K, per-char predict.

**Novel application** at byte-level WikiText scale. The most closely related published
result is the FALKON YELP text classification benchmark with 3-grams as input features
(see spec_04). No published per-char polynomial-kernel result on raw text streams.

## 6. Roofline analysis

The advantage of polynomial-kernel sketches over RFF is **higher arithmetic intensity for
the same m**, because the sketch construction does m-dim FFT chains rather than dense
m*d matmuls. For m=4096, p=3:

- Sketch cost per sample: p * (d + m log m) ~= 3 * (4096 + 4096 * 12) = 1.5e5 ops.
- Per-position FLOPs: 1.5e5.
- Total for N=5e6: 7.5e11 FLOPs — sub-second on A100.
- Ridge solve: same as spec_02, 0.5–2 s.

**Higher-degree variant (p=4, m=8192) for capacity:**
- Sketch: 4 * (4096 + 8192 * 13) = 4.4e5 ops per sample.
- Per N=5e6: 2.2e12 FLOPs — still ~5 s.
- Phi @ Phi.T: 2 * 5e6 * 8192^2 ~= 6.7e14 FLOPs ~= 2 s on A100.
- HBM for Phi: 5e6 * 8192 * 2 = 80 GB streamed (~40 s at 2 TB/s). **HBM-bound for the
  Gram accumulation step.** Mitigation: stream in chunks, accumulate Phi.T @ Phi on
  the fly without materializing Phi.

Arithmetic intensity for streamed Gram accumulation: a B x m matrix multiplied by its
transpose, output reused = N * m^2 FLOPs vs N * m bytes read = m FLOPs/byte = 8192 ~
**deeply compute-bound** at p=4, m=8192.

## 7. Top references

1. Pham & Pagh 2013, "Fast and Scalable Polynomial Kernels via Explicit Feature Maps", KDD.
   <https://dl.acm.org/doi/10.1145/2487575.2487591>
   *TensorSketch original.*
2. Avron, Nguyen, Woodruff 2014, "Subspace Embeddings for the Polynomial Kernel", NeurIPS.
   *Improved sample complexity for TensorSketch.*
3. Cho & Saul 2009, "Kernel Methods for Deep Learning", NeurIPS.
   <https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf>
   *Arc-cosine kernel; clean exposition of polynomial-kernel ↔ MLP-with-fixed-activation equivalence.*
4. Wacker, Aydore, Honorio, Yang 2023, "Complex-to-Real Sketches for Tensor Products with
   Applications to the Polynomial Kernel", AISTATS. <https://arxiv.org/abs/2202.02031>
   *Modern improved tensor sketch; lower variance.*
5. Franchi, Maravelle, Bidaut 2025 (preprint), "Tensor Sketch: Fast and Scalable Polynomial
   Kernel Approximation". <https://arxiv.org/abs/2505.08146>
   *Recent revisit; relevant timing benchmarks on GPU.*

## 8. Limitations / failure modes

- **Degree-vs-capacity tradeoff.** p=2 may be too weak; p=4 may overflow the m budget.
  A budgeted sweep within 300 s should cover p in {2, 3, 4} at m in {4096, 8192}.
- **Choice of input encoding** dominates. One-hot byte * window is the obvious starting
  point but may not produce a useful interaction structure under polynomial expansion.
  Position-aware features (one-hot per (position, byte)) are higher-d but encode order.
- **The polynomial kernel is not shift-invariant on positional one-hots** — degree-3
  monomials at positions (1,2,3) are different features from (2,3,4). This is a feature,
  not a bug, for language: byte-position is informative.
- **Conditioning of Gram matrix** at high m/p can be worse than RFF — needs aggressive
  jitter.
- **No known stochasticity-filter risk** — soft scores.

## 9. Experiment spec

**Setup.**
- Context window: K=16.
- Input encoding: one-hot per (position, byte) = 16*256 = 4096-dim sparse-binary input.
- TensorSketch m=8192, degree p=3 (sweep p in {2,3} if time permits).
- bf16 sketch construction (FFT in fp32 to control numeric error).
- Ridge solve in fp32.

**Training procedure.**
1. Stream N=5e6 positions, build sketch phi_p(x_t) using FFT-based TensorSketch.
   Implementation: torch.fft + 1-d CountSketch hash tables sampled once at init.
2. Accumulate Phi.T @ Phi (m x m) and Phi.T @ Y (m x 256) in streaming bf16 chunks.
3. Single Cholesky in fp32.

**CharModel translation.** Identical to spec_02; predict() = phi @ W (m -> 256), one
matmul per char.

**Energy budget.** ~30–90 s total wall, ~5–15 kJ.

**Char-acc ceiling estimate.** Conjectured 0.55–0.70. The polynomial kernel directly
captures n-gram structure that PPM exploits via its trie; it should at minimum match
PPM order-3 (~0.55). Clearing 0.70 depends on whether degree-3 interactions are enough.

## 10. Verdict — **Tier A**

Direct claim-verification of the polynomial-kernel UAT story at byte-LM scale (an open
question in the literature). Highest arithmetic intensity of any spec in this portfolio.
Cheaper than the modded_nanogpt baseline by 10–100x if it clears 0.70. Should be run
right after RFF (spec_02) as a direct comparison.
