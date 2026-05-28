# Experiment 01: Closed-Form Kernel Ridge Regression over Byte n-gram Features

## Hypothesis
A closed-form kernel ridge regression trained on hashed byte n-gram features (n ∈ {1..6}) **cannot** clear the 0.70 val char-acc gate, but can clear a non-trivial floor (unigram ≈ 0.19, bigram ≈ 0.30+) while using <1 minute and <1 kJ. This calibrates the kernel-LM family's *floor* against neural baselines and validates the experimental rig before more expensive kernel runs.

## Motivation
This is a **paradigm-A "kernel machine replaces the model"** experiment (per `finding_kernel_two_paradigms.md`) and a **claim-verification** check: kernel methods historically lose to deep nets on text *for representation-learning reasons*, but the floor — what they get for free — has not been measured on this specific benchmark. If KRR can clear ~0.50 acc at 0 kJ, that's an interesting reference point for hybrid designs. If it cannot break the bigram floor, that bounds the family from below.

Cross-references: `project_nbb_diagnostic.md` (bigram baseline at 0.30 is the floor any byte-context method must beat), `finding_krr_gradfree.md` (closed-form solve = gradient-free, fits the survey angle).

## Method
Closed-form KRR with explicit (non-implicit) feature map. The "kernel" is implicit in the n-gram count features under a polynomial/linear inner product.

Algorithm:
1. Slide a window of width W = 16 bytes over `wiki.train.raw`. For each window of bytes `b[i-W:i]`, emit a hashed n-gram feature vector:
   ```
   φ(context) ∈ R^F     # F = 2^16 = 65536, hashing trick
   for n in 1..6:
       for j in W-n+1..W-1:
           ng = b[j:j+n]
           h = murmur3(ng) % F
           φ[h] += weight(n)        # weight(n) = 1/n typical
   ```
2. Target: one-hot of next byte y_i ∈ R^256.
3. Subsample N = 200K (window, next-byte) pairs uniformly from train.
4. Solve (Φᵀ Φ + λI)W = Φᵀ Y for W ∈ R^(F × 256) using `torch.linalg.solve` on GPU in fp32. Φ ∈ R^(N × F) is sparse (~6W non-zeros per row, total ~600K nz × 200K rows). Use sparse matmul.
5. predict(): hash the current context window, dot with W (one sparse vec × dense matrix → 256-d row vector), softmax, return dict.

Bandwidth/feature analysis:
- This is *linear* in feature space, so the implicit kernel is k(c, c') = φ(c)ᵀφ(c') (a count kernel, related to the spectrum kernel but unweighted).
- Closed-form ridge solve, no SGD.

## Memory-Movement Analysis
- Feature construction: 200K rows × 6×W = 12K nz per row average → ~2.4B token-touches, mostly L1-resident hashing
- Φᵀ Φ: F × F = 65536² = 4.3G entries dense = **17 GB fp32** → too big. Either (i) drop F to 8192 and accept hash collisions, or (ii) compute ΦᵀΦ directly without materializing Φ.
- Solve: 8192³ Cholesky ≈ 5.5×10¹¹ FLOPs, on A100 BF16 ≈ 1.7 s.
- Inference: per byte, one sparse-vec × dense-matrix mult, 8192 × 256 = 2M FLOPs, dominated by HBM read of W (8 MB). Trivially fast.
- **Total HBM traffic for training: ~1 GB read of Φ + 256 MB write of W + temporaries → well under 10 GB total, vs. modded-nanogpt's >100 GB per forward pass × 2150 steps.**

## Setup
- Dataset: `/data/wiki.train.raw` (Modal-baked path); eval against `/data/wiki.valid.raw` first 60K chars
- Tokenization: byte-level (256-class output), as required
- Model scale: F = 8192 features, W ∈ R^(8192 × 256) → 2.1M params; closed-form, no optimizer state
- Hardware budget: 1 × A100-80GB PCIe, target wall <60 s
- Baseline: modded_nanogpt (51,704 J, 0.7374 acc); bigram floor (~0.30 acc); LWTA-k=4 (46,222 J, 0.7238 acc)
- Metric: val char-acc on first 60K val chars, NVML energy in joules

## Procedure
1. Create `submissions/krr_ngram/submission.py`. Copy the `CharModel` ABC import; do not import torch.nn (closed-form, no gradients).
2. In `train(train_text, valid_text=None)`:
   a. Subsample 200K context-byte pairs (`torch.randint(0, len(train_text) - W - 1, (200_000,))`).
   b. Build sparse Φ as `torch.sparse_coo_tensor`. Use murmur3 (or fast Python `hash()` truncated; verify zero-collision-rate on a 1000-key check).
   c. Compute `A = Φ.T @ Φ + λ I` (F × F dense) using sparse-dense matmul.
   d. Compute `b = Φ.T @ Y` (F × 256 dense). Y is sparse one-hot.
   e. Solve `torch.linalg.solve(A, b)` → W ∈ R^(F × 256).
   f. Stash W and the hashing function on `self`.
3. In `predict(self)`: hash the current context window, sparse-dot with W, softmax with temperature τ = 1.0, return dict.
4. In `observe(self, char)`: append byte to a rolling W-byte deque.
5. Run `python submit.py submissions/krr_ngram --yes`.
6. Energy and val char-acc go into `submissions/krr_ngram/result.json`.

## Success Criteria
- **Passes:** val char-acc ≥ 0.50, energy < 5 kJ (closed-form solve will use <50 J of GPU; mostly idle).
- **Interesting failure:** val char-acc in [0.30, 0.50] — kernel ridge over n-gram features beats trivial baselines but cannot match neural representation learning. Confirms paradigm-A limit and bounds the family.
- **Boring failure:** val char-acc < 0.30 — implementation bug; n-gram features are at least as expressive as bigram smoothing, which clears 0.30.

## Failure Modes & Diagnostics
- **Hash collisions silently destroy signal:** log effective rank of Φ; if << F, drop W to fit.
- **λ poorly tuned:** sweep λ ∈ {1e-2, 1e0, 1e2} *inside* the train() call — cheap.
- **Memory OOM on ΦᵀΦ:** fall back to streaming (compute ΦᵀΦ block-wise over chunks of N=10K rows).
- **predict() too slow:** log per-byte microseconds; if >100 µs, the eval loop will time out separately (eval is not gated but still gates the run completing in finite time).

## Estimated Cost
- 1 Modal A100 run, ~5 minutes wall (image cold-start + eval, since training is <60s)
- ~$0.20 / 0.3 kJ measured energy (mostly idle subtraction territory — most of the 300 s is unused)

## References
- Hofmann/Schölkopf/Smola 2008 "Kernel Methods in Machine Learning" — KRR canonical
- Rahimi/Recht 2007 "Random Features for Large-Scale Kernel Machines" — feature-map ridge
- Lodhi et al. 2002 "Text Classification using String Kernels" — n-gram count kernels for text
- existing submissions: `/home/seneca/wikitext/submissions/modded_nanogpt`, `/home/seneca/wikitext/research/catalog/new_directions/ppm_c`
