# Experiment 08: Parzen / Nadaraya-Watson Kernel Density Char-LM (Explicit Memory)

## Hypothesis
A non-parametric Parzen-window / Nadaraya-Watson estimator over (context_embedding, next_byte) pairs from train, with a softmax kernel over context similarities, can clear val char-acc 0.50 with *zero gradient steps* (all "training" is just memorizing pairs). Tests the pure paradigm-A explicit-memory bound and is also the conceptual limit of "modern Hopfield = softmax attention" (Ramsauer 2020).

## Motivation
This is the *simplest* kernel-LM: predict next byte as a kernel-weighted average of next bytes from similar contexts in train. No backprop, no optimizer, just `softmax(Q Kᵀ / τ) V` where K, V come from a frozen subsample of training contexts. Ramsauer 2020 "Hopfield Networks Is All You Need" shows this is mathematically the same as a single transformer attention layer where memory comes from data instead of being learned.

Information value:
- If it passes 0.50 → kernel density on raw context is meaningful even without representation learning
- If it passes 0.70 → striking capability demo: char-LM with no training
- If it fails the unigram floor → reveals that raw byte contexts are too high-dimensional / sparse for direct similarity to help, justifying experiments that first learn an embedding (exp 07)

Cross-references: `survey_kernel_methods_2026_05.md` (modern Hopfield = paradigm B but here used as pure paradigm A); `finding_kernel_stochasticity_filter.md` (this method is a kernel density estimator → soft outputs → survives the stochasticity filter).

## Method
"Train" = build memory: sample N = 200K (context_window of W=64 bytes, next_byte) pairs. Encode each context as a fixed embedding (option A: bag-of-n-grams hash; option B: random projection of byte-onehots).

predict():
1. Encode current context → q ∈ R^d (same fixed encoding)
2. Compute similarities `s_i = qᵀ k_i / τ` for all i (or top-k via approximate NN if N>100K)
3. Aggregate: `p(byte = b) = Σ_i softmax(s)_i · 1[next_byte_i = b]`
4. Return p as dict.

Two variants:
- **Hashing-bag-of-n-grams encoding:** d=4096, deterministic. φ(context)[hash(ngram) % d] += 1/n
- **Random-projection encoding:** sample R ∈ R^(W·256 × 1024), φ(context) = R · onehot(context). Test the "random feature" framing on the raw context.

## Memory-Movement Analysis
- Memory store: N × d × 2 bytes (fp16) for K, plus N × 256 bits one-hot for V (or just store byte IDs). N=200K, d=4096 → 1.6 GB for K. Fits in HBM.
- predict(): q × Kᵀ = (1, 4096) × (4096, 200000) = 800M FLOPs per byte → 4 ms per byte at A100 peak. **Too slow for streaming eval** at default scale. Need top-k NN to drop cost.
- Mitigation: use FAISS-IVF or HNSW to retrieve top-K=128 neighbors. Per-query cost: 5K-20K FLOPs → 50 µs → 50 ms per char. Acceptable.
- Total "training" energy: just memory construction; ~100 J. The dominant cost is *eval* — but eval is not charged against the energy budget per task rules.
- HBM traffic for predict: read top-K vectors = 128 × 4096 × 2 = 1 MB per byte read → bandwidth-bound but fast.

## Setup
- Dataset: `/data/wiki.train.raw`, `/data/wiki.valid.raw`
- Tokenization: byte (256)
- Encoding: d=4096 hashed n-gram bag (n ∈ 1..5, W=64)
- Memory: N=200K random (context, next-byte) pairs from train
- Library: `faiss-gpu` for top-K retrieval (verify Modal image; install if needed)
- Hardware: 1 × A100-80GB, 300 s
- Baseline: modded_nanogpt 51.7 kJ / 0.7374; unigram 0.19; bigram ~0.30
- Metric: val char-acc; energy expected near idle baseline

## Procedure
1. Create `submissions/parzen_kde/submission.py`.
2. In `train()`:
   a. Subsample 200K (context_64, next_byte) pairs from train_text.
   b. Compute hashed-n-gram encoding for each context → K ∈ R^(200K, 4096) fp16.
   c. Build a FAISS-GPU IndexFlatIP (inner product) on K. (If FAISS unavailable, use exhaustive matmul with batch=1 — slower but correct.)
   d. Store V = byte IDs as torch.uint8 of shape (200K,).
3. In `predict()`:
   a. Hash current context → q ∈ R^4096.
   b. Top-K=128 search → indices, distances.
   c. Compute softmax weights = softmax(distances / τ).
   d. Aggregate per-byte: `probs = torch.zeros(256); probs.scatter_add_(0, V[indices], weights)`
   e. Return as dict.
4. Test temperature τ ∈ {0.1, 1.0, 10.0} (cheap; can sweep inside one submission).
5. Submit.

## Success Criteria
- **Striking pass:** val ≥ 0.70 with energy < 5 kJ → capability demo: char-LM with no training
- **Strong pass:** val ≥ 0.50 → kernel density meaningful on raw context
- **Floor:** val in [0.30, 0.50] → matches bigram baseline, doesn't beat representation-free baselines materially
- **Refuted:** val < 0.30 → context encoding too sparse; need learned embedding (see exp 07)

## Failure Modes & Diagnostics
- **Temperature too sharp:** all weight goes to one neighbor; effective k=1 NN. Sweep τ.
- **Hash collisions:** monitor effective rank of K; if degenerate, increase d.
- **FAISS not available:** fall back to brute-force `topk(K @ q, 128)` — works but is bandwidth-limited at ~10 ms/byte (60K val bytes → 600s eval → exceeds wall? eval is not gated by 300s but the run still needs to finish in finite time; verify by running first 5K chars locally before submitting).
- **n-gram encoding throws away order:** try a position-aware encoding (e.g. concatenate per-position byte one-hots + project).

## Estimated Cost
- 1 Modal A100 run, ~15 min wall (slower eval due to per-byte retrieval), expected energy 5-15 kJ
- ~$0.60

## References
- Ramsauer et al. 2020 "Hopfield Networks Is All You Need" (arXiv 2008.02217)
- Parzen 1962 "On Estimation of a Probability Density Function and Mode"
- Khandelwal et al. 2020 "Generalization through Memorization: Nearest Neighbor Language Models" (kNN-LM, arXiv 1911.00172) — closest in spirit
- FAISS: https://github.com/facebookresearch/faiss
