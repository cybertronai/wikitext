# Experiment 02: Structured Memory Bank — k-means Centroids vs. Uniform Sampling

## Hypothesis
A K_mem populated by k-means centroids over the encoded-train distribution (instead of M uniform-random samples) yields a denser, less-redundant memory and outperforms uniform sampling at fixed M. Quantitatively: k-means K_mem at M=4096 ≥ uniform K_mem at M=16384.

## Motivation
`hopfield_layer` (exp 11) sampled K_mem uniformly from train contexts. Train text is highly redundant — many of those 4096 patterns are near-duplicates (the same article header, common phrases) and waste capacity. The dense-associative-memory literature (Krotov 2016 onward) treats stored patterns as orthogonal "prototypes." K-means with M centroids approximates that. Builds on the only winning direction in the prior portfolio.

## Method
Same architecture as `hopfield_layer`. Change only the K_mem/V_mem construction:
1. Sample 100k contexts and encode each through (embed + first 2 blocks, random init) → 100k × d points.
2. Run k-means (k = M = 4096) on those 100k points to get centroid keys K_mem.
3. For each centroid, assign V_mem by mean-pool of the V_mem of contexts in its cluster (or by picking the closest training point's next-byte embed — A/B).
4. Train rest of model as in exp 11.

## Memory-Movement Analysis
- K-means is one-shot (init) cost: 100k × M × d × n_iter FLOPs ≈ 10 GFLOP × 10 iter = 100 GFLOP, ~50 ms on A100.
- Run-time Hopfield layer cost unchanged from exp 11 (same M).
- Net training-energy delta = init-cost delta = sub-1% increase, vs. potentially large accuracy lift from less-redundant memory.

## Setup
- Dataset, model, optimizer identical to `hopfield_layer`.
- Two configurations: `hopfield_kmeans_M4k` (M=4096 centroids) and `hopfield_kmeans_V_mode_A` vs `_B` for the V_mem assignment rule.
- Hand-rolled k-means (no sklearn): repeated `argmin` of (point − centroid) L2 + scatter-mean. Pure torch, ~50 lines.
- Baseline: `hopfield_layer` (uniform K_mem, M=4096).

## Procedure
1. `cp -r submissions/hopfield_layer submissions/hopfield_kmeans_M4k`
2. Add `_init_hopfield_kmeans_memory()` that replaces `_init_hopfield_memory()`:
```python
def _kmeans(X, k, n_iter=10):  # X: (N, d) bf16, k: int
    idx = torch.randperm(X.shape[0])[:k]
    C = X[idx].clone().float()
    for _ in range(n_iter):
        # assign
        d2 = ((X[:, None] - C[None]) ** 2).sum(-1)  # CHUNK if N*k too large
        a = d2.argmin(-1)
        # update
        C.zero_().index_add_(0, a, X.float())
        cnt = torch.zeros(k, device=X.device).index_add_(0, a, torch.ones_like(a, dtype=torch.float))
        C /= cnt.clamp_min(1)[:, None]
    return C, a
```
   Chunk the assignment over N to bound memory.
3. After encoding 100k contexts, run k-means → centroid keys; assign V_mem via mode A (closest-train-point next-byte embed) and mode B (mean-pool of cluster V_mem).
4. Run both. Pick the A/B winner for the leaderboard.

## Success Criteria
- **Strong**: kmeans-M4k val acc > uniform-M4k by ≥ 0.005 absolute → centroids are a better memory.
- **Pass**: kmeans-M4k matches uniform-M16k from exp 01 at lower init cost.
- **Refutation**: kmeans-M4k acc ≈ uniform-M4k → memory redundancy is not the bottleneck.

## Failure Modes & Diagnostics
- Empty clusters: a few centroids may collect no points. Log cluster sizes; reseed empty clusters to the farthest-point heuristic.
- Encoder is random-init → k-means clusters reflect random projection geometry, not semantics. Mitigation: run encoder for 500 SGD steps before k-means.
- N·k memory blow-up: chunk N over 4096-sized batches.

## Estimated Cost
2 Modal runs (mode A, mode B) × ~10 min ≈ $0.85.

## References
- Same as exp 01.
- Lloyd 1957 k-means; not heavy machinery.
