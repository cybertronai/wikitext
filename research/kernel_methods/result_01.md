# Result 01: Closed-Form Kernel Ridge Regression over Byte n-gram Features

## Hypothesis (recap)
KRR with hashed byte n-gram features (n in 1..6, W=16) cannot clear the
0.70 val char-acc gate but clears the bigram floor (~0.30) at near-zero
energy — a *floor* measurement for the kernel-LM family.

## Final numbers
| metric                  | value                          |
|-------------------------|--------------------------------|
| `val_char_accuracy`     | **0.3634**                     |
| `training_energy_J`     | **99.8 J**                     |
| `training_duration_s`   | **3.1 s**                      |
| `gpu_name`              | NVIDIA A100-SXM4-80GB          |
| `disqualified`          | **True** (val < 0.70 floor)    |
| best `lambda`           | 1e-2 (sweep over {1e-2, 1, 1e2}, heldout-tied with 1.0 within 0.001) |

Sweep heldout-acc (10K subsample):

| lambda | heldout-acc | solve time |
|--------|-------------|------------|
| 1e-2   | 0.3663      | 0.49 s     |
| 1.0    | 0.3650      | 0.06 s     |
| 1e2    | 0.3177      | 0.06 s     |

## Success-criterion bracket
**Interesting failure**, per the spec's bracket:

- Passes  (>= 0.50 acc):     no
- Interesting failure (0.30 - 0.50): **yes — 0.3634**
- Boring failure (< 0.30):    no

KRR beats the bigram floor (~0.30) by a healthy margin, confirming that
hashed n-gram count features carry real signal up to n=6, but it falls
*far* short of neural baselines (modded-nanogpt 0.7374, LWTA-k4 0.7238).

## Interpretation
This is the **paradigm-A "kernel machine replaces the model" floor on
this benchmark**: ~0.36 char-acc at <100 J and <3 s of GPU work, i.e.
**~500x less energy than modded-nanogpt for ~half its accuracy**. The
gap between this number and the 0.70 gate is the *representation-learning
gap* — n-gram count kernels cannot synthesize long-range / hierarchical
features that the attention stack discovers. The lambda sweep showed a
flat optimum at lambda <= 1.0 with sharp degradation at 1e2, consistent
with the Cholesky being well-conditioned for F=8192 over N=190K (effective
rank close to F).

Implication for the family: any paradigm-A kernel-LM submission that
relies only on hand-designed byte features will live in this 0.30 - 0.40
band regardless of solver. To clear 0.70 the kernel must either (i) take
a learned feature map (paradigm B — hybrid linear head on a deep
encoder) or (ii) exploit an implicit infinite-feature map (NTK / arccos /
RFF over a wide MLP) coupled to learned context embeddings.

## Implementation deviations from the spec
- **Hash function:** spec suggested murmur3 or `hash()`; we used a
  vectorized splitmix64 in PyTorch instead — fully GPU-resident and
  avalanche-quality, no Python-loop hash. Sanity-checked at 943/1000
  distinct buckets-of-8192 on sequential keys (birthday-paradox
  baseline).
- **Sign trick:** added a `+/-1` sign drawn from an independent hash bit
  (standard feature-hashing trick) to keep collisions unbiased — not
  explicitly in the spec but a free correctness improvement.
- **Sparse-sparse matmul:** spec mentions "use sparse matmul" but
  doesn't prescribe how to compute Phi^T Phi without materializing
  dense Phi. We used `torch.sparse.mm(phi.T, phi)` then `.to_dense()` —
  256 MB peak for the F x F result, well under HBM. PtY is built by
  scatter-add over Phi's COO entries (no dense Phi anywhere).
- **N-gram packing key:** include the n-value in the top 8 bits of the
  packed int64 key so 1-gram `b'A'` and a 2-gram starting with `b'A\0'`
  do not collide in the hash input space.

All other knobs match the spec: F=8192, N=200K, W=16, n=1..6, weight=1/n,
lambda sweep {1e-2, 1, 1e2}.

## Files
- submission: `/home/seneca/wikitext/submissions/krr_ngram/submission.py`
- result JSON: `/home/seneca/wikitext/submissions/krr_ngram/result.json`
- run log: `/home/seneca/wikitext/submissions/krr_ngram/run.log`
- nvml evidence: `/home/seneca/wikitext/submissions/krr_ngram/nvml.json`

## Review (post-hoc audit)

**Validity for discarding KRR over hashed byte n-grams:** *Insufficient.*

**Core limitations:**
- **Budget under-saturation by ~100×.** Wall 3.1 s of a 300 s cap; energy 100 J of an implicit ~50 kJ ceiling. `N_SAMPLES = 200_000` and `F_FEATS = 8192` are hard-coded constants, not budget-derived. A fair test would scale N to several million (the sibling `rff_ridge_v1` used 8 M positions on the same hardware) and F to ~32 K (Cholesky still fits HBM).
- **Hashing-trick variance dominates signal.** At ~16 M nnz hashed into 8 192 cols, each col absorbs ~2 K collisions; the ±1 sign trick keeps the mean unbiased but variance scales like √collisions / signal_count, which is large for rare n-grams (the ones carrying actual context information).
- **λ grid is too coarse and skewed.** {1e-2, 1, 1e2} — heldout-acc was nearly flat at the bottom of the range; the optimum may be at λ < 1e-2 or even λ = 0. Cheap to extend.

**Verdict:** The 0.36 number is best read as "what hashed-n-gram-KRR at F=8 192, N=200 K gives", *not* as "what the method's ceiling is on this benchmark". The interesting-failure bracket is a misclassification — the result is uninformative until the method is actually budget-saturated.
