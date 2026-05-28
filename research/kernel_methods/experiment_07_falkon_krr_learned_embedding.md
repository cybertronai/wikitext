# Experiment 07: Nyström Kernel Ridge on Learned-Embedding Features (NN backbone + KRR head)

## Hypothesis
A two-stage method — (1) train a small transformer for ~150s to extract context embeddings, then (2) closed-form solve a Nyström kernel ridge regression (KRR) mapping embedding → next-byte distribution — can reach 0.65+ val acc at lower total energy than running modded-nanogpt for 250s. Tests the **representation-learning + kernel-readout** combo (deep kernel learning) on the wikitext task.

## Motivation
The historical case against kernel methods on text is "they can't learn representations." Modern compromise: let a small NN learn the embedding, then put a kernel ridge head on top — captures the universal-approximator property of kernels with the representation-learning power of NNs. **This is paradigm-A for the readout but uses paradigm-B-style embedding learning** — a hybrid worth measuring.

Cross-references: `finding_krr_gradfree.md` (the KRR readout solve is fully gradient-free); FALKON paper (Rudi 2017) demonstrates GPU-scale Nyström KRR matching SGD-trained nets on classification.

The interesting energy comparison: if the NN backbone trains for half the time of modded-nanogpt and the KRR solve costs <1 kJ, the total could be well under baseline even if accuracy is a bit lower.

## Method
Two-phase training:

**Phase 1 (gradient-based, ~120s):** Small transformer encoder, 4L/256d, trained with cross-entropy on next-byte. After phase 1, *discard* the output head — keep only the encoder.

**Phase 2 (closed-form Nyström KRR, <30s):**
1. Sample N = 100K (context, next-byte) pairs from train.
2. Forward-pass through the frozen phase-1 encoder to get embeddings e_i ∈ R^256.
3. Targets Y ∈ {0,1}^(N×256) one-hot, or Y ∈ R^(N×256) smoothed (label-smoothing 0.1).
4. Choose M = 1024 Nyström landmarks via uniform sampling (k-means++ ablation listed under Procedure).
5. Solve KRR with cosine kernel k(a, b) = aᵀb / (‖a‖·‖b‖) — see `finding_rbf_text_isotropy.md` for why cosine over RBF.
6. Output weights α ∈ R^(M × 256).

predict(): encode current context via phase-1 encoder, evaluate kernel against M landmarks (one matmul), weighted sum → 256-d logits, softmax.

## Implementation (hand-rolled Nyström KRR)

The `falkon` package is not in the Modal image and submissions cannot pip-install. The Nyström + ridge solve fits in ~40 lines of PyTorch — no preconditioned CG needed at M=1024 since the M×M solve is sub-second on A100.

```python
import torch

def nystrom_krr_fit(
    E: torch.Tensor,         # (N, d)  encoder embeddings, L2-normalized
    Y: torch.Tensor,         # (N, C)  one-hot or smoothed targets
    M: int = 1024,
    penalty: float = 1e-3,
    landmark_idx: torch.Tensor | None = None,  # (M,) into [0, N)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (landmarks Z ∈ R^{M×d}, alpha ∈ R^{M×C}).

    Closed-form Nyström KRR with cosine kernel (inputs assumed L2-normalized
    so cosine kernel == linear kernel on normalized inputs).
    Solve  (K_MM + λ K_MN K_NM / N)  α  =  K_MN Y / N.
    """
    N, _ = E.shape
    if landmark_idx is None:
        landmark_idx = torch.randperm(N, device=E.device)[:M]
    Z = E[landmark_idx]                                  # (M, d)
    # K_MN  = Z @ E.T   (M, N)
    # K_MM  = Z @ Z.T   (M, M)
    # Build in fp32 for the solve, even if E is bf16
    Z32 = Z.float()
    E32 = E.float()
    K_MM = Z32 @ Z32.t()                                 # (M, M)
    K_MN = Z32 @ E32.t()                                 # (M, N)
    # Normal-equations form, regularized
    A = K_MM + (penalty / N) * (K_MN @ K_MN.t())         # (M, M)  PSD
    A.diagonal().add_(1e-6)                              # numerical jitter
    rhs = (K_MN @ Y.float()) / N                         # (M, C)
    # Cholesky is ~3× faster than torch.linalg.solve at M=1024
    L = torch.linalg.cholesky(A)
    alpha = torch.cholesky_solve(rhs, L)                 # (M, C)
    return Z, alpha


def nystrom_krr_predict(
    e_query: torch.Tensor,   # (B, d)  L2-normalized
    Z: torch.Tensor,         # (M, d)  L2-normalized
    alpha: torch.Tensor,     # (M, C)
) -> torch.Tensor:
    """Return logits (B, C). Caller applies softmax."""
    K_query = e_query.float() @ Z.t().float()            # (B, M)
    return K_query @ alpha                                # (B, C)
```

Notes for the implementer:
- L2-normalize both `E` and the landmarks before the solve. Cosine kernel = linear kernel on normalized inputs, so the matmul form above is exact, not an approximation.
- M=1024 → the (M, M) Cholesky is 1 GFLOP, well under 100 ms on A100.
- Memory: K_MN at N=100K, M=1024 is 400 MB in fp32 — fits, but if you push to N=500K, switch to a chunked accumulation of K_MN @ K_MN.t() to keep peak memory bounded.
- Validate against a (toy) N=64, M=8, C=4 brute-force `(K + λI)⁻¹ Y` reference before plugging into the model.

## Memory-Movement Analysis
- Phase 1 training: same as a small modded-nanogpt, ~40 kJ if run for 120s
- Phase 2 KRR solve: Nyström KRR is O(N·M·d + M³). For N=100K, M=1024, d=256: ~26 GFLOPs encoder forward + 1 GFLOP solve → <2 s on A100, ~0.5 kJ
- predict() cost: encoder forward (~0.1 ms) + (M, d) matmul (~5 µs) = dominated by the encoder, same order as a transformer forward
- **Total energy budget:** phase 1 ~40 kJ + phase 2 ~0.5 kJ ≈ 41 kJ projected, less than 51.7 kJ baseline *if accuracy survives*

## Setup
- Dataset: `/data/wiki.train.raw`, `/data/wiki.valid.raw`
- Tokenization: byte (256)
- Model: phase-1 encoder 4L/256d, head_dim 64, seq 512; phase-2 hand-rolled Nyström KRR with M=1024, cosine kernel
- Libraries: pure PyTorch — no external kernel library required (the Modal image ships torch 2.5.1+cu124 + numpy + pyarrow + tiktoken; nothing else)
- Hardware: 1 × A100-80GB, 300 s (120s for phase 1, <30s for phase 2, rest unused)
- Baseline: modded_nanogpt 51.7 kJ / 0.7374; LWTA-k=4 46.2 kJ
- Metric: val char-acc, NVML energy

## Procedure
1. Create `submissions/nystrom_krr_hybrid/submission.py`.
2. Implement phase 1: small encoder + temporary linear head + AdamW. Train 120s wall.
3. After phase 1: discard output head. Sample 100K context windows from train, encode them. Pool over the last token's residual stream → e_i (256-dim). L2-normalize.
4. Run the hand-rolled `nystrom_krr_fit` (see Implementation block). Store `(Z, alpha)` on the CharModel instance.
5. Implement `predict()`: encode current context with the phase-1 encoder → e_query (L2-normalized) → `nystrom_krr_predict(e_query, Z, alpha)` → softmax over 256 → return dict.
6. Ablations to run within this submission (controlled by env var or constants — pick the best for the recorded run):
   - Landmark choice: uniform random vs. k-means++ (hand-roll k-means++ in ~30 lines — torch only)
   - Targets: one-hot vs. label-smoothed (0.1)
   - Kernel: cosine (default) vs. arc-cosine order 2 (see Failure Modes; matches a one-hidden-layer ReLU net)
7. Submit: `python submit.py submissions/nystrom_krr_hybrid --yes`.

## Success Criteria
- **Strong pass:** val ≥ 0.70 AND total energy < 45 kJ → deep-kernel-learning hybrid beats baseline
- **Pass:** val ≥ 0.70 → demonstrates the hybrid works at all
- **Capability demo:** val in [0.60, 0.70] → bounds the deep-kernel approach; tells us phase-1 encoder needs more capacity or training time (will be reported DQ by the harness's 0.70 floor, but the result is informative)
- **Refuted:** val < 0.60 → the readout step throws away too much; deep kernel learning at this scale doesn't pay off

## Failure Modes & Diagnostics
- **Cholesky fails (matrix not PSD):** raise `1e-6` jitter to `1e-4`; if still failing, switch to `torch.linalg.solve(A, rhs)` (LU-based, slower but more permissive).
- **Phase 1 encoder collapses without an output head being trained jointly:** train phase 1 *with* an output head (standard LM loss), then discard the head. This is what gives meaningful embeddings.
- **Cosine kernel underperforms:** try arc-cosine kernel of order 2 — replace the kernel matmul with `k(a,b) = (1/π) · (sin θ + (π - θ) cos θ)` where `cos θ = a·b/(‖a‖‖b‖)`. Captures one-layer-ReLU-net behavior. Cost: an extra elementwise op; no architectural change.
- **Smoothed Y vs. one-hot Y matters:** try both. Smoothed (e.g. label-smoothing 0.1) often helps kernel ridge on stochastic targets, which is the regime here per `feedback_charmodel_stochastic_targets`.
- **K_MN OOMs at large N:** chunk the construction — accumulate `K_MN @ K_MN.t()` over batches of rows of E, never materialize the full K_MN. Cuts peak memory at the cost of one extra matmul pass.
- **Encoder forward in `predict()` is too slow per byte:** the small encoder must use a KV-cache (same pattern as modded-nanogpt). Without that, autoregressive eval over 120K bytes will blow the 50-min Modal function timeout.

## Estimated Cost
- 1 Modal A100 run, ~10 min wall, expected energy 35-55 kJ
- ~$0.40

## References
- Rudi, Carratino, Rosasco 2017 "FALKON: An Optimal Large Scale Kernel Method" (NeurIPS, arXiv 1705.10958) — algorithm reference; we use the closed-form Cholesky path, not the preconditioned-CG iterations
- Meanti et al. 2020 "Kernel methods through the roof: handling billions of points efficiently" (NeurIPS)
- Cho & Saul 2009 "Kernel Methods for Deep Learning" — arc-cosine kernel definition
- Reference open-source impl (read-only; do not import — image lacks it): https://github.com/FalkonML/falkon
