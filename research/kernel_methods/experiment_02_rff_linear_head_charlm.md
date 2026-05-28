# Experiment 02: Random-Fourier-Features over Embedded Context → Linear Head (No Attention)

## Hypothesis
A char-LM whose entire forward pass is `embed(context) → mean-pool / 1D conv → RFF feature map (k=4096) → linear head (256 classes)` can clear val char-acc 0.50 with no attention, and may approach 0.70 at small scale. Tests whether *kernel-induced nonlinearity alone* (no attention, no deep MLP) suffices for the floor a sub-trivial char-LM needs.

## Motivation
Random Fourier Features (Rahimi/Recht 2007) approximate a shift-invariant kernel — typically Gaussian — with k explicit random features, turning kernel evaluation into a single matmul. RFFs sit between paradigm-A (kernel machine) and paradigm-B (kernel as a component): the RFF layer is a fixed (not learned) nonlinearity that is mathematically a kernel, and the only learned parameters are the embedding and the linear head. If this clears the gate, it's a clean *capability demo* — non-trivial char-LM with a kernel feature map and no attention.

The interesting question vs. exp 01 is: does adding a learned input embedding lift KRR's floor materially? Cross-reference: `finding_rbf_text_isotropy.md` (RBF assumes L2 distance is meaningful — we *learn* the embedding, so by training-time the embedding will be approximately isotropic if RFF is downstream).

## Method
Architecture:
```
x: (B, W) byte IDs
e = embed[x]                         # (B, W, d), d=128, learned
c = mean(e, dim=1) or causal_conv(e) # (B, d)
z = RFF(c)                           # (B, k), k=4096, fixed random
                                     # z_j = √(2/k) cos(ω_jᵀ c + b_j)
                                     # ω_j ~ N(0, σ⁻² I), b_j ~ U(0, 2π)
logits = z @ W_out                   # (B, 256), W_out ∈ R^(k × 256)
```
Training: cross-entropy SGD/AdamW on (embed weights, W_out only). RFF parameters (ω, b) are frozen Gaussian + uniform draws. Bandwidth σ is a hyperparameter; default σ = √d using the median-heuristic (compute on a 4K sample at init).

Causal context: in autoregressive setting, c at position t must depend only on bytes <t. Use either (a) mean over last W bytes (translation-invariant context), or (b) causal 1D conv of width W with stride 1. **Default: causal conv** because mean-pool throws away order.

## Memory-Movement Analysis
- Embedding lookup: B × W × d = 32 × 512 × 128 = 2M elements per step, ~8 MB read — trivial
- Causal conv (depth 1, width W=8, channels d): B × T × W × d = 32 × 512 × 8 × 128 ≈ 17M FLOPs/step — trivial
- RFF projection: B × T × d × k = 32 × 512 × 128 × 4096 ≈ 9G FLOPs/step (this is the dominant cost; one big matmul)
- Linear head: B × T × k × 256 = 32 × 512 × 4096 × 256 ≈ 17G FLOPs/step
- Total: ~26 GFLOPs/step × 5000 steps ≈ 130 TFLOPs — fits comfortably in 300 s on A100 (300 TF bf16 peak × 60% util = 54 P available)
- HBM traffic per step: ~150 MB (W_out is the largest tensor at 4 MB). **Arithmetic intensity ≈ 170 FLOPs/byte — high; ~5× better than attention's intensity at this scale.**

## Setup
- Dataset: `/data/wiki.train.raw`, `/data/wiki.valid.raw`
- Tokenization: byte-level (256)
- Model scale: embed 256→128, causal conv width 8, RFF k=4096, head 4096→256 → ~1.1M params
- Hardware: 1 × A100-80GB, 300 s wall
- Baseline: modded_nanogpt 51.7 kJ / 0.7374; LWTA-k=4 46.2 kJ / 0.7238
- Metric: val char-acc, energy joules

## Procedure
1. Create `submissions/rff_linear_head/submission.py`.
2. Define module with embed / causal_conv (1D conv stride 1) / fixed RFF layer / linear head.
3. RFF layer:
   ```python
   class RFF(nn.Module):
       def __init__(self, d, k, sigma):
           super().__init__()
           self.register_buffer("W", torch.randn(d, k) / sigma)
           self.register_buffer("b", torch.rand(k) * 2 * math.pi)
       def forward(self, x):
           return math.sqrt(2.0 / self.k) * torch.cos(x @ self.W + self.b)
   ```
4. Compute median-heuristic σ on init: sample 4K pre-RFF activations, σ = median of pairwise distances (or use σ = √d as a cheap default — log both).
5. Train AdamW (lr 3e-3, wd 0) for as many steps as fit in ~250 s; batch 64, seq 512.
6. CharModel wrapper: maintain a rolling W-byte buffer; predict() = forward pass on the last W bytes; observe() = append.
7. Submit: `python submit.py submissions/rff_linear_head --yes`.

## Success Criteria
- **Strong pass:** val char-acc ≥ 0.70, energy ≤ 51 kJ → competitive capability demo of attention-free kernel-LM (interesting result regardless of energy)
- **Capability demo:** val char-acc in [0.60, 0.70] → kernel feature map alone almost suffices; failure-mode is plausibly representation depth not the kernel
- **Floor hit:** val char-acc in [0.40, 0.60] → bounds RFF + linear from above. Test next: replace RFF with a deeper learned net (which is paradigm B in disguise) to see if RFF is what's gating
- **Bug:** val char-acc < 0.30 → eval impl error, debug

## Failure Modes & Diagnostics
- **σ wrong:** at σ → 0 every RFF feature is locally linear (kernel collapses to delta); at σ → ∞ all features are constant. Log RFF feature variance per dim; should be near 0.5.
- **Bandwidth mismatch with embedding scale:** the embedding can drift far from N(0,1) — log ‖c‖ over training; if it diverges, either RMSNorm c before RFF or retie σ.
- **Position aliasing from mean-pool:** if you used mean instead of conv, "abc" and "bca" have identical pool — log a small ablation with bigram-mean (last 2 chars only) to detect.
- **predict() returns 256 evenly-split probs:** indicates W_out collapsed. Check ‖W_out‖ at the end of train.

## Estimated Cost
- ~5 min Modal wall, expected energy 30-50 kJ depending on training duration
- ~$0.20

## References
- Rahimi & Recht 2007 "Random Features for Large-Scale Kernel Machines" (NeurIPS)
- Choromanski et al. 2020 "Rethinking Attention with Performers" — same RFF math, applied to softmax kernel of QK
- Han & Avron 2021 "Random Features for the Neural Tangent Kernel" (arXiv 2104.01351) — RFF for arc-cosine if you want to swap kernel family
