# Experiment 06: Nyström-Based Attention Approximation (Nyströmformer)

## Hypothesis
Nyströmformer (Xiong et al. 2021) approximates softmax attention via Nyström landmarks at O(N) and reportedly achieves 22.7× memory reduction at N=8192. At our short context (T=1024), the constant factors may dominate — testing whether Nyström is useful at this scale, or only kicks in at long context.

## Motivation
Nyström methods (low-rank kernel approximation via landmark sub-sampling) are the *non-random-feature* answer to scaling kernels. Nyströmformer is the cleanest 2021 application to attention; choosing landmarks adaptively (rather than randomly) can outperform random-feature Performer at small m. Worth a direct comparison vs. exp 03 (Performer) to see which approximation family is friendlier to char-LM.

This is also a hedge: if Performer (exp 03) fails on numerical-instability grounds (the `exp()` overflow problem), Nyström sidesteps it entirely — Nyström uses the *true* softmax on a small subset.

## Method
Replace `scaled_dot_product_attention` with the Nyströmformer approximation:
```
A ≈ softmax(Q Kₗᵀ / √d) · pinv(softmax(Qₗ Kₗᵀ / √d)) · softmax(Qₗ Kᵀ / √d) · V
```
where Q_l, K_l are m = 64 landmarks chosen as segment-means of Q, K along the sequence axis (T → m via avg-pool with stride T/m). pinv() is implemented via iterative Newton's method (matches the modded-nanogpt Newton-Schulz aesthetic).

Causal masking: Nyströmformer naturally supports a triangular sparsity pattern by restricting Q_l to landmarks at positions ≤ t. Use the original paper's causal-mask correction.

## Memory-Movement Analysis
- Three softmax matmuls of size (T, m), (m, m), (m, T) instead of one (T, T) softmax
- For T=1024, m=64: 3 · 1024·64 + 64·64 ≈ 200K ops vs. 1024² = 1M ops baseline → **5× FLOP reduction**
- Memory: never materializes T×T = 1M-entry attention matrix; instead three small matrices summing to ~130K entries
- The iterative pinv() (~6 Newton steps on 64×64) is negligible
- **Arithmetic intensity:** dominated by the (T, m) softmax which is bandwidth-bound; ~3-4× HBM win vs. full attention

## Setup
- Dataset: `/data/wiki.train.raw`, `/data/wiki.valid.raw`
- Tokenization: byte (256)
- Model: 6 layers, 384 d, head_dim 64, seq 1024 (could push to 2048 to give Nyström a bigger win), batch 32
- Optimizer: AdamW + Muon (modded_nanogpt baseline)
- Hardware: 1 × A100-80GB, 300 s
- Baseline: modded_nanogpt 51.7 kJ; exp 03 Performer (same paradigm-B competitor)
- Metric: val char-acc, NVML energy

## Procedure
1. Copy `submissions/modded_nanogpt/submission.py` → `submissions/nystromformer/submission.py`
2. Replace attention forward with Nyström:
   ```python
   def nystrom_attn(q, k, v, num_landmarks=64):
       B, H, T, D = q.shape
       m = num_landmarks
       seg = T // m
       q_l = q.reshape(B, H, m, seg, D).mean(dim=3)  # landmarks
       k_l = k.reshape(B, H, m, seg, D).mean(dim=3)
       a1 = F.softmax(q @ k_l.transpose(-1, -2) / D**0.5, dim=-1)  # (T, m)
       a2 = F.softmax(q_l @ k_l.transpose(-1, -2) / D**0.5, dim=-1)  # (m, m)
       a3 = F.softmax(q_l @ k.transpose(-1, -2) / D**0.5, dim=-1)  # (m, T)
       inv_a2 = iterative_pinv(a2, n_iter=6)  # Newton iteration
       out = a1 @ inv_a2 @ a3 @ v  # (B, H, T, D)
       return out
   ```
3. For causal masking: apply per-position landmark restriction (only landmarks at positions ≤ t enter a1[t, :]; Xiong et al. §3.3). At T=1024 with m=64 this means each row has a different "allowed landmarks" mask — implement as a block-causal mask of shape (T, m).
4. Increase n_steps to consume the FLOP savings (target same wall-clock, ~2× more steps than baseline).
5. Submit.

## Success Criteria
- **Pass + energy win:** val ≥ 0.70 AND energy < 45 kJ
- **Pass:** val ≥ 0.70 AND energy ∈ [45, 55] kJ → Nyström claim transfers
- **Refuted at this scale:** val < 0.70 → Nyström's claimed scaling benefit doesn't materialize at T=1024
- **Direct A/B vs Performer (exp 03):** if Nyström clears 0.70 and Performer doesn't (or vice-versa), that's an informative dichotomy on approximation family choice

## Failure Modes & Diagnostics
- **Causal-mask correction wrong:** loss should decrease smoothly; if it plateaus near unigram floor, recheck the per-position landmark restriction.
- **iterative_pinv diverges:** Newton's method on near-singular A2 oscillates. Add a small λI = 1e-3 ridge to A2 before inversion.
- **Landmarks chosen badly:** the segment-mean choice may collapse for highly periodic byte streams. Try learned landmarks (small MLP from positions → landmark queries) as a follow-up if this fails.

## Estimated Cost
- 1 Modal A100 run, ~10 min wall, expected energy 30-55 kJ
- ~$0.40

## References
- Xiong et al. 2021 "Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention" (arXiv 2102.03902, AAAI 2021)
- HuggingFace impl: https://huggingface.co/docs/transformers/en/model_doc/nystromformer
- modded_nanogpt baseline: `/home/seneca/wikitext/submissions/modded_nanogpt/submission.py`
