# Experiment 03: Performer FAVOR+ Drop-In Replacement of Attention in modded-nanogpt

## Hypothesis
Replacing the softmax attention in modded-nanogpt with Performer's FAVOR+ positive random-feature approximation (k=128 features) clears the 0.70 val gate at *lower energy* than the 51.7 kJ baseline, because FAVOR+ has linear complexity in sequence length and reduces HBM traffic during the attention forward pass.

## Motivation
**Paradigm-B kernelization** (per `finding_kernel_two_paradigms.md`). This is the strongest 2020-vintage claim about kernels in LMs: Performer (Choromanski et al. 2020) provably approximates softmax attention as a kernel with random features. Reproducing this claim on the wikitext benchmark is a high-information experiment: either (a) the result transfers to char-LM at 6L/384d (energy win, claim verified) or (b) the random-feature approximation degrades accuracy enough to miss 0.70 (claim refuted at this scale, interesting negative result).

Cross-reference: `survey_kernel_methods_2026_05.md` (Performer is the highest-ranked paradigm-B candidate).

## Method
Replace the `F.scaled_dot_product_attention` call in `submissions/modded_nanogpt/submission.py` (line ~131) with FAVOR+ positive random features:

```python
# Pseudo:
# For each head:
#   q', k' = positive_random_feature_map(q, k)
#       φ(x) = (1/√m) · exp(-‖x‖²/2) · [exp(ω_jᵀ x)]_j=1..m
#   For autoregressive: maintain running sums
#       S_t = Σ_{s≤t} φ(k_s) v_sᵀ      # (m, d_v)
#       Z_t = Σ_{s≤t} φ(k_s)            # (m,)
#       attn_t = (φ(q_t)ᵀ S_t) / (φ(q_t)ᵀ Z_t)
```

For parallel training: use the closed-form linear-attention formula
```
attn = φ(Q) · (φ(K)ᵀ V) / (φ(Q) · φ(K)ᵀ 1)
```
with the causal mask handled by `torch.cumsum`-based scan (O(N·m·d)) — see Katharopoulos 2020 §3.2.

Random features ω are drawn N(0, I) at init (orthogonal for variance reduction per FAVOR+).

## Memory-Movement Analysis
- Baseline attention: O(B·H·T²·d) ≈ 32·6·1024²·64 = 13 G ops per layer per step
- FAVOR+ training: O(B·H·T·m·d) with m=128 → 32·6·1024·128·64 = 1.6 G ops per layer per step → **~8× FLOP reduction** with k=128. Energy win mostly comes from lower HBM traffic: the T×T attention matrix (32·6·1024² = 200M fp16 = 400 MB) is never materialized, replaced by an m×d running state (32·6·128·64 = 1.6M fp16 = 3 MB).
- For autoregressive eval: FAVOR+ has O(1) per-token cost vs. O(T) for SDPA with cache — wins more on eval throughput.

## Setup
- Dataset: `/data/wiki.train.raw`, `/data/wiki.valid.raw`
- Tokenization: byte-level (256)
- Model: same shape as modded_nanogpt — 6 layers, 384 d, head_dim 64, seq 1024, batch 32
- Optimizer: same (AdamW for embed/head/scalars + Muon for 2D)
- Hardware: 1 × A100-80GB, 300 s
- Baseline: **modded_nanogpt 51,704 J / 0.7374 acc** (direct comparison; same hyperparams except attention)
- Metric: val char-acc, NVML energy

## Procedure
1. Copy `submissions/modded_nanogpt/submission.py` → `submissions/performer_favor/submission.py`.
2. Replace `CausalSelfAttention.forward` body. Implement FAVOR+ feature map:
   ```python
   def favor_features(x, omegas):  # x: (B, H, T, D), omegas: (H, D, M)
       # h(x) = exp(-‖x‖²/2)
       # f(x) = (1/√M) · exp(omegasᵀx) · h(x)
       norm = (x ** 2).sum(-1, keepdim=True) / 2
       proj = torch.einsum("bhtd,hdm->bhtm", x, omegas)
       return torch.exp(proj - norm) / math.sqrt(omegas.size(-1))
   ```
3. Causal linear-attention with cumsum:
   ```python
   qf = favor_features(q, omegas_q)              # (B, H, T, M)
   kf = favor_features(k, omegas_k)              # same omegas; FAVOR+ uses identical ω
   # S_t = Σ_{s<=t} kf_s vᵀ_s
   kv = torch.einsum("bhtm,bhtd->bhtmd", kf, v)  # (B, H, T, M, D)
   S = torch.cumsum(kv, dim=2)                   # (B, H, T, M, D)
   z = torch.cumsum(kf, dim=2)                   # (B, H, T, M)
   num = torch.einsum("bhtm,bhtmd->bhtd", qf, S)
   den = torch.einsum("bhtm,bhtm->bht", qf, z).clamp(min=1e-6).unsqueeze(-1)
   out = num / den
   ```
4. Sample ω from `torch.randn(H, D, M, device='cuda') / math.sqrt(D)`. **Optional:** orthogonalize per head via QR for variance reduction (FAVOR+ recommendation).
5. Increase n_steps proportionally to per-step FLOP savings (target same wall-clock; expect 2-3× more steps).
6. Submit: `python submit.py submissions/performer_favor --yes`.

## Success Criteria
- **Verified + energy win:** val char-acc ≥ 0.70 AND energy < 51.7 kJ → Performer claim confirmed at char-LM scale; energy ranking
- **Verified + no energy win:** val ≥ 0.70 but energy ≥ 51.7 kJ → kernel-attention works but per-step savings eaten by needing more steps to converge
- **Refuted:** val < 0.70 → FAVOR+ approximation degrades accuracy too much at this scale (6L/384d may be below the threshold where positive random features track softmax with low variance)
- **Bug:** val < 0.20 → NaN explosion (FAVOR+ is numerically delicate; the exp() can overflow if q,k aren't normalized)

## Failure Modes & Diagnostics
- **Numerical instability:** the `exp()` in the feature map blows up if ‖q‖, ‖k‖ are not bounded. Log max ‖q‖ over training; if >10, divide by √d or RMSNorm q,k harder.
- **Variance too high at m=128:** symptom is loss oscillation. Bump m to 256 or 512 (energy cost increases but still sub-quadratic).
- **The cumsum tensor (B, H, T, M, D)** = 32·6·1024·128·64·2bytes = **3 GB** — fits but tight. Drop M=64 if OOM, or use chunked cumsum.
- **Orthogonal feature draws not seeded reproducibly:** log the omega RNG state and assert it doesn't change across re-runs.

## Estimated Cost
- 1 Modal A100 run, ~10 min wall, expected energy 30-55 kJ
- ~$0.40

## References
- Choromanski et al. 2020 "Rethinking Attention with Performers" (arXiv 2009.14794, ICLR 2021)
- Choromanski et al. 2023 "FAVOR#: Sharp Attention Kernel Approximations" (arXiv 2302.00787)
- modded_nanogpt baseline: `/home/seneca/wikitext/submissions/modded_nanogpt/submission.py`
- Code reference: https://github.com/google-research/google-research/tree/master/performer
