# Experiment 04: Linear Transformer with elu(x)+1 Feature Map (Katharopoulos)

## Hypothesis
Replacing softmax attention in modded-nanogpt with the **simplest** linear-attention feature map φ(x) = elu(x) + 1 (Katharopoulos 2020) preserves the 0.70 gate at lower energy than the Performer/FAVOR+ variant (exp 03), because the feature map is positive-by-construction, deterministic (no random ω), and arithmetically far cheaper than `exp(ωᵀx)`. A useful comparison point for whether the random-feature *kernel approximation* per se is worth its complexity over a *deterministic* positive feature map.

## Motivation
The Katharopoulos 2020 paper is the canonical paradigm-B simplification: explicitly drops the "softmax kernel approximation" framing and just uses elu+1 as a positive feature map, recovering all the same O(N) and recurrent-form properties. This experiment tests whether the kernel *theory* matters or whether just having *any* positive feature map + the linear-attention algebra is what wins. A direct A/B against exp 03 isolates "fidelity to softmax" from "linear-cost attention scaffold."

This is also a **cross-pollination** opportunity: the FWP / fast-weights stub on the user's existing shortlist (`reference_method_shortlist.md`) is mathematically equivalent to linear-attention under the right feature map (Schlag/Irie/Schmidhuber 2021). So this experiment is also a baseline for FWP-delta on the same task.

## Method
Same code shape as exp 03, but the feature map and update are:
```
φ(x) = elu(x) + 1                            # cheap, positive, deterministic
S_t = S_{t-1} + φ(k_t) v_tᵀ                  # m×d state, m = d (no projection)
z_t = z_{t-1} + φ(k_t)                       # m-d normalizer
out_t = φ(q_t)ᵀ S_t / φ(q_t)ᵀ z_t
```
No random ω, no orthogonal QR step. Output dimension of the feature map equals head_dim (m = D = 64 here).

## Memory-Movement Analysis
- φ(x) is one elementwise op — essentially free vs. the multi-op `exp(ωᵀx - ‖x‖²/2)` in FAVOR+
- Same cumsum-based parallel-training algorithm as exp 03; FLOPs identical
- State per head: D × D = 64 × 64 = 4 KB per layer, ~25 KB total — *register-resident* on A100
- **Arithmetic intensity:** the linear-attention path is bandwidth-bound by reading q,k,v, not by attention math. Same intensity as exp 03 but with lower constant factors.

## Setup
- Dataset: `/data/wiki.train.raw`, `/data/wiki.valid.raw`
- Tokenization: byte-level (256)
- Model: 6 layers, 384 d, head_dim 64, seq 1024, batch 32 (matches modded_nanogpt)
- Optimizer: same as modded_nanogpt (AdamW + Muon)
- Hardware: 1 × A100-80GB, 300 s
- Baseline: modded_nanogpt 51.7 kJ / 0.7374; exp 03 Performer FAVOR+ (run first if possible for direct A/B)
- Metric: val char-acc, NVML energy

## Procedure
1. Copy `submissions/modded_nanogpt/submission.py` → `submissions/linear_tx_elu/submission.py`
2. Replace `CausalSelfAttention.forward` to use:
   ```python
   q_phi = F.elu(q) + 1
   k_phi = F.elu(k) + 1
   # Causal: cumsum-based linear attention (same algebra as exp 03 but no ω)
   kv = torch.einsum("bhtm,bhtd->bhtmd", k_phi, v)
   S = torch.cumsum(kv, dim=2)
   z = torch.cumsum(k_phi, dim=2)
   num = torch.einsum("bhtm,bhtmd->bhtd", q_phi, S)
   den = torch.einsum("bhtm,bhtm->bht", q_phi, z).clamp(min=1e-6).unsqueeze(-1)
   y = num / den
   ```
3. Keep everything else identical to modded_nanogpt — same Muon optimizer, same training loop, same Newton-Schulz, same Rotary positional encoding.
4. Streaming eval: implement the recurrent form for predict(). Maintain per-head state S (m×d) and z (m); each predict() does one matmul of size (m, d) — *O(1) per byte regardless of context length*. **This is a major eval-energy win.**
5. Submit: `python submit.py submissions/linear_tx_elu --yes`.

## Success Criteria
- **Strong pass:** val ≥ 0.70 AND energy < 45 kJ → wins on energy + capability. Compare to LWTA-k=4 46.2 kJ result; if cleaner mechanism + similar score, it's a leaderboard candidate.
- **Pass:** val ≥ 0.70 AND energy in [45, 52] kJ → comparable to baseline; interesting because mechanism is qualitatively different (constant-state attention).
- **Capability demo:** val in [0.60, 0.70] → linear attention can't quite close the gap; matches some literature findings on char-level.
- **Refuted:** val < 0.60 → linear attention with this feature map breaks at small scale; recommend exp 05 (elu+1 + delta rule).

## Failure Modes & Diagnostics
- **Loss diverges early:** without softmax's normalization, the linear-attn quotient can be unstable. Add a `RMSNorm` on q,k before the feature map.
- **Slow convergence:** linear attention has weaker inductive bias at low data scale. Compensate by increasing learning rate on q,k,v,proj weights (Muon makes this safer).
- **Numerator-denominator cancellation:** when q has a near-orthogonal direction to all of k, both num and den go to 0. The clamp(1e-6) handles this; log fraction of clamped positions to monitor.
- **Eval throughput drop:** despite O(1) per byte, the per-head matmul (m·d = 64·64 = 4K ops) has bad HBM intensity. Fuse the predict() inner loop into a single kernel via `torch.compile`.

## Estimated Cost
- 1 Modal A100 run, ~7 min wall, expected energy 25-50 kJ
- ~$0.30

## References
- Katharopoulos et al. 2020 "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (ICML)
- Schlag, Irie, Schmidhuber 2021 "Linear Transformers Are Secretly Fast Weight Programmers" (ICML) — bridges to FWP shortlist
- modded_nanogpt baseline: `/home/seneca/wikitext/submissions/modded_nanogpt/submission.py`
