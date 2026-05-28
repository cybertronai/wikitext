# Experiment 05: DeltaNet — Linear Attention with Delta-Rule Updates

## Hypothesis
Replacing additive linear-attention state updates (S_t = S_{t-1} + φ(k_t)v_tᵀ; exp 04) with delta-rule updates (S_t = S_{t-1}(I - β_t φ(k_t)φ(k_t)ᵀ) + β_t v_t φ(k_t)ᵀ) closes the accuracy gap to softmax attention on char-LM at the modded-nanogpt scale, possibly at lower energy. Tests the strongest 2024 paradigm-B claim (Yang et al. NeurIPS 2024) at small scale.

## Motivation
Yang et al. 2024 (NeurIPS) showed DeltaNet outperforms Mamba and GLA at 1.3B / 100B tokens; the delta rule replaces the additive "kernel state" with a multiplicative *associative-write* that flushes old key-value bindings. This unifies linear-attention with fast-weight programmers (FWP) — the same delta rule used in `fast-weights-rehearsal` on the user's existing shortlist. So this experiment is also a **cross-pollination**: linear-attention algebra (paradigm B kernel-component) × delta-rule fast-weight write (FWP shortlist item from gradfree survey).

This experiment is information-rich because:
- If it beats exp 04 (elu+1 additive), the delta rule earns its complexity
- If it beats softmax attention (modded-nanogpt baseline), it's a leaderboard-grade result
- If it ties exp 04, the 2024 claim doesn't transfer to char-LM scale

## Method
Architecture identical to exp 04 except the linear-attention recurrence is the **chunkwise delta rule of Yang et al. 2024 Algorithm 2**, implemented end-to-end in pure PyTorch (the `flash-linear-attention` library is not in the Modal image and submissions cannot install pip packages — see Implementation below). The per-token update is

```
β_t = sigmoid(W_β x_t)                # learned per-token write strength
                                       # OR fixed β = 1.0 for ablation
S_t = S_{t-1} - β_t S_{t-1} φ(k_t) φ(k_t)ᵀ / ‖φ(k_t)‖² + β_t v_t φ(k_t)ᵀ / ‖φ(k_t)‖²
```

The chunkwise algorithm makes this parallelizable: process tokens in chunks of C=64; within a chunk use Householder products to compose C delta-updates in closed form; across chunks carry the running state S.

Feature map: keep φ = elu+1 (matches exp 04 to isolate the delta-rule effect; an L2-normalized φ(x) = x/‖x‖ ablation is listed under Procedure step 5).

## Implementation (hand-rolled chunkwise delta rule)

Reference: Yang et al. 2024, Algorithm 2. Sketch (one head; vectorize over heads):

```python
# Inputs per layer per forward pass:
#   q, k, v : (B, T, D)   queries, keys, values after φ, RoPE, norms
#   beta    : (B, T)      per-token write gates (sigmoid of W_β x)
# State carried across chunks:
#   S       : (B, D, D)   fast-weight matrix, zero-initialized

def delta_rule_chunkwise(q, k, v, beta, chunk_size=64):
    B, T, D = q.shape
    S = q.new_zeros(B, D, D)
    out = q.new_empty(B, T, D)
    # k_norm so the rank-1 update is well-conditioned
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        q_c = q[:, start:end]            # (B, C, D)
        k_c = k[:, start:end]
        v_c = v[:, start:end]
        b_c = beta[:, start:end].unsqueeze(-1)   # (B, C, 1)
        # Intra-chunk: compose C rank-1 updates exactly via the
        # "T" matrix of Yang et al. §3.2 (lower-triangular system).
        # T[i,j] = -b_j (k_i · k_j) for i>j, plus 1 on the diagonal.
        kkT = k_c @ k_c.transpose(-1, -2)         # (B, C, C)
        mask = torch.tril(torch.ones_like(kkT), diagonal=-1)
        Tm = torch.eye(end - start, device=q.device).expand(B, -1, -1) \
             - b_c.transpose(-1, -2) * kkT * mask
        # Solve Tm @ U = v_c for U  ⇒  U = Tm^{-1} v_c   (forward-substitution)
        U = torch.linalg.solve_triangular(Tm, b_c * v_c, upper=False)
        # Chunk output: q_c @ (S + k_c^T @ U)
        out[:, start:end] = q_c @ S + (q_c @ k_c.transpose(-1, -2)).tril() @ U
        # Inter-chunk: update S with the composed chunk update
        S = S - (S @ k_c.transpose(-1, -2)) @ (b_c.transpose(-1, -2) * kkT).tril() @ U \
              + k_c.transpose(-1, -2) @ U
    return out, S
```

Notes for the implementer:
- `torch.linalg.solve_triangular` exists in torch 2.5 and runs on GPU; the C=64 triangular solve is ~4 K FLOPs per chunk per batch, negligible.
- All ops are bf16-safe except the triangular solve — do it in fp32 (cast in, cast out). Drift in S over long sequences is the main numerical risk; see Failure Modes.
- For predict() / streaming inference at eval time the chunk loop reduces to the elementwise recurrence S_t ← S_{t-1}(I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ; emit y_t = q_tᵀ S_t. This is O(D²) per byte, constant in context length — matches the modded-nanogpt KV-cache pattern in `submissions/modded_nanogpt/submission.py`.

Full hand-roll is ~150–200 LoC including the gated variant. Validate against a brute-force O(T²) reference on a tiny (B=2, T=8, D=4) tensor before plugging into the model.

## Memory-Movement Analysis
- State S_t is (B, H, D, D) per layer. With H=6 heads, head_dim=64 → 6·64·64·4B = 96 KB per batch element per layer; for B=32 → 3 MB per layer, ~18 MB across 6 layers. Fits comfortably in L2.
- Chunkwise update reads q,k,v of chunk (64 tokens × 64 dim ≈ 4 KB per head) into SRAM, does the Householder/triangular composition, writes back.
- Per training step: same FLOP order as exp 04 + ~20% for the triangular solve and the extra matmul against S.
- **HBM-wise this is the same win as linear attention** — never materializes a T×T matrix; trades a few extra (D×D) products per chunk for that.
- Eval inference: O(D²) per token = ~4 K ops, *constant in context length*.

## Setup
- Dataset: `/data/wiki.train.raw`, `/data/wiki.valid.raw`
- Tokenization: byte (256)
- Model: 6 layers, 384 d, head_dim 64, 6 heads, seq 1024, batch 32 (match modded_nanogpt)
- Optimizer: AdamW + Muon (same as modded_nanogpt)
- Hardware: 1 × A100-80GB, 300 s
- Baseline: modded_nanogpt 51.7 kJ / 0.7374; exp 04 elu+1 linear-tx (direct A/B); LWTA-k=4 46.2 kJ
- Metric: val char-acc, NVML energy

## Procedure
1. Copy `submissions/linear_tx_elu/` → `submissions/deltanet/` (assumes exp 04 was run first; otherwise copy `submissions/modded_nanogpt/`).
2. Add a `delta_rule_chunkwise` helper as above and the gate parameter `W_β` (single linear `d_model → 1`).
3. Replace the attention call in the transformer block with `delta_rule_chunkwise(q, k, v, beta)`. Keep RMSNorm, RoPE, ReLU² MLP, soft-capped logits, and the Muon/AdamW split unchanged.
4. Add a brute-force O(T²) reference `delta_rule_naive(q, k, v, beta)` (the explicit per-token recurrence) and a unit test asserting equivalence on a (B=2, T=16, D=4) random tensor. Run the test before submitting.
5. Ablations within the single submission (controlled by env var or constant):
   - `BETA_MODE=learned` (default) vs `BETA_MODE=fixed_one` (β ≡ 1 — pure delta rule, no gate)
   - `FEATURE_MAP=elu_plus_1` (default) vs `FEATURE_MAP=l2_norm` (φ(x)=x/‖x‖)
6. For streaming `predict()`, implement the per-token recurrence as a cache update on S; verify against the chunkwise forward by running the same prefix through both and asserting matching logits.
7. Run `python submit.py submissions/deltanet --yes`.

## Success Criteria
- **Leaderboard candidate:** val ≥ 0.70 AND energy < 45 kJ → DeltaNet at small scale beats modded-nanogpt; high-impact result
- **Capability demo:** val ≥ 0.70 AND energy in [45, 55] kJ → matches baseline; useful as proof of mechanism
- **Refuted:** val < 0.70 → DeltaNet's 2024 claim doesn't extend to char-LM at 6L/384d
- **A/B win vs exp 04:** if val(exp 05) > val(exp 04), the delta rule earns its complexity even if neither hits 0.70

## Failure Modes & Diagnostics
- **Triangular solve numerically unstable in bf16:** cast `Tm` and `v_c` to fp32 around the solve; rest of the network stays bf16. Log `Tm.cond()` periodically — if condition number > 1e4, reduce chunk size to 32.
- **State S drifts to non-PSD / explodes:** log `S.norm()` per layer per chunk. If it blows up, clamp β_t < 0.5 or re-orthogonalize S every 16 chunks via a single QR (cost: a few µs).
- **β_t saturates to 0:** if the learned gate collapses, the delta rule degenerates to "ignore". Fix β = 1 and re-run; if that succeeds, the gate learning was the issue.
- **Chunk size 64 underutilizes SRAM:** try chunk=128 (more memory per chunk, fewer chunks, better wall-clock) — but the triangular solve cost grows as C³, so verify the wall-clock actually drops before committing.
- **Streaming `predict()` disagrees with chunkwise forward:** check that `S` is initialized to zero at `reset()`, that `k` is normalized identically in both paths, and that fp32-promotion around the solve also happens (or doesn't happen) in both paths.

## Estimated Cost
- 1 Modal A100 run, ~10 min wall, expected energy 25-55 kJ
- ~$0.40

## References
- Yang et al. 2024 "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" (NeurIPS 2024, arXiv 2406.06484) — Algorithm 2 is the chunkwise hand-roll target
- Schlag, Irie, Schmidhuber 2021 "Linear Transformers Are Secretly Fast Weight Programmers"
- Gated DeltaNet (2024 follow-up): arXiv 2412.06464
- Reference open-source impl (read-only; do not import — image lacks it): https://github.com/fla-org/flash-linear-attention
- modded_nanogpt baseline: `/home/seneca/wikitext/submissions/modded_nanogpt/submission.py`
