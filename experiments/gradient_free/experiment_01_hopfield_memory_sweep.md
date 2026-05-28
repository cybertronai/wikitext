# Experiment 01: Hopfield Memory-Bank Size Sweep (M ∈ {1k, 4k, 16k, 65k})

## Hypothesis
Val char-acc of the Hopfield+4L stack increases monotonically with memory size M up to a saturation point, and the energy cost grows sub-linearly in M because per-step softmax over M is bandwidth-bound on bf16 (K_mem stays in HBM/L2). The energy-optimal M is ≥ 16k, not 4k.

## Motivation
`hopfield_layer` (exp 11, 40.2 kJ / 0.7293) used M=4096 — the spec's first guess. The Hopfield-as-attention identity (Ramsauer 2020) and the dense-associative-memory scaling laws (Krotov 2016, Burns 2024) predict storage capacity in the hundreds of thousands of patterns for d=384. We are likely operating far below capacity. This is the highest-EV follow-up because (a) it tests a clear scaling hypothesis, (b) it sits on the only winning direction in the prior portfolio, and (c) it costs only 4 Modal runs.

## Method
Identical to `submissions/hopfield_layer/submission.py` except `hopfield_M ∈ {1024, 4096, 16384, 65536}`. K_mem/V_mem are still frozen, sampled once at init from the random-init encoder. Insertion point and all other hyperparameters held constant.

For M=65536 with d=384 bf16, K_mem = 50 MB, V_mem = 50 MB → 100 MB total, fits in HBM but exceeds L2 (40 MB on A100). The softmax over M=65536 logits is 256 KB per (B,T) position in fp32; for B=32, T=1024 the score tensor is 32·1024·65536·4 B = 8.6 GB which is too large — must chunk T or use flash-attention-style streaming softmax.

## Memory-Movement Analysis
- M=4096 (baseline): K_mem fits in L2. Per step: B·T·M·d = 32·1024·4096·384 = 50 GFLOPs Hopfield; ~3 MB K_mem touched, reused across T positions → bandwidth ≪ compute.
- M=16384: K_mem = 12.6 MB, still in L2. Per step 200 GFLOPs.
- M=65536: K_mem = 50 MB, **exceeds L2** — every Hopfield call re-streams K_mem from HBM. Arithmetic intensity drops by ~3×. Solution: split forward into T-chunks of 64 so each K_mem load is reused over 64·B positions (= 2048 queries per K_mem byte loaded — restores compute bound).
- The score matrix must be chunked: compute it for T_chunk=64 at a time so the (B,T_chunk,M) score tensor is 32·64·M·4 B = 32 MB (fits in HBM working set).

## Setup
- Dataset, model, optimizer, n_steps, batch=32, T=1024 — all identical to `hopfield_layer/submission.py`
- Sweep: M ∈ {1024, 4096, 16384, 65536}
- Baseline (already on leaderboard): `hopfield_layer` (M=4096) = 40,158 J / 0.7293
- Reference: `modded_nanogpt` = 51,704 J / 0.7374

## Procedure
1. `cp -r submissions/hopfield_layer submissions/hopfield_M1k` (and likewise for M16k, M65k)
2. In each, edit `TrainConfig.hopfield_M` to the target M.
3. For M=65536, add T-chunked Hopfield forward:
```python
def forward(self, q):
    B, T, d = q.shape
    out = torch.empty_like(q)
    K = self.K_mem.float(); V = self.V_mem.float()
    for t in range(0, T, 64):
        qc = q[:, t:t+64].float()
        scores = (qc @ K.t()) * self.scale
        out[:, t:t+64] = (F.softmax(scores, dim=-1) @ V).type_as(q)
    return out + q
```
4. `python submit.py submissions/hopfield_M{1k,16k,65k} --yes` (three Modal runs; baseline already exists).
5. Compare energy, val char-acc, train duration.

## Success Criteria
- **Confirmation**: monotonic acc improvement from M=1k → 16k.
- **Strong**: M=16k or M=65k beats modded_nanogpt (0.7374) at energy ≤ 50 kJ.
- **Refutation**: M=16k acc ≤ M=4k acc → capacity already saturated; the lift is from the layer's *existence*, not its capacity.

## Failure Modes & Diagnostics
- bf16 softmax underflow over M=65536: log max-vs-mean of attention weights at training step 1000 — if max attn weight > 0.99 for >50% of queries the softmax has collapsed; if so drop temperature scale to 1/√(4d).
- Memory bank stale: K_mem built from random-init encoder; for large M the noise dominates. Diagnostic: rebuild K_mem at step 500 using the partly-trained encoder; compare against no-rebuild.
- Score tensor OOM at M=65k: T-chunking above prevents this.

## Estimated Cost
3 Modal A100-80GB runs × ~10 min wall each ≈ $1.25 (M=4k baseline already in leaderboard).

## References
- `experiments/kernel_methods/result_11.md` — prior Hopfield result
- Ramsauer et al. 2020 "Hopfield Networks Is All You Need" (arXiv 2008.02217)
- Burns 2024 "Modern Hopfield Networks with Continuous-Time Memories" (arXiv 2502.10122)
- Wu et al. 2022 "Memorizing Transformers" (ICLR)
