# Result 03 — Performer FAVOR+ Drop-In Replacement of Attention in modded-nanogpt

## Hypothesis (recap)
FAVOR+ (Performer, Choromanski 2020) with M=128 positive random features replaces softmax attention in modded-nanogpt and clears the 0.70 val gate at *lower energy* than the 51.7 kJ baseline because attention is linearized.

## Final numbers (best of two iterations)
| field | value |
|---|---|
| training_energy_J | **70,742 J** (at 300 s kill) |
| training_duration_s | 300.0 (wall-cap killed) |
| val_char_accuracy | **not evaluated** — training never completed |
| DQ status | **DISQUALIFIED** — `train_time_exceeded` |
| best train loss before kill | 1.6162 (step 500 of 1200, M=64 run) |

Run-1 (M=128, n_steps=2500): 300 steps before kill, loss 2.09, 71,806 J.
Run-2 (M=64 + bf16 cumsum, n_steps=1200): 500 steps before kill, loss 1.62, 70,742 J.

## A/B vs modded_nanogpt baseline
Baseline: **51,704 J → 0.7374 acc** in ~250 s wall.

Performer FAVOR+: **70,742 J → DQ (no eval)** in 300 s wall, completing only ~42% of the planned schedule. Even per joule, Performer is ~37 % *more* energy-hungry than the baseline at the cap point and never reaches the gate.

The Performer per-step throughput on this model (B=32, H=6, T=1024, D=64) is bottlenecked by the (B,H,T,M,D) cumsum tensor (~3 GB at M=128 bf16, ~1.5 GB at M=64 bf16). The HBM traffic for materializing and grad-checkpointing that tensor erased the theoretical FLOP win — empirically FAVOR+ was **~7.5× slower per step** at M=128 and **~4.0× slower per step** at M=64 than the SDPA baseline (which uses FlashAttention internally and never materializes a T×T matrix). The 8× per-step FLOP reduction claimed in the spec did not translate to wall-clock at this scale because (a) FlashAttention is already extremely HBM-efficient and (b) FAVOR+'s outer-product cumsum is HBM-heavy.

## Success Criterion bracket
**Refuted on energy** (not on accuracy — accuracy was never measured because we hit the wall-cap first).

This is closest to spec criterion "Verified + no energy win" extrapolated to its failure mode: per-step compute did not amortize even with a halved n_steps. The qualitative loss trajectory (1.62 at step 500 with no NaNs and a healthy curve) suggests FAVOR+ would converge given enough wall budget, so the algorithm is *correct* but loses to a tuned softmax baseline on this hardware at this scale.

## Interpretation (does the Performer claim transfer to char-LM at 6L/384d?)
**No, not as an energy win on A100-80GB.** Two structural reasons:

1. **The Performer FLOP advantage is asymptotic in T, not the operative regime here.** At T=1024 with FlashAttention SDPA, softmax attention is already near memory-bandwidth-limited at ~0.12 s/step. FAVOR+ replaces a fused FlashAttention kernel with two large einsums + a cumsum that PyTorch does not fuse, so the "8× cheaper" FLOP count is moot — wall time is dominated by HBM round-trips on the (B,H,T,M,D) state tensor.
2. **M is a hard knob.** Dropping M from 128 to 64 nearly doubled per-step throughput with no loss-curve degradation visible in the first 500 steps, suggesting M=64 is sufficient at 6L/384d. But M=32 risks loss-of-precision (Choromanski reports variance scaling as 1/M), so this only buys us ~2×.

The numerical-instability fix from the spec (pre-divide q, k by √D before the feature map) worked: no NaNs in either run, healthy monotone-decreasing loss. The recurrent O(1)-per-byte streaming inference path was also implemented (running (S, z) state instead of (K, V) cache) but was never exercised because eval never ran.

**Verdict on the survey claim:** at char-LM scale with FlashAttention available, Performer's linear-attention claim is a wash-or-loss for training energy. It would likely become a win at T ≥ 8192 or on hardware without FlashAttention kernels.

## Implementation deviations from the spec
1. **M = 64, not 128** (iteration 2). Spec authorized this as the OOM-fallback; we used it as a throughput-fallback after run-1's wall-time DQ.
2. **n_steps = 1200, not "2-3× more steps than baseline"** (iteration 2). With per-step throughput ~4× slower than baseline, even matching baseline's step count was infeasible in 300 s.
3. **Cumsum tensor cast to bf16** (iteration 2). Spec implicitly assumed fp32; I downcast features+v to bf16 before the (B,H,T,M,D) outer product so the dominant HBM-bound op moves half the bytes. Features are still computed in fp32 (exp() safety) and downcast only after the max-subtraction; this is mathematically safe because the post-softmax-substitute values are bounded in [0, 1].
4. **Per-(B,H,T) max-subtraction on qf only, per-(B,H) global max-subtraction on kf.** The spec's bare `exp(omega^T x − ‖x‖²/2)/√M` overflows in bf16 on some heads. Subtracting any constant from the qf row, or any global constant from kf, cancels in the num/den ratio — so this is a free numerical stabilizer that the spec didn't call out but which the codebase needed.
5. **Streaming inference uses (S, z) recurrent state (O(1) per byte)** as the spec recommended, not the parallel-cumsum-on-trailing-context fallback. Never exercised because we DQ'd before eval.

## Paths
- Submission: `/home/seneca/wikitext/submissions/performer_favor/submission.py`
- Result JSON: `/home/seneca/wikitext/submissions/performer_favor/result.json`
- Run logs: `/home/seneca/wikitext/submissions/performer_favor/submit_console.log` (M=128), `submit_console2.log` (M=64 + bf16)

## Review (post-hoc audit)

**Validity for discarding Performer / FAVOR+ on char-LM:** *Insufficient.*

**Core limitations:**
- **DQ reason is budget over-allocation, not method failure.** Run was killed at step 500/1 200 with `train_time_exceeded`; the loss trajectory (5.55 → 2.43 → 1.85 → 1.62 → ...) was still falling rapidly at the kill point. A correctly-scoped run would set `n_steps` so that the cooldown finishes inside the 300 s cap — there is no per-step pathology, only step-count mis-sizing.
- **Two iterations both DQ'd on time.** Neither M=128 nor M=64+bf16 produced a final val number. Without a completed run there is no val-acc to compare against the 0.70 gate, so no claim can be made about the method's accuracy ceiling on this benchmark.
- **No throughput/step-cost diagnostic in the spec.** The spec's roofline analysis predicted ~3 GB activation memory at M=128 but did not predict step-time; the actual ~500 ms/step was outside what the 1 200-step plan allowed.

**Verdict:** Result contains zero evidence about whether Performer FAVOR+ clears 0.70 on this benchmark. A re-run with `n_steps` chosen to fit (≈ 500 steps with cooldown) is the prerequisite for any verdict.
