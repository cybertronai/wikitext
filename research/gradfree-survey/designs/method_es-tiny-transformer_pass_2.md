# ES on a *micro* char-Transformer — pass 2

Direction: **A — much smaller model + many more iters, full 270 s budget.**

## 1. Hypothesis
Pass 1 spec-capped at 150 ES iters and used only 62 s of the 270 s training
budget; mean per-iter NLL was still falling steeply at iter 150 (5.56 → ~3.6).
The dominant constraint was iter count, not algorithm. A 5-10x smaller model
makes each ES forward cheaper and — more importantly — drops the search
dimension `D` from ~232k to ~10k, which is the actual bottleneck for
variance-limited finite-difference gradient estimators (signal ~ 1/sqrt(D)
for fixed `P`). Combined with using the full 270 s budget and a larger
population, we predict **val_char_acc 0.32-0.42** (vs pass-1's 0.19), a clean
lift well above the unigram floor (~0.18). Stretch: 0.50.

## 2. Model
Char-byte vocab (256). Same `GPT` building blocks as pass 1 (so the
`CharModel` streaming wrapper is unchanged), but shrunk hard:

- `n_layers = 2`
- `d_model = 32`, `head_dim = 32`, `n_heads = 1`
- `ffn_dim = 2 * d_model = 64` (ReLU^2 MLP, RMSNorm pre-norm)
- `ctx_len = 32` (smaller — most local char-n-gram signal is within ~16 chars;
  shorter context also cuts forward FLOPs ~2x and KV memory)
- Separate `lm_head` (not tied — keep flat layout simple).

Param count: embed 256*32 = 8,192; per block qkv+proj (4*32*32 = 4,096) +
mlp (32*64 + 64*32 = 4,096) + 2 norms (64) ≈ 8,256; 2 blocks ≈ 16,512;
lm_head 32*256 + 256 ≈ 8,448; +norms. **Target D ≈ 33k params.**

This is ~7x smaller than pass 1's 232k. Embedding + lm_head dominate (~50%);
genuine "search dimension that matters for fitness" is the ~16k transformer
core. That's small enough that P=128 antithetic perturbations give a
reasonable rank-1 finite-difference estimate per iter.

## 3. Training procedure
Identical antithetic OpenAI-ES loop as pass 1, with three tweaks:

1. **Mirrored sampling** (already antithetic — keep) + **centered rank
   shaping** (Wierstra et al. NES): rank fitnesses, map to weights
   `u_i = max(0, log(P/2 + 1) - log(i)) / sum`, then center to zero mean.
   Stronger than the linear rank-norm used in pass 1; cheap.
2. **Sigma annealing**: linear decay from `sigma_0 = 0.05` to `sigma_T = 0.01`
   over the run. Early exploration, late refinement.
3. **Common minibatch within an iter** (already in pass 1), **resample every
   iter** — unchanged.

Use `functorch.functional_call` + `torch.vmap` over the P axis exactly as
pass 1. Fall back to a Python `for p in range(P)` loop if vmap collides with
attention kernels — at d=32, L=2 the sequential loop is ~2 ms/forward, so
P=128 = 0.26 s/iter, still leaves room for ~1000 iters.

## 4. Hyperparameters
- `P = 128` (antithetic, 64 independent noise draws).
- `sigma_0 = 0.05`, `sigma_T = 0.01`, linear schedule.
- `alpha = 0.03` (slightly lower than pass 1's 0.05 — larger P justifies it).
- `B = 32` sequences, `ctx_len = 32` → 1024 tokens/forward (same token count
  as pass 1 for stable fitness variance).
- `n_iters` target: **~800** (budget §5). Hard wall-time cutoff at 260 s of
  training (10 s reserved for setup + final inference / unflatten).
- Init: modded init scheme (zero proj, normal embed, scaled normal others) —
  from scratch, no pretrain.
- Optional global gradient-norm clip: `||dtheta||_inf <= 5 * sigma_t`.

## 5. Expected wall time (A100-80GB, bf16)
One forward at d=32, L=2, T=32, B=32 is ~ 32 * 32 * (2 * 32^2 * 4) FLOPs
≈ 8 MFLOPs (negligible). Vmap over P=128: kernel-launch + memory traffic
dominates. Empirically expect 0.25-0.40 s/iter. Budget: 260 s / 0.35 s ≈
**~740 iters** (vs pass 1's 150). Worst case 0.5 s/iter → 520 iters;
best case 0.15 s/iter → 1700 iters. We will *not* spec-cap early — the
training loop runs until 260 s wall clock elapses, then exits.

## 6. Success criterion
Honest target: **val_char_acc >= 0.32** on the standard 60k val-char slice
within 300 s total on A100-80GB. Stretch: **0.45**. Stretch++: 0.55. We do
not target 0.70 — that needs ~10^3-10^4x more forward passes than the budget.
Energy: secondary; report the harness joule reading, no target. Anti-target:
if val_char_acc < 0.22 (i.e. barely above pass 1), the experiment is a clear
negative result for "more iters + smaller model" as an ES knob at this
budget.

## 7. Failure modes anticipated
- **Capacity ceiling** (design): 33k params may underfit even with infinite
  ES iters. Mitigation: if val NLL plateaus before 50% of budget, the
  bottleneck is capacity, not optimization — report and stop. Tag: design.
- **Sigma collapse / divergence** (design): annealing schedule already
  shrinks sigma; clip per-step ||dtheta||_inf as above. Tag: design.
- **Fitness plateau early** (design): mirrored sampling + centered rank
  shaping already strongest variance-reduction we can do without going to
  CMA. If plateau, accept and report. Tag: design.
- **vmap OOM at P=128** (execution): 128 * 33k * 2 B ≈ 8.5 MB weights;
  activations 128 * 32 * 32 * 32 * 2 * 2 B ≈ 17 MB — trivial. Fallback to
  sequential loop. Tag: execution.
- **functional_call quirks with RMSNorm buffers** (execution): same code
  path as pass 1, which ran cleanly. Tag: execution.

## 8. What we will NOT do
- **No backprop, no autograd.grad, no requires_grad=True on weights**.
- **No pretrained init / no SGD warmup**. From scratch.
- **No CMA-ES** (covariance is 33k^2 = 1 GB fp32 — borderline, but covariance
  updates eat budget; saved for a different pass).
- **No PEPG / per-parameter sigma** this pass — keep direction A clean.
- **No architectural change** (still a transformer, just shrunk) — direction
  B is reserved for a different pass.
- **No guided-ES / surrogate gradients** (would require backprop).
- **No reuse of modded-nanogpt's training loop** — only `GPT` module and
  `CharModel` wrapper.

---

**Param count target: ~33k**
**Success criterion: val_char_acc >= 0.32 on first 60k val chars within 300 s on A100-80GB; stretch 0.45**
