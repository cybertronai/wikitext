# ES on a tiny char-Transformer — pass 1

## 1. Hypothesis
A 4-layer, d=64, ctx=64 char-transformer (~155k params) trained by antithetic
OpenAI-ES can show clearly above-chance next-byte prediction within a 300 s
A100 budget. We expect ES to be many orders of magnitude less sample-efficient
than SGD on the same net (modded-nanogpt reaches ~0.74 char-acc with backprop)
— so the experiment's value is calibrating *how far* a pure forward-pass
evolutionary update can travel in 300 s on a 1.2B-char corpus, and whether
antithetic + rank normalization is enough variance reduction at this scale.

## 2. Model
Char-byte vocab (256). Architecture (reuses building blocks from the
modded_nanogpt baseline so the streaming `CharModel` wrapper is unchanged):
- `n_layers = 4`
- `d_model = 64`, `head_dim = 32`, `n_heads = 2`
- `ffn_dim = 4 * d_model = 256` (ReLU^2 MLP, RMSNorm pre-norm)
- `ctx_len = 64`
- Tied embedding/lm_head NOT used; separate `proj` to keep flat layout simple.

Param count: embed 256*64=16,384; per block ~ qkv+proj (4*64*64=16,384) +
mlp (64*256 + 256*64 = 32,768) + 2 norms (128) ≈ 49,280; 4 blocks ≈ 197k;
lm_head 64*256+256 ≈ 16,640; +norms. **Target D ≈ 230k params.**

Why this tiny: ES update cost scales with `P` forward passes per iter.
At D≈230k, one perturbed parameter vector is ~0.45 MB (fp16). Vectorizing
`P=64` perturbed models via stacked weights costs 64 * (ctx * d^2 * layers)
FLOPs per forward — fits in <1 s/iter on A100, leaving budget for 200+ iters.
A larger net (say modded-nanogpt's 11M params) would either OOM under
vmap-stacking of P copies or require P-fold sequential forwards that
crater the iter rate.

## 3. Training procedure
1. Build `GPT` (above), allocate on CUDA in bf16. Flatten all parameters into
   `theta: Tensor[D]` (fp32 master copy) plus a `_unflatten(theta) -> dict` that
   writes back into module parameters via `torch.nn._functional_call`-style
   stateless forward (functorch `functional_call`), so we never construct an
   autograd graph.
2. Hold a contiguous CUDA `uint8` buffer of the train corpus (as in baseline).
3. Per iter:
   - Sample `eps_half ~ N(0, I)` of shape `[P//2, D]` on GPU, fp32.
   - Antithetic stack: `eps = torch.cat([eps_half, -eps_half], 0)` → `[P, D]`.
   - Build `theta_perturbed = theta[None, :] + sigma * eps` → `[P, D]`.
   - Sample one minibatch: `B` sequences of `ctx_len+1` bytes, shape `[B, T+1]`.
   - For each of the `P` perturbed models compute mean per-token NLL on that
     same minibatch under `torch.no_grad()`. Implementation: `torch.vmap`
     over the leading `P` axis of `theta_perturbed`, calling
     `functional_call(model, unflatten(theta_p), (x,))`. If vmap fails on
     attention kernels, fall back to a Python loop of `P` forwards (still
     fits — see §5).
   - `fitness[i] = -mean_NLL_i` (higher better).
   - Rank-normalize: sort fitnesses, map ranks linearly to `[-0.5, +0.5]`,
     then standardize to unit std. Yields `f_norm: [P]`.
   - `dtheta = (alpha / (P * sigma)) * (f_norm[:, None] * eps).sum(0)`.
   - `theta += dtheta`. Optional weight decay: `theta *= (1 - alpha * wd)`.
4. After loop: unflatten final `theta` into module, return `CharModel` wrapper
   identical to baseline (KV-cached streaming inference is gradient-free
   already, so no changes).

## 4. Hyperparameters
- `P = 64` (antithetic, so 32 independent noise draws).
- `sigma = 0.02` (per-coord perturb std; standard ES default).
- `alpha = 0.05` (learning rate on theta).
- `B = 16` sequences, `ctx_len = 64` → 1024 tokens per forward.
- `n_iters` target: 150 (budget §5).
- Minibatch resample: **every iter** (fresh sample per iter — keeps fitness
  unbiased; common-random-numbers across the `P` perturbations within an iter
  is critical for variance reduction and is automatic here since all P
  share the same minibatch).
- Initial `theta` from modded init scheme (zero proj, normal embed, scaled
  normal others) — from scratch, no pretrain.

## 5. Expected wall time (A100-80GB, bf16)
One forward of d=64, L=4, T=64, B=16 is ~ 16 * 64 * (4 * 64^2 * 4) FLOPs
≈ 17 MFLOPs (negligible). With vmap over P=64: ~1.1 GFLOPs/iter of useful
work, but the kernel launch + memory traffic dominates: empirically expect
~0.5-1.5 s/iter on A100. Budget: leave 30 s for setup + final inference;
270 s / 1.5 s ≈ 180 iters worst case, 540 iters best case. **Target 150 iters
as a safe plan.** If vmap can't stack attention kernels cleanly we fall back
to a `for p in range(P)` loop — at ~5 ms/forward this is 320 ms/iter, still
~800 iters/300 s, so the fallback is *faster* than vmap if vmap-overhead is
large. We will benchmark both in the first 5 iters and pick.

## 6. Success criterion
Honest target: **val_char_acc ≥ 0.20** on first 60k val chars (chance for
printable-ASCII-heavy wikitext is ~0.05-0.10 since space is ~17% of
characters; a unigram baseline reaches ~0.18). Stretch: 0.30. We do not
expect 0.70 — that would require ~10^4× more forward passes than the budget
allows. Joules: secondary; report whatever the harness measures, no target.

## 7. Failure modes anticipated
- **High fitness variance** (design-failure if uncured): rank-norm + antithetic
  + common minibatch already applied; if still noisy, increase `B` to 32
  (halves iter count). Tag: design.
- **Slow convergence / plateau near init** (design): expected; honest target
  reflects this. Mitigation only via more iters (out of scope).
- **vmap OOM when stacking P=64 weight sets** (execution): 64 * 230k * 2 B
  ≈ 30 MB of weights — fine. Activations: 64 * B * T * d * L bf16
  ≈ 64*16*64*64*4*2 = 33 MB — fine. Fallback to sequential loop if vmap
  errors. Tag: execution.
- **Sigma collapse / divergence** (design): if theta explodes, clamp
  `||dtheta||_inf` to `5 * sigma` per step. No adaptive sigma (CMA territory,
  rejected below).
- **functional_call incompatible with KV-cache forward path** (execution):
  training calls model with `kv_caches=None`, identical to baseline's training
  path, so should be fine.

## 8. What we will NOT do
- **No CMA-ES**: D=230k → covariance matrix is 50 GB. Off-budget.
- **No backprop, no autograd.grad, no requires_grad=True on weights**.
- **No pretrained init** — gradient-free constraint forbids prior SGD; we also
  want a clean ES-only signal.
- **No guided-ES / surrogate gradients** this pass — would require backprop on
  a surrogate. Possible follow-up pass.
- **No population-size annealing / sigma adaptation** this pass — fixed
  schedule keeps the experiment legible.
- **No reuse of modded-nanogpt's training loop** — only its `GPT` module and
  `CharModel` streaming wrapper are reused.

---

**Param count target: ~230k**
**Success criterion: val_char_acc ≥ 0.20 on first 60k val chars within 300 s on A100-80GB**

