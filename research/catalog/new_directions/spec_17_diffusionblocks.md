# Research Specification 17: DiffusionBlocks AR for Byte-Level LM

**Status:** Hypothesis evaluation (methodological replication of Sakana AI ICLR 2026)
**Priority:** Medium — methodological / paradigm-completeness play
**Estimated effort:** 3–5 days

---

## Hypothesis

A 6-layer/384-dim modded-nanogpt-style transformer trained as **DiffusionBlocks** (Shing, Koyama, Akiba — ICLR 2026, [arxiv:2506.14202](https://arxiv.org/abs/2506.14202)) with `B=3` blocks reaches val char-acc ≥ 0.70 within 300 s on A100-80GB, qualifying as the first principled block-wise, single-block-gradient training method in the catalog. Energy is reported but not targeted — whether it lands above or below current leaders is itself the experimental question.

This is a controlled replication of the only block-wise training method in the literature that has been shown to match end-to-end backprop on a modern generative task (OWT gen-PPL 14.99 vs 15.05; see paper Table 4). The catalog currently has Forward-Forward (spec 10) as its block-wise representative and that line of work has stayed sub-floor — DiffusionBlocks is the obvious next attempt because it has *already shown* it can match E2E on language.

---

## Background

### What DiffusionBlocks does

Re-interprets a residual stack `z_{ℓ+1} = z_ℓ + f_θ(z_ℓ)` as an Euler discretization of the EDM probability-flow ODE (Karras et al. 2022). Partition `L` layers into `B` contiguous blocks; assign each block a noise interval `[σ_b, σ_{b-1}]`. Each block independently denoises its target `y` against the EDM loss

```
L_b(θ_b) = E_{(x,y), σ ~ p_noise|[σ_b, σ_{b-1}], ε ~ N(0,I)} [ w(σ) · || f̄_θb|σ(x, y + σε) − y ||² ]
```

with `p_noise = LogNormal(P_mean=-1.2, P_std=1.2)`, `w(σ) = (σ² + σ_data²)/(σ·σ_data)²`, and EDM preconditioning `c_skip, c_out, c_in, c_noise`. Conditioning enters via AdaLN with `c_noise = 0.25·log σ`.

**Block boundaries** use **equi-probability partitioning** (paper §3.3, `dblock_modules.py:6`): each `σ_b` chosen so that `∫_{σ_b}^{σ_{b-1}} p_noise dσ = 1/B`. Closed-form: `σ_b = exp(P_mean + P_std · Φ⁻¹(q_min + (b/B)·(q_max − q_min)))`.

**Per-step training** (paper Algorithm "DiffusionBlocks – Training"):
1. Sample `(x, y)` and block `b ~ Uniform{1..B}`
2. Sample `σ ~ p_noise|[σ_b, σ_{b-1}]`
3. Forward through layers below block `b` (no grad), then forward through block `b` with `z_t = y + σε` and noise conditioning
4. Backward only through block `b`
5. Step optimizer on `θ_b`

This gives **B× memory reduction** for activations and stores gradients for only `L/B` layers, but does **not** reduce per-step FLOPs proportionally (you still forward through everything; only backward is local).

### AR-specific adaptation (paper §5.4)

The paper's AR experiment is the closest precedent to what we need; details (paraphrased, since hyperparameters are not published and the public repo only releases the CIFAR-100 ViT variant):

- **Architecture:** 12-layer Llama-2-style transformer, `B=4` blocks (3 layers each), token-level BPE (not bytes).
- **Datasets:** 1 Billion Words (LM1B) and OpenWebText (OWT).
- **Training reframe:** at every position, the target `y_t` is the **next token's L2-normalized output embedding** (weight-tied with the unembed). A noised version `z_t = embed(y_t) + σε` is fed in alongside the causal-attended prefix features `h_{≤t}`; the block at the sampled `σ` denoises back to `embed(y_t)`. Loss = EDM-weighted L2 on the embedding plus an auxiliary CE on `softmax(pred · E_outᵀ)` against `y_t` (verbatim from `model.py:254–259`).
- **Inference:** standard left-to-right token-by-token, but **each token requires `T=B=4` Euler steps** through the blocks from `z_0 ~ N(0, σ_max²·I)` to the predicted embedding, then a final unembed projection. Generation cost ≈ one full network forward pass per token (same wall-clock per token as a standard AR Llama).
- **Evaluation:** *not* standard validation perplexity. The paper notes "computing traditional perplexity is non-trivial for our diffusion framework as it is not derived from ELBO." They instead report **MAUVE** (similarity of generated to real text, following SEDD) and **generative perplexity** scored by two external teacher models, **Llama-2-7B and GPT2-XL**. Headline numbers (Table 4): OWT MAUVE 0.82 (DiffusionBlocks) vs 0.85 (E2E), Llama-2 gen-PPL 14.99 vs 15.05, GPT2-XL gen-PPL 26.33 vs 25.24.
- **Not stated in paper:** sequence length, batch size, optimizer / LR, total training tokens, FLOPs, peak memory delta vs E2E. The `B=4 → 4× memory reduction` claim is the only quantitative training-side number.

**The wrinkle for this benchmark:** the leaderboard scores **greedy-argmax next-char accuracy** via the streaming `CharModel.predict()` API — a standard next-token-prediction metric that Sakana explicitly sidestepped for the AR experiment. The auxiliary CE term during training is what lets us project the denoised embedding back to a calibrated byte distribution and read off `argmax`. We are therefore measuring DiffusionBlocks-AR under a metric (greedy next-byte accuracy) the original work *did not validate*, on a corpus (WikiText-103 raw bytes) it was not run on. Phase 0 below probes whether this metric/objective combination is even sensible before any Modal spend.

### Catalog placement

| Method | Block-wise? | Matches E2E? | Status here |
|---|---|---|---|
| Forward-Forward (spec 10) | yes | no (paper: ViT CIFAR-100 7.85% vs 60.25%) | sub-floor in `research/forward-forward-deep/` |
| NoProp (Li et al. 2025) | yes | partial; classification only | not catalogued |
| **DiffusionBlocks** | **yes** | **yes (paper Tables 1–5)** | **this spec** |

DiffusionBlocks is the only published method that gives both (a) one-block-at-a-time gradients and (b) E2E-matching on language. Even a DQ entry adds a real comparison point against the spec-10 FF line.

---

## What to build

A new submission `submissions/diffusionblocks_ar/submission.py` that mirrors `submissions/modded_nanogpt/submission.py`'s scaffolding (byte vocab, RMSNorm, RoPE, ReLU² MLP, Muon+AdamW) but replaces the training loop and inference path with DiffusionBlocks.

### Architecture

- Vocab 256 (bytes), `L=6` transformer layers, `d=384`, `head_dim=64`, `max_len=1024`, RoPE.
- Partition into `B=3` contiguous blocks of 2 layers each. (Phase-2 sweep: `B ∈ {2, 3, 4}`.)
- **AdaLN injected at each block's two RMSNorms.** Standard form: `LayerNorm(x) * (1 + γ(c)) + β(c)` where `(γ, β) = MLP(c_noise(σ))`. `c_noise = 0.25 · log σ` (EDM convention, `model.py:206`).
- **Target embedding head.** A normalized embedding table `E_out ∈ R^{256 × d}`, weight-tied with the next-byte projection. Embeddings are L2-normalized along `d` so `σ_data ≈ 1/√d ≈ 0.05`. The exact `σ_data` enters EDM preconditioning — calibrate from data (see Phase 0 below).

### Noise schedule (verbatim from Sakana `dblock_modules.py`)

```python
def get_block_sigmas(B, sigma_min=0.002, sigma_max=80.0,
                    P_mean=-1.2, P_std=1.2):
    cdf_min = norm.cdf((np.log(sigma_min) - P_mean) / P_std)
    cdf_max = norm.cdf((np.log(sigma_max) - P_mean) / P_std)
    return [np.exp(P_mean + P_std *
                   norm.ppf(cdf_min + (cdf_max - cdf_min) * b / B))
            for b in range(B + 1)]
```

For `B=3` this gives boundaries roughly `[0.002, 0.13, 1.85, 80]` — block 0 handles `[0.002, 0.13]` (low noise), block 1 the high-density intermediate range, block 2 high noise.

### Per-step training pseudocode

```python
for step in range(B * baseline_steps):                  # paper compensates B× for one-block grads
    x_<t, y_t = sample_batch()                          # standard causal LM batch, B×T
    b = randint(0, B)
    sigma = sample_log_normal_truncated([sigma_b, sigma_{b-1}])

    # Conditioning context (no grad): standard causal transformer features
    # at every position from the *frozen-this-step* lower blocks.
    with torch.no_grad():
        h = base.embed(x_<t)
        for bb in range(b):
            h = blocks[bb].forward_features(h, sigma=None)  # std residual, no AdaLN

    # Targets at every position: next-byte normalized embedding
    y_embed = normalize(E_out[y_t])                       # (B, T, d)
    z_t = y_embed + sigma * torch.randn_like(y_embed)

    # Apply preconditioning; only block b has grad
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out  = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in   = 1.0 / (sigma**2 + sigma_data**2).sqrt()
    c_noise = 0.25 * sigma.log()

    z_in = z_t * c_in
    out = blocks[b](h, z=z_in, cond=c_noise)               # AdaLN conditioning
    pred_y = out * c_out + z_t * c_skip

    # EDM L2 on the embedding...
    loss_l2 = w(sigma) * (pred_y - y_embed).pow(2).mean()
    # ...plus auxiliary CE on the projected logits (Sakana model.py:254–259 hybrid)
    logits = pred_y @ E_out.T
    loss_ce = F.cross_entropy(logits.view(-1, 256), y_t.view(-1))
    loss = loss_l2 + lambda_ce * loss_ce                   # lambda_ce ≈ 1.0 to start

    loss.backward()  # only block b accumulates grads
    opt[b].step()
```

Per-block optimizer state is independent (separate AdamW/Muon instances keyed on `b`), or one optimizer with `requires_grad` toggled per step — both work. The Sakana repo (`model.py:50`) uses one optimizer across all params and relies on Lightning's `find_unused_parameters` (`main.py:67`) to absorb the per-step subset.

### Inference (`CharModel`)

The streaming contract requires `predict()` → byte distribution and `observe(byte)` → commit. With B Euler steps per byte:

```python
@torch.no_grad()
def predict(self) -> dict[str, float]:
    # 1. Get cached prefix features h_<t at current position (all B blocks have
    #    independent context KV-caches, computed during observe()).
    h = self._cached_h_at_pos  # (1, d), set by previous observe()

    # 2. Initialize z from pure noise
    z = torch.randn(1, self.d, device=device) * self.sigmas[0]

    # 3. Euler-step through B blocks (one step per block, σ_i ↓)
    for i in range(B):
        sigma = self.sigmas[i]
        c_in, c_out, c_skip, c_noise = edm_precond(sigma)
        denoised = (blocks[i](h, z=z*c_in, cond=c_noise) * c_out
                    + z * c_skip)
        d = (z - denoised) / sigma
        dt = self.sigmas[i+1] - sigma
        z = z + d * dt

    # 4. Project final z to byte logits
    logits = z @ self.E_out.T
    probs = F.softmax(logits, dim=-1)
    return {bytes([i]).decode("utf-8", errors="ignore"): p
            for i, p in enumerate(probs.tolist()) if p > 0}

@torch.no_grad()
def observe(self, ch: str) -> None:
    for byte in ch.encode("utf-8"):
        x = torch.tensor([[byte]], device=device)
        h = self.embed(x)
        # Extend per-block KV-caches; cache the final-block h for the next predict()
        for b in range(B):
            h, self._kv[b] = blocks[b].forward_features(
                h, kv_cache=self._kv[b], offset=self._pos)
        self._cached_h_at_pos = h[0, -1]
        self._pos += 1
```

**Inference cost per byte:** B block forwards (each L/B layers) ≈ one full L-layer pass — i.e., same as standard AR. No worse on per-char eval throughput than modded_nanogpt. (Note: B Euler iterations of dimension `d` per char are negligible relative to per-layer matmul.)

### Hyperparameters (starting point)

The Sakana paper does **not** publish AR hyperparameters. EDM defaults plus their CIFAR-100 recipe imply:

| Knob | Value | Source |
|---|---|---|
| `σ_min, σ_max` | 0.002, 80.0 | EDM, `dblock_modules.py:8` |
| `P_mean, P_std` | -1.2, 1.2 | EDM, `dblock_modules.py:10` |
| `σ_data` | calibrate (see Phase 0); fallback 0.5 | `model.py:114` |
| `γ` (block-overlap) | 0.05 | `model.py:113`, extends each block's training range by 5% in log-σ |
| `λ_ce` | 1.0 | inferred from `model.py:254–259` (CE and L2 both included unweighted) |
| Optimizer | AdamW, lr=5e-4 | repo README "rand-aug" recipe |
| LR schedule | cosine, warmup `3·baseline_warmup` | repo README, scales with B |
| Total steps | `B × N_baseline` | `main.py:46–49` |
| Batch / seq | 32 / 1024 | from modded_nanogpt to keep apples-to-apples |
| CFG / class dropout | 0.0 / 0.0 | n/a for text-AR |

Muon vs AdamW split: keep modded_nanogpt's split (Muon for 2-D block weights, AdamW for embeddings/scalars). AdaLN params join the AdamW group.

---

## Phase 0: σ_data calibration and seq-budget check (≤ 1 day)

Before any Modal spend, two sanity checks locally (CPU is fine, ~minutes):

1. **σ_data calibration.** Build the byte embedding, L2-normalize, sample a batch of next-byte embeddings, compute the empirical per-coordinate std. This is `σ_data`. Expected ~0.05 for `d=384`. Plug into the loss weighting; without this, EDM weighting is miscalibrated and training won't move.

2. **Step-count check.** Profile one DiffusionBlocks training step (forward all blocks + backward one) vs one modded_nanogpt step on CPU at small scale. Confirm the per-step time ratio matches the expected `(L + L/B) / (2L) ≈ (B+1)/(2B)` — for B=3 that's ~0.67×. Combined with B× more steps, total wall-clock ratio is ~2× modded_nanogpt's per-step time × B = ~2× total. **If this projection exceeds 300 s at modded_nanogpt's settings, drop batch_size or n_steps before Modal dispatch.**

---

## Phase 1 experiment (go/no-go gate)

**Goal:** confirm DiffusionBlocks at byte level reaches the 0.70 floor and measure the joule delta vs modded_nanogpt.

**Procedure:**

1. Implement `submissions/diffusionblocks_ar/submission.py` per the spec above.
2. Local smoke-test: one 10-step training run on CPU with d=64/L=2/B=2 to shake out shape bugs.
3. Submit via `python submit.py submissions/diffusionblocks_ar/`.
4. Record val char-acc, training joules, training duration, peak GPU memory, per-block loss curves.
5. If val char-acc < 0.70 on first run, **one** remediation: try `B=2` (shorter Euler chain, more layers per block, closer to E2E). One retry — no extended hparam search at Phase 1.

**Measurements to record (write to `result.json` + `report.json`):**

- Val char-acc and energy (J), test char-acc (informational)
- Training duration (s), per-step time (ms), total optimizer steps
- Parameter count vs modded_nanogpt baseline (should match within ±5%)
- Peak GPU activation memory (anecdotal — not scored, but the *whole point* of DiffusionBlocks)
- Per-block training loss curves (3 series for B=3): validates equi-probability balanced learning
- Eval throughput (chars/s) — sanity-check that the B-step Euler inference is not a runaway

---

## Go/no-go criteria

**Go (qualifying):** val char-acc ≥ 0.70. The first principled block-wise method in the catalog that clears the floor. Triggers Phase 2 sweep. Energy is recorded and reported but does not gate Phase 2 — we don't know what value to set the threshold at, and the point of the spec is to find out.

**No-go (DQ):** val char-acc < 0.70 after the one allowed remediation. Two interpretations distinguish themselves by *which* block fails:
1. *Low-σ block (block 0) fails:* the denoiser cannot recover bytes from near-clean embeddings — calibration / `σ_data` issue. Recoverable in a follow-up; doesn't kill the paradigm.
2. *High-σ block (block B-1) fails:* the network cannot map noise to anything useful — fundamental: paradigm doesn't transfer. Report and close.

Either way, file a DQ row with per-block loss curves so the failure mode is in the record.

---

## Phase 2 (conditional on Go)

1. **B sweep: {2, 3, 4}.** Sakana Table 8 shows non-monotone FID in B on ImageNet (best at B=2). Repeat the pattern here. Costs 2 Modal runs.

2. **Width sweep at fixed wall-clock.** The point of B× memory reduction is to scale. Hold `B=3` and total wall-clock ≈ 280 s; bump `d_model` from 384 → 512 → 640. If saved memory translates to better accuracy at the same energy, we have a real story. Costs 2 Modal runs.

3. **Ablation: uniform vs equi-probability partitioning.** Paper Table 7 shows equi-prob gives FID 38 vs uniform's 43 on CIFAR-10 — a major effect. Repeat at byte level. Costs 1 Modal run.

4. **(Stretch)** DiffusionBlocks on top of `alpha_06`'s recipe instead of modded_nanogpt's, to test composition with the current leader.

---

## What a positive result means

A positive result is the **first evidence that one-block-at-a-time gradient training is competitive with end-to-end backprop on a real character-level LM benchmark**. The catalog gains a working reference implementation of a published 2026 ICLR method, and the spec-10 Forward-Forward line gets a contrasting data point ("FF fails at byte level; DiffusionBlocks does not").

The energy number that comes out is itself the most interesting deliverable, because there is no published comparison: training-step FLOPs are comparable to E2E (full forward + partial backward, ×B more iterations per Sakana's iteration-multiplier convention) but the activation-memory headroom may enable wider models inside the 300 s budget that E2E can't fit. Phase 2's width sweep is the natural follow-up to test that.

---

## What a negative result means

A DQ tells us one of three things:

1. **Per-block CE objective + EDM denoising signal does not transfer to byte-level discrete outputs in 300 s.** The image-domain successes used continuous targets; the masked-diffusion text result (text8, BPC 1.45 vs 1.56) used a discrete-diffusion native formulation that we are *not* replicating here. The next-byte-embedding denoising is closer to Sakana's AR setup but unproven at byte level under tight wall-clock.

2. **B× iteration multiplier blows the 300 s budget.** Mitigated by Phase 0 step-budget check, but worth restating: if we cannot afford `B × N_baseline_steps` in 300 s, we cannot match Sakana's reported convergence behavior.

3. **Equi-probability partitioning misallocates capacity at byte level.** Bytes have very local statistics; the noise levels where "structure emerges" in EDM (intermediate σ) may not correspond to anything semantically meaningful for a 256-vocab discrete distribution. This would be a *theory* finding (the framework doesn't decompose well for discrete-vocab AR) worth a short writeup even on a DQ.

In all three cases the result is informative about block-wise training's reach into language modeling, which is the open question the spec exists to answer.

---

## Resources

- Paper: Shing, Koyama, Akiba — "DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation" — ICLR 2026, [arxiv:2506.14202v3](https://arxiv.org/abs/2506.14202)
- Reference impl: https://github.com/SakanaAI/DiffusionBlocks (CIFAR-100 ViT only; AR not released)
- Key code: `dblock_modules.py` (noise scheduling), `model.py:110-291` (training/inference loop with EDM preconditioning, hybrid L2+CE loss, equi-probability sampling)
- EDM reference: Karras et al. 2022, [arxiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Baseline to modify: `submissions/modded_nanogpt/`
- Related catalog entries: `spec_10_forward_forward_bytes.md` (block-wise, sub-floor), `research/forward-forward-deep/` (multi-phase FF investigation)
- Harness: 300 s, A100-80GB, NVML+CPU joules, val char-acc ≥ 0.70 on 60K val chars
