# Research Specification 17: DiffusionBlocks AR for Byte-Level LM

## Hypothesis

A 6-layer/384-dim modded-nanogpt-style transformer trained as **DiffusionBlocks** (Shing, Koyama, Akiba — ICLR 2026, [arxiv:2506.14202](https://arxiv.org/abs/2506.14202)) with `B=3` blocks reaches val char-acc ≥ 0.70 within 300 s on A100-80GB, qualifying as the first principled block-wise, single-block-gradient training method in the catalog.

This is a controlled replication of the only block-wise training method in the literature that has been shown to match end-to-end backprop on a modern generative task. The catalog currently has Forward-Forward (spec 10) as its block-wise representative and that line of work has stayed sub-floor — DiffusionBlocks is the obvious next attempt because it has *already shown* it can match E2E on language.

---

## Background

### What DiffusionBlocks does

Re-interprets a residual stack `z_{ℓ+1} = z_ℓ + f_θ(z_ℓ)` as an Euler discretization of the EDM probability-flow ODE (Karras et al. 2022). Partition `L` layers into `B` contiguous blocks; assign each block a noise interval `[σ_b, σ_{b-1}]`. Each block independently denoises its target `y` against the EDM loss

```
L_b(θ_b) = E_{(x,y), σ ~ p_noise|[σ_b, σ_{b-1}], ε ~ N(0,I)} [ w(σ) · || f̄_θb|σ(x, y + σε) − y ||² ]
```

with `p_noise = LogNormal(P_mean=-1.2, P_std=1.2)`, `w(σ) = (σ² + σ_data²)/(σ·σ_data)²`, and EDM preconditioning `c_skip, c_out, c_in, c_noise`. Conditioning enters via AdaLN with `c_noise = 0.25·log σ`.

**Block boundaries** use **equi-probability partitioning** (paper §3.3, `dblock_modules.py:6`): each `σ_b` chosen so that `∫_{σ_b}^{σ_{b-1}} p_noise dσ = 1/B`. Closed-form: `σ_b = exp(P_mean + P_std · Φ⁻¹(q_min + (b/B)·(q_max − q_min)))`.

**Per-step training** (paper Algorithm 1, `paper.md:326-341`):
1. Sample `(x, y)` and block `b ~ Uniform{0..B-1}`
2. Sample `σ ~ p_noise|[σ_b, σ_{b+1}]`
3. Compute `z_σ = y + σε` and the denoiser output `D_θb(z_σ, σ, x)` — block `b`'s transformer layers are the only ones forwarded this step; the lower blocks are *not* forwarded (the shared embed / unembed are still touched, since they're the only path between tokens and the latent space)
4. Backward through block `b` (only block touched this step)
5. Step optimizer on `θ_b`

This gives **B× memory reduction** for activations (only the active block's forward is materialised) and a **B× per-step FLOP reduction** — both the forward and backward at a step are local to one block. The matching 3× inference speedup is reported in paper §4.1 (`paper.md:174`).

### AR-specific adaptation (paper §5.4)

The paper's AR experiment is the closest precedent to what we need; details (paraphrased, since hyperparameters are not published and the public repo only releases the CIFAR-100 ViT variant):

- **Architecture:** 12-layer Llama-2-style transformer, `B=4` blocks (3 layers each), token-level BPE (not bytes).
- **Training reframe:** at every position, the target `y_t` is the **next token's L2-normalized output embedding** (weight-tied with the unembed). A noised version `z_t = embed(y_t) + σε` is fed in alongside the causal-attended prefix features `h_{≤t}`; the block at the sampled `σ` denoises back to `embed(y_t)`. Paper Algorithm 1 step e specifies **CE-only** for LM: `L = w(σ) · CE(Normalize(D_θᵢ(...)), y)`. **We deliberately diverge here**: see "What to build" below — we add an L2-on-embedding term alongside the CE because (a) the paper's CE-only recipe was validated on BPE tokens, not byte-level, and the EDM L2 signal is what drives denoising convergence on the continuous embedding target; (b) the auxiliary CE is what calibrates `E_out`-projected logits for our greedy-argmax `predict()` path.
- **Inference:** standard left-to-right token-by-token, but **each token requires `T=B=4` Euler steps** through the blocks from `z_0 ~ N(0, σ_max²·I)` to the predicted embedding, then a final unembed projection. Generation cost ≈ one full network forward pass per token (same wall-clock per token as a standard AR Llama).
- **Not stated in paper:** sequence length, batch size, optimizer / LR, total training tokens, FLOPs, peak memory delta vs E2E. The `B=4 → 4× memory reduction` claim is the only quantitative training-side number.

**The wrinkle for this benchmark:** the leaderboard scores **greedy-argmax next-char accuracy** via the streaming `CharModel.predict()` API — a standard next-token-prediction metric the paper did not validate for the AR experiment. The auxiliary CE term during training is what lets us project the denoised embedding back to a calibrated byte distribution and read off `argmax`. We are therefore measuring DiffusionBlocks-AR under a metric (greedy next-byte accuracy) the original work *did not validate*, on a corpus (WikiText-103 raw bytes) it was not run on. Phase 0 below probes whether this metric/objective combination is even sensible before any Modal spend.

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
    sigma = sample_log_normal_truncated([sigma_b, sigma_{b+1}])

    # Targets at every position: next-byte normalized embedding
    y_embed = normalize(E_out[y_t])                     # (B, T, d)
    z_t = y_embed + sigma * torch.randn_like(y_embed)

    # EDM preconditioning; only block b's transformer layers are forwarded
    # this step. Lower blocks are NOT forwarded — that's the source of the
    # B× memory and B× per-step FLOP reductions. (The shared token embed
    # for x and the shared E_out unembed are still touched, but their cost
    # is small relative to a block of transformer layers.)
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out  = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in   = 1.0 / (sigma**2 + sigma_data**2).sqrt()
    c_noise = 0.25 * sigma.log()

    z_in = z_t * c_in
    out = blocks[b](x_<t, z=z_in, cond=c_noise)         # block does its own causal attn over x
    pred_y = out * c_out + z_t * c_skip

    # Paper Alg.1 step e for LM is CE-only on the normalised denoiser
    # output. We add an EDM-weighted L2 term on the embedding — deliberate
    # divergence (see "AR-specific adaptation" above). L2 drives denoising
    # on the continuous target; CE calibrates E_out for greedy-argmax eval.
    loss_l2 = w(sigma) * (pred_y - y_embed).pow(2).mean()
    logits = pred_y @ E_out.T
    loss_ce = F.cross_entropy(logits.view(-1, 256), y_t.view(-1))
    loss = loss_l2 + lambda_ce * loss_ce                # lambda_ce ≈ 1.0 to start

    loss.backward()  # only block b accumulates grads
    opt[b].step()
```

Per-block optimizer state is independent (separate AdamW/Muon instances keyed on `b`), or one optimizer with `requires_grad` toggled per step — both work.

### Inference (`CharModel`)

The streaming contract requires `predict()` → committed character (`str`) and `observe(char)` → commit. The submission picks its own sampling rule; here we take greedy argmax over byte logits, which is well-defined because the auxiliary CE head (the `λ_ce` term in training, above) calibrates `E_out`-projected logits to a byte distribution.

Per paper §4.1 (`paper.md:174`), each Euler step at noise level σ uses **only the block responsible for σ** — blocks are not chained at inference any more than they are at training. That means `observe()` extends each block's KV-cache independently from the shared token embedding (no h flowing block-to-block), and `predict()`'s B Euler steps each read from one block's own KV-cache only.

```python
@torch.no_grad()
def predict(self) -> str:
    # 1. Initialise z from pure noise at σ_max
    z = torch.randn(1, self.d, device=device) * self.sigmas[0]

    # 2. Euler-step through B noise levels. Step i uses ONLY block i,
    #    which reads from its own per-block KV-cache (built independently
    #    by observe() — no shared "h" flowing through other blocks).
    for i in range(B):
        sigma = self.sigmas[i]
        c_in, c_out, c_skip, c_noise = edm_precond(sigma)
        out = blocks[i](z=z*c_in, cond=c_noise, kv_cache=self._kv[i])
        denoised = out * c_out + z * c_skip
        d = (z - denoised) / sigma
        dt = self.sigmas[i+1] - sigma
        z = z + d * dt

    # 3. Project final z to byte logits, commit argmax byte.
    #    Multi-byte UTF-8 continuation bytes (>127) decode to "" via
    #    errors="ignore", which the runner treats as an abstain.
    logits = z @ self.E_out.T
    byte_idx = int(logits.argmax().item())
    return bytes([byte_idx]).decode("utf-8", errors="ignore")

@torch.no_grad()
def observe(self, ch: str) -> None:
    for byte in ch.encode("utf-8"):
        x_emb = self.embed(torch.tensor([[byte]], device=device))
        # Each block independently extends its OWN KV-cache from the same
        # shared token embedding. No chaining: block b's input is x_emb,
        # not the output of block b-1. (This matches the training graph,
        # where block b also never sees a lower-block forward.)
        for b in range(B):
            self._kv[b] = blocks[b].extend_kv(
                x_emb, self._kv[b], offset=self._pos)
        self._pos += 1
```

**Inference cost per byte:** ~2L layer-forwards — `observe()` runs B blocks × L/B layers each (= L) to extend all caches, then `predict()` runs B Euler steps × L/B layers each (= L). That's ~2× a standard L-layer AR transformer per byte, matching the paper's Appendix A.2 footnote about autoregressive inference overhead. At L=6 / B=3 on A100, the absolute cost is small (each step is only 2 layers of d=384), but the 2× constant factor is the relevant per-char eval throughput delta vs modded_nanogpt — call it out in `report.json`.

### Hyperparameters (starting point)

The Sakana paper does **not** publish AR hyperparameters. EDM defaults plus their CIFAR-100 recipe imply:

| Knob | Value | Source |
|---|---|---|
| `σ_min, σ_max` | 0.002, 80.0 | EDM, `dblock_modules.py:8` |
| `P_mean, P_std` | -1.2, 1.2 | EDM, `dblock_modules.py:10` |
| `σ_data` | calibrate (see Phase 0); fallback 0.5 | `model.py:114` |
| `γ` (block-overlap) | 0.10 | paper §3.3 + §4.3 ablation: γ=0.10 is the FID-optimal point on CIFAR-10 (41.39 vs 42.98 at 0.05, 42.84 at 0.15) |
| `λ_ce` | 1.0 | starting weight for the L2+CE hybrid (see "deliberate divergence" note above); sweep if Phase 1 underperforms |
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

2. **Step-count check.** Profile one DiffusionBlocks training step (one active block's forward + backward — paper §3.1, Appendix C Algorithm 1) vs one modded_nanogpt step on CPU at small scale. Per-step layer-updates are `L/B` vs E2E's `L` (paper §4.3 table, "Layers/Step" column), so the per-step time ratio should be `~1/B` — for B=3 that's ~0.33×. Combined with `B × N_baseline` steps (paper Appendix D.1 "Fair comparison" matches total layer updates between DB and E2E), total wall-clock ≈ 1× modded_nanogpt. **If profiling shows a per-step ratio meaningfully above 1/B, an extra block forward is sneaking in — find and remove before Modal dispatch.**

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

The energy number that comes out is itself the most interesting deliverable, because there is no published comparison: total training FLOPs equal E2E (per-step cost is `1/B` of E2E's, with `B × N_baseline` iterations to match total layer updates — paper Appendix D.1 "Fair comparison") but the activation-memory headroom may enable wider models inside the 300 s budget that E2E can't fit. Phase 2's width sweep is the natural follow-up to test that.

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
- Paper markdown (local): `research/catalog/new_directions/block-diffusion/paper.md` — read this before the arxiv PDF; section refs in this spec point at it
- Reference impl (upstream): https://github.com/SakanaAI/DiffusionBlocks (CIFAR-100 ViT only; AR not released)
- Reference impl (local clone): `/tmp/DiffusionBlocks` — `dblock_modules.py` (noise scheduling, equi-prob partitioning), `model.py:110-291` (training/inference loop with EDM preconditioning). Note: the line-number citations elsewhere in this spec are against this clone; verify before relying on them
- EDM reference: Karras et al. 2022, [arxiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Baseline to modify: `submissions/modded_nanogpt/`
- Related catalog entries: `spec_10_forward_forward_bytes.md` (block-wise, sub-floor), `research/forward-forward-deep/` (multi-phase FF investigation)
- Harness: 300 s, A100-80GB, NVML+CPU joules, val char-acc ≥ 0.70 on 60K val chars
