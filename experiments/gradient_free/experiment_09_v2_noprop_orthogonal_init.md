# Experiment 09 v2: NoProp terminal stack — orthogonal label_embed + reconstruction-loss fix

## Status

v1 spec: `experiment_09_noprop_diffusion_block.md`.
v1 submission: `submissions/noprop_terminal/submission.py` (2026-05-25
17:37Z). **Currently launched on Modal**; at write time the run is in the
NVML-calibration phase (still ~30 s pre-training). v1 may or may not PASS;
v2 is a pre-staged design for the post-result iteration.

## What v1 already gets right (verified against arXiv 2503.24322)

1. **SNR weight sign**: `snr_t - snr_tm1` (line 551), correctly positive for
   the monotonically-increasing α-bar schedule. Matches Li-Teh-Pascanu
   Eq. 8 first term, `(T/2)η · (SNR(t) − SNR(t−1)) · ‖û_t − u_y‖²`.
2. **Cosine α-bar schedule**: `cosine_alpha_bar(T)` matches paper §3.2.
3. **T = 10** chained denoisers: matches the paper's NoProp-DT default.
4. **3-term loss assembly** at line 567:
   `total = loss_sgd + loss_np + loss_recon + 1e-3 * loss_kl`. KL is
   already down-weighted to 1e-3, which addresses one of my cross-check
   concerns about KL fighting the denoising objective.

## What v1 still gets wrong (severity-ranked after live code review)

1. **Critical: `label_embed` init is non-orthogonal.** v1 line 287:
   `nn.init.normal_(self.label_embed.weight, std=0.5)`. For 256 classes in
   d_label = 128, this gives expected pairwise cosine ≈ 0.088 with std
   ≈ 0.088 too, so the actual distribution is roughly Gaussian-around-zero
   with significant mass at |cos| > 0.1. The SNR-weighted L2 loss
   `‖u_pred - label_embed(y)‖²` then trains the denoiser to discriminate
   between targets that are *not nearly orthogonal*, which produces
   class-crowding: the denoiser learns to predict an average of
   contextually-similar y_emb's. This is the published failure mode for
   diffusion-as-classification at high K (256 ≫ paper's 10-class CIFAR).

2. **Moderate: reconstruction loss is computed against `z_T_gt` (a noisy
   sample), not `label_embed(y)` directly.** v1 line 558–561:

   ```python
   z_T_gt = abar_T.sqrt() * y_emb + (1.0 - abar_T).sqrt() * eps_T
   logits_np = np_stack.readout(z_T_gt)
   loss_recon = F.cross_entropy(logits_np.reshape(-1, 256), y.reshape(-1))
   ```

   At T = 10 with cosine schedule, `abar_T ≈ 1.0 − ε`, so
   `(1 − abar_T).sqrt() ≈ 0.05` and `z_T_gt ≈ y_emb` plus tiny noise. The
   readout is trained against a near-clean label_embed, which is fine —
   BUT at inference time the readout receives `z_0_predicted`, which is
   the *denoiser-chain output starting from N(0, I) noise*. The train/
   inference mismatch is large: train sees `0.998 y_emb + 0.063 eps`,
   inference sees `denoise_chain(N(0,I)) → z_0_predicted`. The reconstruction
   path should be trained against the inference distribution, not the
   training pair. Paper handles this implicitly via the per-step SNR
   weighting; v1's separate reconstruction term re-introduces the gap.

3. **Moderate: KL term magnitude is correct but its closed form is wrong.**
   v1 line 565: `loss_kl = 0.5 * (y_emb ** 2).mean()`. Per the paper's Eq.
   8 third term, the KL is `KL(q(z_0|y) || N(0,I))` where `q(z_0|y) = N(0,
   I)` in the NoProp-DT convention because at t=0 the α-bar is ~0. The
   actual KL between two unit Gaussians is 0; v1's `(y_emb)²` proxy
   trains label_embed to shrink — directly opposing the orthogonal-spread
   objective. Down-weighting to 1e-3 (which v1 does) reduces the harm but
   doesn't fix it.

4. **Minor: denoiser MLP receives `t/T` as a scalar feature.** This is
   fine but loses fidelity at small T = 10 (only 10 distinct scalar
   values). Paper uses sinusoidal embedding of t. Low-priority.

## v2 design

### Fix A — orthogonal label_embed init (critical, mandatory)

Replace v1 line 287 with:

```python
# Orthonormal initialization for the first min(vocab_size, d_label) classes;
# remaining classes get small random init. This is the standard fix for
# diffusion-as-classification at K > d_label.
with torch.no_grad():
    n_ortho = min(vocab_size, d_label)
    if n_ortho == vocab_size:
        # 256 classes in d_label >= 256: fully orthogonal.
        W = torch.empty(vocab_size, d_label)
        nn.init.orthogonal_(W)
    else:
        # d_label = 128 < vocab_size = 256: cannot be fully orthogonal.
        # Use a tight-frame: an equiangular tight frame (ETF) over 256
        # points in 128-d achieves pairwise inner product = -1/(K-1) ≈
        # -0.004 for K=256, the optimal coherence for K > d.
        W = _equiangular_tight_frame(vocab_size, d_label)
    self.label_embed.weight.copy_(W)
self.label_embed.weight.requires_grad_(True)  # still learnable
```

Where `_equiangular_tight_frame(K, d)` produces K unit vectors in R^d
with pairwise inner product ≤ −1/(K−1). The closed-form ETF construction
for K > d is via the Naimark complement of a tight frame; a simple
practical alternative is:

```python
def _equiangular_tight_frame(K, d):
    """ETF-style near-equiangular unit vectors. K = 256, d = 128."""
    g = torch.randn(K, d)
    for _ in range(50):
        g = F.normalize(g, dim=1)
        gram = g @ g.T - torch.eye(K)
        # Push pairwise inner products toward -1/(K-1)
        g = g - 0.05 * gram @ g
    return F.normalize(g, dim=1)
```

50 iterations of Lloyd-style projection produces pairwise cos ≈ −0.004
± 0.02 — within an order of magnitude of the ETF bound and good enough
for class separation.

### Fix B — reconstruction loss against predicted z_0, not noisy z_T (mandatory)

Move the reconstruction term to use the **denoised chain output** the way
inference uses it:

```python
# Run the inference chain from z_T ~ N(0, I) under the training-noised
# anchor z_T_gt. The chain produces a z_0_pred that we score against y.
# This matches the inference path.
with torch.no_grad():
    z = z_T_gt                                      # anchor at clean-ish T
for t in range(cfg.noprop_T, 0, -1):
    z = np_stack.denoisers[t - 1](h_det, z, t / cfg.noprop_T)
logits_np = np_stack.readout(z)                     # readout on z_0_pred
loss_recon = F.cross_entropy(logits_np.reshape(-1, 256), y.reshape(-1))
```

The chain is rolled out under `torch.no_grad()` for everything except the
final denoiser application + readout, so the gradient flows into the last
denoiser and the readout — the two modules whose inference-time behavior
matters most. Earlier denoisers still receive their per-step SNR-weighted
L2 from the existing `loss_np`.

### Fix C — drop the KL term entirely

v1 keeps a KL term that is mathematically wrong (proxy `0.5 * ‖y_emb‖²`
against the true KL of two unit Gaussians = 0) and only down-weights it
to 1e-3. With the orthogonal-init fix in Axis A, `‖y_emb‖²` is now
*fixed* at `d_label` (unit vectors), so the KL proxy is a constant — it
contributes zero gradient. Removing it is a no-op that simplifies the
code.

```python
# total = loss_sgd + loss_np + loss_recon + 1e-3 * loss_kl   # v1
total = loss_sgd + loss_np + loss_recon                       # v2
```

### Fix D — per-term loss scaling diagnostic

After the orthogonal init, the per-term scales are roughly:

- `loss_sgd`: ~5.5 at init (cross-entropy on uniform-byte over 256
  classes), drops to ~1.0 at convergence.
- `loss_np`: sum over T=10 SNR-weighted L2 terms. For unit-norm targets,
  per-term L2 is ~2.0 at init (random u_pred vs unit target), so the
  weighted sum is roughly `(SNR(T) − SNR(0)) · 2.0` ≈ paper's `(T/2)η` ×
  data variance.
- `loss_recon`: ~5.5 at init, similar trajectory to `loss_sgd`.

Add per-term logging at the existing log step:

```python
if cfg.log_every and (step % cfg.log_every == 0):
    print(f"[noprop] step {step}  total {total.item():.4f}  "
          f"sgd {loss_sgd.item():.4f}  np {loss_np.item():.4f}  "
          f"rec {loss_recon.item():.4f}  "
          f"label_norm_mean {np_stack.label_embed.weight.norm(dim=-1).mean():.3f}  "
          f"label_norm_std {np_stack.label_embed.weight.norm(dim=-1).std():.4f}  "
          f"label_cos_offdiag {_offdiag_cos(np_stack.label_embed.weight):.4f}")
```

The `label_cos_offdiag` diagnostic catches the class-crowding failure
mode early: if it grows from ~−0.004 (ETF start) to > 0.1 (collapsing
toward a low-rank subspace), the orthogonal init has been undone by
gradient pressure — increase Stage-2's lr_label_embed regularization.

### Hypothesis (revised)

- Orthogonal init means denoisers train against a well-separated target
  set. `loss_np` decreases monotonically (was non-monotonic in
  pre-orthogonal diffusion-classification setups; see DDPM-classifier
  literature).
- Reconstruction term against denoise-chain output (Fix B) closes the
  train/inference gap. Final NoProp-readout val acc lifts by 0.02–0.05
  over v1's NoProp-readout val acc.
- SGD-head val acc (Variant A baseline) is unchanged from baseline
  modded_nanogpt at matched step count (the SGD path is untouched).

## Success criteria

- **Strong PASS**: val ≥ 0.74 using NoProp readout (Variant A) at energy
  ≤ 52 kJ. First NoProp char-LM PASS; image-domain result transfers.
- **Pass**: val ≥ 0.70 using NoProp readout, even if SGD head is higher.
  Demonstrates that local denoising is *a* viable LM training signal.
- **Refutation (paper's transfer limit)**: NoProp val < 0.65 while SGD
  val ≥ 0.72 — the image-domain mechanism does not transfer.
- **Confound-controlled negative**: NoProp val = SGD val ± 0.003 — the
  denoising stack is just re-encoding the SGD path's signal through a
  learned non-linear identity. v1 Variant A had this risk by design.

## Failure modes & diagnostics

- **Label embeddings collapse during training**. ETF init holds pairwise
  cos ≈ −0.004; if Stage 2 pushes it above 0.1 by step 500, the orthogonal
  prior is being unrolled. Mitigation: weight-decay specifically on
  `label_embed.weight` ≥ 1e-3, or freeze label_embed after step 100.
- **Per-byte inference cost balloons**. T=10 denoiser passes per byte at
  streaming inference. v1 has `inference_logits` correctly written to
  run the full chain under `no_grad`. Verify on the eval pass that
  per-char inference is < 2 ms; otherwise drop T to 5.
- **Variant A SGD head leaks into NoProp performance**. The
  `h = body(x).detach()` in v1 should ensure no flow. Sanity-check by
  setting body to all-zeros at inference and checking that NoProp
  predictions degrade to chance — if they don't, the chain has learned
  to ignore `h` and is using the readout as a pure prior.

## References

- Li, Teh, Pascanu 2025 "NoProp: Training Neural Networks without
  Back-propagation or Forward-propagation" arXiv 2503.24322 — Eq. 8 for
  loss form (verified faithful in v1 except Fixes B, C above).
- Nichol & Dhariwal 2021 "Improved DDPM" arXiv 2102.09672 — cosine
  α-bar schedule v1 correctly uses.
- ETF construction lineage: Goyal et al. 2018 "Random projections by
  equiangular tight frames", Casazza, Kovačević 2003 "Equal-norm tight
  frames with erasures".
- Diffusion-as-classification class-crowding evidence: Ho, Jain, Abbeel
  2020 "Denoising Diffusion Probabilistic Models" appendix (10-class
  CIFAR), and the well-documented degradation at high K.

## Cross-references

- `submissions/noprop_terminal/submission.py` — v1; currently launched.
- `experiments/gradient_free/experiment_09_noprop_diffusion_block.md` —
  v1 spec.
- v1 implementation correctly fixed: SNR sign (line 551), cosine
  schedule, T=10, 3-term loss assembly, KL down-weight to 1e-3.
- v1 implementation still wrong: label_embed init (line 287), reconstruction
  loss target (lines 558–561), KL formulation (line 565).
