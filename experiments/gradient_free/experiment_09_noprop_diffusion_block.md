# Experiment 09: NoProp-Style Local-Denoising Block as a Last-Layer Substitute

## Hypothesis
A NoProp-style stack (Li, Teh, Pascanu 2025, arXiv 2503.24322) — T denoising blocks each trained by independent local denoising of the *target label embedding* with NO global backprop and NO global forward chain — can substitute for the *output head + final transformer block* of the modded_nanogpt stack, with the rest of the body still SGD-trained. The combined model reaches ≥ 0.70 val acc, demonstrating that NoProp-style local training has a viable LM hook.

## Motivation
NoProp (Li, Teh, Pascanu, arXiv 2503.24322, March 2025 — **note: not a Hinton paper**, authored at Oxford / Mila) is the newest credible gradient-free training paradigm with a representation-learning story. Each block independently learns to denoise a noisy target label embedding via diffusion-style updates; the noise schedule is variance-preserving Ornstein-Uhlenbeck (NOT flow-matching), with `q(z_t|y) = N(sqrt(ᾱ_t) u_y, 1-ᾱ_t I)`, `ᾱ_t = ∏ α_s`. To our knowledge nobody has tried it for next-byte prediction.

For a 256-class char output the "label embedding" is well-defined: a learned 256-vector lookup. The denoising objective is supervised but local — no error signal flows across blocks. This is genuinely orthogonal to every method we've measured.

We are NOT testing NoProp as a full LM. We test it where it has the best chance: as a small denoising stack on top of an SGD-trained body. Critically, **we keep T=10 denoising steps** (matching the paper's discrete-time setting). A prior version of this design collapsed to T=1 for budget reasons; that breaks the NoProp mechanism (the chained denoising is the headline contribution) and would refute single-step denoising, not NoProp. Per-block compute is ~1% of a transformer block, so T=10 is well within budget.

## Method
Standard 5-layer modded_nanogpt body, but the 6th block + output head are replaced with a `NoPropTerminalStack` of T=10 denoising sub-blocks plus a final readout:

```python
class NoPropTerminalStack(nn.Module):
    """T=10 chained denoising sub-blocks trained independently (no cross-step grad).

    Holds:
      label_embed:  nn.Embedding(256, d_label)         — learned, local update
      denoisers:    [denoiser_1, ..., denoiser_T] each a small MLP
                    (input: hidden state h, prev latent z_{t-1}, scalar t/T) -> d_label
      readout:      Linear(d_label, 256)               — terminal classifier head

    Forward-noise (variance-preserving, Eq. 6 of Li-Teh-Pascanu):
        q(z_t|y) = N( sqrt(alpha_bar_t) * label_embed(y), (1 - alpha_bar_t) * I )
        alpha_bar_t = product_{s<=t} alpha_s,  cosine schedule on alpha_s.

    Training step:
      h = body(x).detach()                # body's representation, NO grad into body
      for t in 1..T:
        sample z_{t-1} ~ q(z_{t-1} | y)   # ground-truth noisy latent at t-1
        u_pred = denoisers[t-1](h, z_{t-1}, t/T)
        loss_t = (SNR(t) - SNR(t-1)) * ||u_pred - label_embed(y)||^2
      logits = readout(z_T_predicted)     # using ground-truth z_T at training
      loss_recon = cross_entropy(logits, y)
      loss_kl = KL( q(z_0|y) || N(0, I) )       # small, analytic
      total_loss = sum_t loss_t + loss_recon + loss_kl
      total_loss.backward()               # updates denoisers, readout, label_embed only

    The losses update denoisers / readout / label_embed; h is detached so no flow
    into the body. NO inter-block gradient flow: each denoiser_t only sees its
    own loss_t.

    Inference: run the chain z_T -> z_{T-1} -> ... -> z_0 deterministically,
    starting from z_T ~ N(0, I); classify via argmax of readout(z_0).
    """
```

Three notable corrections relative to a naive "NoProp = label denoising" reading:
1. **T=10 denoising steps**, not T=1 — the chained denoising IS the contribution.
2. **The loss has three terms**, matching the paper's Eq. 8: per-step SNR-weighted L2 + reconstruction (cross-entropy via readout) + KL(z_0 prior). Dropping the reconstruction term removes the only force making z_0 class-discriminable.
3. **Noise schedule is VP-OU with cosine α_s**, matching the paper, not the linear `z_t = sqrt(1-t)y + sqrt(t)ε` schedule (which is a degenerate special case).

The SGD-trained body learns h by being supervised through a *separate* simple linear classification head during training (so the body has a learning signal). The NoProp stack runs in parallel and at inference time we use ONLY the NoProp readout. This isolates "does NoProp produce a usable classifier."

Variant B: SGD body has NO classification head (no global supervision); the body is just an autoencoder pretrained on a reconstruction loss. NoProp stack provides the only classification signal. (Stronger test, lower-acc expected.)

## Memory-Movement Analysis
- label_embed: 256 × d_label (d_label=128) = 32 KB. Negligible.
- Each denoiser: small (≤256, 256, 128) MLP; ~100 KB params. Cheap. T=10 of them → ~1 MB total.
- Per training step: T=10 denoiser forwards + 1 readout + local optimizer step over all denoisers/readout/label_embed. Total compute ≈ 10% of a transformer block — still cheap relative to the 5-layer body it supplements.
- The NoProp stack runs *alongside* the SGD head — both are trained per step. No cross-flow into the body (h is detached).

## Setup
- 5-layer modded_nanogpt body + (parallel: linear classification head with full backprop) + (parallel: NoProp T=10 denoising terminal stack with readout).
- Inference uses NoProp readout only (Variant A) or NoProp readout only with autoencoded body (Variant B).
- Baseline: `modded_nanogpt` (full 6L + classification head, 51.7 kJ / 0.7374).

## Procedure
1. `cp -r submissions/modded_nanogpt submissions/noprop_terminal`
2. Reduce body to 5 layers.
3. Add `NoPropTerminalStack` (T=10 denoisers + label_embed (size 256 × 128) + linear readout).
4. Modify training step:
```python
logits_sgd, _ = model(x)        # body + classical head, normal CE
loss_sgd = F.cross_entropy(logits_sgd.flatten(0, 1), y.flatten())

h = model.body_hidden(x).detach()        # body output, detached
y_emb = label_embed(y)                   # (B, T_seq, d_label)
loss_np = 0.0
# precompute cosine-schedule alpha_bars
alpha_bars = cosine_alpha_bar_schedule(T=10)  # tensor of shape (T+1,)
for t in range(1, T + 1):
    abar_t = alpha_bars[t]
    abar_tm1 = alpha_bars[t - 1]
    eps = torch.randn_like(y_emb)
    z_tm1 = abar_tm1.sqrt() * y_emb + (1 - abar_tm1).sqrt() * eps
    u_pred = denoisers[t-1](torch.cat([h, z_tm1, torch.full_like(h[..., :1], t/T)], -1))
    snr_diff = (abar_t / (1 - abar_t)) - (abar_tm1 / (1 - abar_tm1))
    loss_np = loss_np + snr_diff * ((u_pred - y_emb) ** 2).mean()

# reconstruction term: train readout against ground-truth z_T
z_T_gt = alpha_bars[T].sqrt() * y_emb + (1 - alpha_bars[T]).sqrt() * torch.randn_like(y_emb)
logits_np = readout(z_T_gt)
loss_recon = F.cross_entropy(logits_np.flatten(0, 1), y.flatten())

# KL term: small / analytic — q(z_0|y) is close to delta on label_embed(y)
loss_kl = 0.5 * (y_emb ** 2).mean()  # vs N(0, I) prior

total = loss_sgd + loss_np + loss_recon + loss_kl
total.backward()
```
   These NoProp losses update `label_embed.weight`, `denoisers.parameters()`, and `readout.parameters()` only; `h` is detached so the body sees no NoProp signal.
5. At inference: start from z_T ~ N(0, I); for t = T..1 apply the reverse denoising step using `denoisers[t-1](h, z_t, t/T)`; classify via `argmax(softmax(readout(z_0)))` for the streaming API.
6. Submit Variant A first.

## Success Criteria
- **Strong**: val ≥ 0.72 using the NoProp output (Variant A) → NoProp-style local denoising is a viable LM training signal.
- **Pass**: val ≥ 0.70 → first known NoProp result on char-LM.
- **Refutation (expected)**: val < 0.65 → confirms NoProp's image-domain win doesn't transfer to next-token LM at this scale. Adds new dead-list entry.
- **Diagnostic value (regardless)**: gap between SGD-head val and NoProp-head val measures how much representation learning the body did vs. how much the heads contribute.

## Failure Modes & Diagnostics
- Denoiser collapses to constant: log mean cosine similarity between predicted z_0 and true label_embed(y) — if < 0.1 by step 500, denoiser is broken.
- label_embed collapses: log row-norms of label_embed; if 80% of rows have ‖row‖ < 0.1, classes are degenerate.
- Variant A: SGD head provides a free baseline (≈ baseline acc); confirm NoProp head doesn't just regress to the SGD head's output via the shared body.

## Estimated Cost
1 Modal run, ~10 min, ~$0.40. Add a Variant B run only if Variant A passes ≥ 0.70.

## References
- Li, Teh, Pascanu 2025 "NoProp: Training Neural Networks without Back-propagation or Forward-propagation" (arXiv 2503.24322, posted March 2025; Oxford / Mila — *not* a Hinton paper). Eq. 6 = forward noise schedule, Eq. 8 = full training objective implemented above.
- Ho, Jain, Abbeel 2020 "Denoising Diffusion Probabilistic Models" (NeurIPS) — origin of the SNR-weighted L2 denoising objective NoProp adapts.
- Nichol & Dhariwal 2021 "Improved DDPM" — cosine α_s schedule used here.
