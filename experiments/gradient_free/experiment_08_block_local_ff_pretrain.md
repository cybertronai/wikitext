# Experiment 08: Block-Local Forward-Forward Pretraining + SGD Fine-Tune Head

## Hypothesis
The prior causal-FF submission (`forward-forward-causal`, ceiling 0.279) failed because FF was asked to be the entire LM. If FF is instead used **only to pretrain block weights via the goodness objective for ~80 s**, then the linear head and the remaining ~200 s are SGD-trained on cross-entropy, the resulting model clears 0.70 val acc at energy comparable to the baseline. Tests the "FF as local pretrainer" hybrid pattern the prior survey explicitly recommended ("Worth further iteration only as a local-learning backbone + closed-form readout hybrid").

## Motivation
The prior gradfree-survey report ends its FF section with: *"Worth further iteration only as a 'local-learning backbone + closed-form readout' hybrid pattern."* This experiment is precisely that — except instead of a closed-form readout (which the survey already showed plateaus from a frozen backbone), we SGD-train the head and let it backprop into block params *as a fine-tuning phase after FF pretrain*. The hypothesis tests whether FF gives a useful initialization, not just whether FF is a complete LM (which is empirically refuted).

## Method
Two-stage training, single submission:

**Stage 1 (FF, 80 s):** For each block in the 6-layer net, run Hinton's goodness objective:
- Positive samples: real (context, true-next-byte) pairs encoded → block output. Goodness = sum-of-squared-activations per token.
- Negative samples: (context, *wrong*-next-byte) pairs. **Design choice** (not Hinton's): corrupt by replacing the last byte of context with a random different byte. Hinton's original FF paper used a "negative-data network" to generate hard negatives for images and did not prescribe a sequence recipe; this corruption strategy is the experimenter's choice, pre-registered here as such.
- Loss per block: `softplus(-(g_pos - θ)) + softplus(g_neg - θ)` with θ = d (the model dim — matches Hinton's reference-implementation convention of θ = N where N is the number of neurons in the layer).
- Update ONLY that block's parameters with one local AdamW step.
- Each block is trained independently (in parallel, no cross-block gradient flow).
- Run ~300 FF steps within 80 s budget.
- **Sanity gate before handing off to Stage 2**: after Stage 1 completes, compute per-block mean activation std on a small held-out batch. If any block's std is > 5× the standard init-time std for that depth, the FF init has degenerated to "make activations big" and Stage 2 will start out of distribution. Either (a) re-normalize per-block outputs to match init std before Stage 2, or (b) abort and submit baseline. This is the first-class failure check, not just a diagnostic.

**Stage 2 (CE, 200 s):** Standard cross-entropy training of the full stack with Muon + AdamW, exactly as in `modded_nanogpt`. The blocks start from the FF-pretrained init.

## Memory-Movement Analysis
- Stage 1: per FF step, each block fwd pass + local backward over only its own weights. ~6× per-block fwd cost vs. one full fwd-pass, but no cross-block backward. Roughly similar per-step compute to normal SGD.
- Stage 2: identical to baseline.
- **The energy question is whether Stage 1's 80 s "buys" a better init worth ~30% of the normal training run.** This is testable directly.

## Setup
- Architecture: 6-layer modded_nanogpt (same as baseline).
- Stage-1 budget: 80 s wall-clock OR 300 steps, whichever first.
- Stage-2 budget: 220 s wall-clock (less than baseline's full 300 s, to give Stage 1 its 80 s and stay within the 300 s cap).
- Negative-sample strategy: corrupt the last byte of each context window to a uniformly random different byte.

## Procedure
1. `cp -r submissions/modded_nanogpt submissions/ff_pretrain_then_sgd`
2. Refactor training loop to two stages:
```python
# Stage 1: FF pretrain
optimizer_ff = [AdamW(b.parameters(), lr=1e-3) for b in model.blocks]
t_ff_start = time.monotonic()
for step in range(300):
    if time.monotonic() - t_ff_start > 80: break
    x_pos, x_neg = sample_pos_neg(train_bytes, B, T)
    # forward sequentially; detach between blocks to enforce locality
    h_pos = model.embed(x_pos)
    h_neg = model.embed(x_neg)
    for i, block in enumerate(model.blocks):
        h_pos_in = h_pos.detach().requires_grad_(True)
        h_neg_in = h_neg.detach().requires_grad_(True)
        h_pos_out, _ = block(h_pos_in)
        h_neg_out, _ = block(h_neg_in)
        g_pos = (h_pos_out ** 2).mean(-1).mean()
        g_neg = (h_neg_out ** 2).mean(-1).mean()
        theta = float(model_dim)
        loss = F.softplus(-(g_pos - theta)) + F.softplus(g_neg - theta)
        optimizer_ff[i].zero_grad()
        loss.backward()
        optimizer_ff[i].step()
        # pass clean (non-detached) forward to next block for next iter
        h_pos = h_pos_out.detach()
        h_neg = h_neg_out.detach()

# Stage 2: standard SGD CE, capped to time remaining
... (existing modded_nanogpt loop, but n_steps reduced proportionally)
```
3. The head and embed are not FF-trained — only block params. Head is initialized as in baseline.
4. Submit.

## Success Criteria
- **Strong**: val > 0.74, energy ≤ 50 kJ → FF gave a meaningfully better init than random.
- **Pass**: val ≥ 0.70, energy ≤ 51 kJ → matches baseline, confirms FF doesn't hurt.
- **Refutation**: val < baseline by ≥0.005 → FF pretrain is wasted compute relative to longer SGD.

## Failure Modes & Diagnostics
- FF goodness blow-up (the documented FF degenerate mode): handled by the Stage-1 sanity gate above. Log mean goodness AND mean activation std per block across stage 1.
- Negative sampling too easy: random-byte corruption of just the last position may be too obvious. Try corrupting positions 256..end (a longer suffix).
- Stage 2 just re-trains everything anyway: the meaningful diagnostic is val acc at *step 500* of Stage 2 vs. step 500 of baseline modded_nanogpt — the FF-initialized run should converge faster. Log val on a 5000-byte held-out probe every 200 Stage-2 steps.

## Estimated Cost
1 Modal run, ~10 min, ~$0.40.

## References
- Hinton 2022 "The Forward-Forward Algorithm: Some Preliminary Investigations" (arXiv 2212.13345) — goodness, threshold = N convention, softplus loss form.
- `research/gradfree-survey/REPORT.md` — prior FF failure as full LM (ceiling 0.279).
- `research/gradfree-survey/designs/method_forward-forward-causal_pass_2.md` — prior design.
