# Experiment 07 Result — Two-phase Nystrom KRR on Learned Embedding Features

## Hypothesis recap
Train a small transformer encoder for ~120s, then replace its LM head with a closed-form Nystrom kernel-ridge readout (cosine kernel, M=1024 landmarks, N=100K sampled context windows) — deep-kernel hybrid should reach val_acc >= 0.65 at < 45 kJ.

## Final numbers
| metric | value |
|---|---|
| val_char_accuracy | **0.2999** |
| training_energy_J | **32,396 J** (32.4 kJ) |
| training_duration_s | **139.6 s** (well under the 300 s cap) |
| DQ status | **DISQUALIFIED** — val_accuracy_below_floor (floor 0.7000) |
| Success bracket | **Refuted** (val < 0.60) |

## Phase 1 vs phase 2 wall split (from run.log)
- Phase 1 (encoder + temp LM head, AdamW, 3164 steps): **122.0 s**
- Phase 2 (feature collection N=100K + Nystrom KRR solve): **16.0 s**
  - Feature collection: 15.8 s
  - KRR solve itself: 0.21 s
- Total train wall: **138.0 s** (NVML-measured 139.6 s)

## Interpretation
The phase-1 encoder converged normally — cross-entropy fell from 5.55 to ~1.10 over 3164 steps, on par with what a 3.3M-param transformer should achieve in 2 min. A traditional softmax LM head on this encoder would likely score around 0.55–0.60 val char-acc (extrapolating from modded-nanogpt's loss/acc curve at similar capacity). Instead, replacing it with a cosine-kernel Nystrom KRR head dropped accuracy to **0.30** — roughly unigram-floor territory.

Two structural failures of the cosine-kernel-on-residual readout are likely:
1. **Cosine kernel discards magnitude.** L2-normalizing the residual stream throws out the per-direction scale that the LM head exploits. The Cholesky failure ("matrix not PSD at leading minor 307") is consistent with the normalized features lying on a near-rank-deficient subspace of S^255 — the LU fallback fired and the alpha that came out is the least-squares projection onto that subspace.
2. **KRR with one-hot targets is a quadratic-loss classifier.** On highly stochastic next-byte targets (conditional entropy ~3 bits/byte), the squared-error optimum is the conditional mean — a smoothed distribution. argmax of a smoothed conditional mean is dominated by the marginal mode (space character) for most contexts, which matches the ~0.30 accuracy floor.

Deep-kernel-learning at this scale **does not pay off** vs. keeping the trained NN head. The phase-1 representation is fine; the readout step is what wrecks it. To rescue the approach you would need (a) un-normalized features with an RBF or arc-cosine kernel, (b) cross-entropy-shaped targets (e.g. log-marginals or temperature-sharpened soft targets), or (c) a much larger M to capture the per-context distribution shape. None of those are tweaks — they are different methods.

## Implementation deviations from the spec
1. **AdamW only for phase 1** (no Muon). Spec says "AdamW + cross-entropy"; modded-nanogpt uses Muon for 2D block weights but the spec did not require it, and AdamW kept the time-budgeted loop simple.
2. **Cholesky fell back to torch.linalg.solve** on the actual run — the A matrix lost PSD at leading minor 307 even with the 1e-6 jitter. Caught by the try/except in `nystrom_krr_fit` per spec's failure-mode guidance (jitter to 1e-4 + LU solve). This did not save the accuracy.
3. **No ablations run** (uniform landmarks, one-hot Y, cosine kernel only). The base configuration scored 0.30, so the per-spec ablation knobs (k-means++ landmarks, label smoothing, arc-cosine order-2 kernel) would not plausibly close a 0.40-point gap and a re-submission was not authorised after one DQ.
4. **Phase-1 wall came in at 122.0 s** (target 120 s). The while-loop checks elapsed at the top of each step so the last step started at 119.5 s and finished at 122.0 s. Inside budget.

## Artifacts
- Submission: `/home/seneca/wikitext/submissions/nystrom_krr_hybrid/submission.py`
- Result: `/home/seneca/wikitext/submissions/nystrom_krr_hybrid/result.json`
- Run log: `/home/seneca/wikitext/submissions/nystrom_krr_hybrid/run.log`
- NVML record: `/home/seneca/wikitext/submissions/nystrom_krr_hybrid/nvml.json`
- Modal run: https://modal.com/apps/ab-10/main/ap-O5G29lgKVogOeKzKYNK9iI

## Review (post-hoc audit)

**Validity for discarding cosine-Nyström-KRR-on-learned-encoder:** *Valid for the specific hybrid; entangled failure modes.*

**Core limitations:**
- **Train/val metric mismatch.** Phase-1 train loss (1.10) is the loss of the *temporary* CE head, not the cosine-Nyström readout that scored val. The writeup correctly notes this; the experiment cleanly does not report what the phase-1 head would have achieved on val, which would have set a clean encoder-quality ceiling for diagnosing whether the 0.40-point drop is readout-mediated or representation-mediated.
- **Two failure modes are confounded.** The interpretation identifies both (a) cosine kernel discarding magnitude and (b) one-hot-MSE giving the marginal-mode argmax — both plausible, but the experiment doesn't isolate which dominates. An RBF-kernel variant on the same encoder would separate (a) from (b); a soft-target (label-smoothed or temperature-sharpened) variant would separate (b) from (a).
- **Cholesky PSD failure caught but not surfaced.** The LU fallback fired at jitter 1e-4; this is logged but the resulting alpha is the least-squares projection onto a ~rank-307 subspace of S^255. Whether the val number reflects the method or the rank-deficient fallback is unresolved.

**Verdict:** Sufficient to discard *this specific hybrid* (CE-trained encoder + L2-normalized features + cosine kernel + one-hot MSE). Does not discard deep-kernel-learning in general — the writeup's own "would need RBF / soft targets / larger M" list names three live alternatives.
