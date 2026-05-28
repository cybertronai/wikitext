# Spec 11 — Direct Feedback Alignment — Results

**Status:** DQ (val accuracy below 0.70 floor).
**Submission:** `submissions/dfa_v1` (block-DFA on a 6-layer 384-d byte-level transformer).
**Run:** 2026-05-25, A100-80GB SXM4, 4000 SGD-momentum steps in 174s training.

## Numbers

| Metric | This run | Baseline (modded_nanogpt) | Spec prediction |
|---|---|---|---|
| Val char-acc | **0.0000** | 0.7374 | 0.65–0.73 |
| Training energy | 40,783 J | 51,704 J | ~35,000 J |
| Training duration | 178 s | 247 s | — |
| Training steps | 4000 | 2150 | — |
| Train-loss floor | 5.42 (uniform = 5.545) | — | — |

The 178 s training cost 40.8 kJ — **~21% under the baseline's 51.7 kJ** despite running
*more* steps, consistent with the spec's "~30% FLOPs reduction from skipping cross-block
backward chains" prediction. So the energy story held. The model just didn't learn.

## What happened

Loss dropped from 5.74 → 5.42 in the first 100 steps and then **plateaued** for the
remaining 3900 steps. 5.42 = `log(256) − ε`, i.e. the model collapsed to the unigram
marginal of the byte distribution (most-frequent byte = space, 0x20). Greedy argmax
locked onto a single byte that doesn't decode to a valid UTF-8 string under the
`CharModel.predict()` byte→str filter, hence 0.0000 accuracy (instead of the ~18%
that "always predict space" would yield).

## Diagnosis

Inner-block weight updates were too small to break out of the unigram fixed point.
A back-of-envelope on the DFA-projected gradient magnitude (`e @ B / (B·T)` with
`B ∼ N(0, 1/d)`, batch×seq = 32×512 = 16 384) gives per-element grads ≈ 3 × 10⁻⁶ —
1–2 orders of magnitude smaller than what the chain-rule gradient at the same point
would be. The lm_head and final norm trained fine on the true CE gradient (loss got
to the unigram floor immediately) but the 6 attention/MLP blocks effectively did
not learn — DFA updates were swamped by SGD-momentum noise around their init.

The Launay 2020 transformer-DFA recipe scales feedback matrices per-component
(separate scales for QKV, attn-out, MLP-fc, MLP-out) and warms the head up before
turning DFA on. My single-scale block-output DFA tap is too coarse.

## FLOPs / energy claim verification

**Per-step energy:** 40 783 J / 4000 steps = **10.2 J/step** at batch 32 × T 512.
**Baseline:** 51 704 J / 2150 steps = **24.0 J/step** at batch 32 × T 1024.

Normalising for sequence length (DFA halved T to fit more steps in the budget):
**DFA per-token energy is ~0.62 mJ vs baseline ~0.73 mJ — a ~15% reduction.**

This is *less* than the spec's ~30% prediction. The gap is consistent with the
fact that, inside each block, attention/MLP backprop still runs (DFA only replaces
the cross-block chain). Modded_nanogpt also benefits from Muon's small Newton-Schulz
overhead per step that DFA's plain SGD doesn't pay. Net: directionally correct,
magnitude smaller than the spec's optimistic floor.

## Verdict

DFA at this implementation depth does NOT clear the 0.70 floor in 300 s. The energy
story is real but not large enough to justify a second-round investment given that
the harder problem — making inner-block DFA updates produce meaningful learning
signal — would require the per-component scaling + warmup recipe from Launay 2020,
which is a significant tuning project. Not a single-knob fix.

**Second-round potential:** modest. The credible path is (a) per-component DFA
scales (separate B for Q/K/V/proj/MLP-fc/MLP-out, each scaled to match its
chain-rule grad RMS), (b) a 200-step backprop warmup before switching to DFA, and
(c) higher LR with a separate schedule for the inner-block params. None of this
moves the energy claim; it just tries to actually hit the 0.70 floor that DFA
under-300-s clearly does not reach on its first attempt.

## Review (post-hoc audit)

**Validity for discarding DFA on a transformer LM:** *Insufficient.*

**Core limitations:**
- **Single-scale block-DFA is the weakest known DFA variant for transformers.** Launay 2020's working recipe requires per-component feedback-matrix scaling (separate B for Q/K/V/attn-out/MLP-fc/MLP-out), each scale matched to its chain-rule grad RMS, plus a ~200-step backprop warmup before switching to DFA, plus per-component LRs. None of those are in `dfa_v1`. The writeup's own diagnosis ("per-element grad ≈ 3×10⁻⁶, 1–2 orders below chain-rule") confirms the inner blocks effectively never updated.
- **val_char_accuracy = 0.0000** is not the unigram-floor accuracy (which is ~0.18 for English chars predicting space). The argmax is landing on a non-decodable byte under the UTF-8 filter in `predict()`. The writeup attributes this to the softcap-shaped logit geometry but does not report the raw top-1 byte, which would have made the diagnosis unambiguous.
- **Plain SGD-momentum was used** because "DFA pseudo-gradients have different statistics than Muon assumes". This is plausible but untested — an SGD-vs-Muon-vs-Adam ablation on the same DFA tap would isolate the optimizer-method interaction from the DFA-magnitude problem.

**Verdict:** Discards naive single-scale block-DFA + SGD-momentum at 4 000 steps. Does *not* discard DFA on transformers — the recipe with known empirical success in the literature was not implemented. A re-run is a prerequisite to any verdict on DFA.
