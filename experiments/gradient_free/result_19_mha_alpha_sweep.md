# Result 19: MHA α' Sweep — Hopfield-coupled attention refuted at char-LM scale, missing baseline attributes prior Hopfield PASS to the trunk

**Date**: 2026-05-25
**Submissions**: `mha_alpha00`, `mha_alpha03`, `mha_alpha05`, `mha_alpha07`
**Spec**: `experiments/gradient_free/experiment_19_hopfield_coupled_attention_mha.md` (v1) with v2 corrections from `experiment_19_v2_mha_alpha_sweep.md` applied (author attribution, sweep order, per-layer α' assert, kernel pre-flight).
**Mechanism**: Modern Hopfield Attention (Masumura & Taki, NeurIPS 2025, arXiv 2511.20698) implemented via SDPA + additive `attn_mask` — see `submissions/mha_alpha05/submission.py` and the "Kernel" section of the v1 spec.

## Summary table

| submission       | α'    | val acc | E (kJ) | dur (s) | SKU                | E / stress |
|------------------|-------|---------|--------|---------|--------------------|------------|
| `mha_alpha00`    | 0.0   | **0.7306** | **34.9** | 170 | A100-PCIe          | 4.00       |
| `mha_alpha03`    | 0.3   | 0.7299  | 53.3   | 237     | A100-PCIe          | 6.12       |
| `mha_alpha05`    | 0.5   | 0.7270  | 56.8   | 243     | **A100-SXM4**      | 4.29       |
| `mha_alpha07`    | 0.7   | 0.7189  | 52.3   | 224     | A100-PCIe          | 6.09       |
| `hopfield_layer` | (4L + frozen-K Hopfield bank, M=4096) | 0.7293 | 40.2 | 184 | A100-PCIe | 4.64 |
| `modded_nanogpt` | (6L baseline)                        | 0.7374 | 51.7 | 247 | A100-PCIe | 5.91 |

`E / stress` = `training_energy_J / _nvml.stress_energy_joules` — a SKU-invariant unit of "training work measured in stress-test equivalents," used because α'=0.5 happened to land on the SXM4 variant (idle 64 W, stress 350 W) while the other three cells got the PCIe variant (idle ~55 W, stress ~232 W).

## Two findings

### 1. The headline: MHA does not transfer to char-level / 4-layer Muon

Accuracy is **monotonically decreasing** in α':

```
α' = 0.0 → 0.7306
α' = 0.3 → 0.7299    (−0.07 pp)
α' = 0.5 → 0.7270    (−0.36 pp)
α' = 0.7 → 0.7189    (−1.17 pp)
```

This is the **refutation** branch of the v1 spec's success criteria. The published wins of Masumura & Taki (GPT-2 small WikiText-103, 22.87 → 20.70 PPL; ViT-Tiny CIFAR-100, α' = 0 → 0.5 giving +2.24 pp) do **not** transfer to 4-layer char-LM on byte-level WikiText-103 with Muon. Plausible explanations (in order of how testable they are):
- **Rank collapse isn't a bottleneck at depth-4.** MHA is motivated as a fix for rank collapse / token uniformity in *deep* transformers. At 4 layers, the residual stream maintains rank naturally; the cross-layer EMA mixes in stale score statistics that *introduce* dilution rather than preventing collapse.
- **Char-level signal is local.** Wikitext byte-level prediction is dominated by within-word local patterns; the EMA's "soft long-range bias from earlier layers" pushes attention away from local-only patterns that vanilla softmax is sharp about.
- **Muon's spectral-update regime may already produce the rank-preservation effect MHA targets**, in which case adding MHA is redundant at best.

The mechanism additionally costs **+17 to +22 kJ** (~50% more training energy) for the EMA's HBM traffic on the (B,H,T,T) score tensor at each layer, with no accuracy upside.

### 2. The substantive byproduct: `hopfield_layer`'s "Hopfield PASS" was the trunk

The α' = 0 cell (`mha_alpha00`) was the explicit purpose of v2's "Fix B — sweep ordering": fill in the missing 4-layer-Muon-without-Hopfield baseline that the existing portfolio never ran. With it now on the leaderboard:

```
mha_alpha00      4L Muon, no Hopfield                   0.7306    34.9 kJ
hopfield_layer   4L Muon + frozen-K Hopfield bank        0.7293    40.2 kJ
                                                        ──────    ───────
                 Hopfield mechanism contribution         −0.0013   +5.3 kJ
```

`hopfield_layer` (M=4096 frozen-random-K retrieval inserted after block 2) is **strictly worse than the bare 4-layer trunk** on both axes. The previously-recorded "Hopfield experiment PASS" was the 4-layer Muon trunk — the same finding that motivated this experiment in the first place. The Hopfield retrieval bank, as instantiated there (random-init encoder for K, raw byte embeddings for V), contributed **negative net value**.

This vindicates the original Tier-3 critique that prompted the experiment: *"the 0.729 finding can't be attributed to the named mechanism."* It now can be attributed — to the *trunk*, not to Hopfield. Every prior Hopfield experiment in the portfolio (planned exps 01–04, 07, 10) shares the same trunk and the same attribution gap; their results, if run, would need to be interpreted against this 0.7306/34.9 kJ baseline, not against the 0.7374/51.7 kJ 6-layer reference.

## Pareto comparison vs. portfolio

`mha_alpha00` dominates the Pareto front in (acc, energy) among the gradient-free-flavored portfolio:

- vs `hopfield_layer` (0.7293 / 40.2 kJ): **better on both axes** (+0.0013 acc, −13% energy).
- vs `modded_nanogpt` (0.7374 / 51.7 kJ): −0.0068 acc, **−32% energy**. ~99% of baseline accuracy at ~68% of baseline energy.
- Stress-normalized: 4.00 vs 5.91 for the 6L baseline and 4.64 for the prior Hopfield row.

The "win" here is mundane — it's the depth-reduction win. Two transformer layers off a Muon trunk at this scale cost ~0.007 acc and save ~17 kJ. Nothing about Hopfield, frozen memory banks, or EMA-coupled attention. Worth noting on the leaderboard.

## Kernel validation

The custom SDPA-with-additive-bias kernel (`HopfieldCoupledAttention` in `submissions/mha_alpha05/submission.py`, the algebraic rewrite `h_n = QK^T·(1−α')scale + α'h_{n−1}` consumed via SDPA's `attn_mask` argument) behaved exactly as the pre-flight smoke test predicted:

- **α' = 0 → exact SDPA match.** All four α' = 0 result cells run through the `is_causal=True` fast path; this completed in **170 s** (vs ~165 s predicted from the 4L SDPA baseline projection).
- **α' > 0 → ~+32 ms/step**. The smoke test measured +32 ms fwd+bwd and projected +69 s training; α' = 0.3 took 237 − 170 = **67 s** more than α' = 0. Within 3% of prediction.
- **No DQs**, no kernel correctness issues, no NaNs reported, gradient flow through the cross-layer `h` chain confirmed at training scale.

## Operational footnotes

- **Modal A100 SKU heterogeneity** (also noted in the v1 spec's failure-modes section): `gpu="A100-80GB"` covers both PCIe and SXM4. α' = 0.5 happened to land on SXM4 (idle 64 W vs PCIe ~55 W, stress 350 W vs ~232 W). Per-row energy comparisons within α' ∈ {0, 0.3, 0.7} (all PCIe) are direct; α' = 0.5 needs the `E/stress` normalization to be cross-comparable with the rest of the portfolio (which is PCIe). The conclusion is robust either way — α' = 0.5 sits squarely between α' = 0.3 and α' = 0.7 on the accuracy curve.
- **`MHA_ALPHA_PRIME` env-var approach failed**: `submit.py` only forwards `SEED`, not arbitrary env vars, so the env-var sweep scheme in the v1 spec doesn't work. Resolved by `cp -r submissions/mha_alpha05 submissions/mha_alpha{00,03,07}` and `sed`-patching the α' fallback. The v1 spec has been updated to reflect this.
- **Author attribution**: v1 spec and `submission.py` docstrings have been corrected from "Tang & Kopp" to "Masumura & Taki" per v2 Fix A.

## What this implies for the portfolio

- **`hopfield_layer` should be re-baselined** in any subsequent comparison: the relevant reference is now `mha_alpha00` (0.7306 / 34.9 kJ), not `modded_nanogpt` (0.7374 / 51.7 kJ).
- **Planned Hopfield variants 01 (M-sweep), 02 (K-means K), 03 (learnable V), 04 (Hopfield+LWTA), 10 (online memory)** all need to clear `mha_alpha00`'s 4.00 stress-normalized energy to be a genuine Hopfield contribution. Several of those (especially 01 with M = 16K, 65K) were predicated on saturating below `hopfield_layer`'s 4.62 — a much weaker bar than 4.00.
- **MHA itself is closed at this scale**. The mechanism is sound at large GPT-2 / ViT scale but does not transfer; pursuing the v2's optional "α + α' both ≠ 0" full-MHA variant would only be worthwhile if there's a specific reason to expect the depth-4 char-LM regime to suddenly start benefiting. We see none.

## References

- v1 spec: `experiments/gradient_free/experiment_19_hopfield_coupled_attention_mha.md`
- v2 spec: `experiments/gradient_free/experiment_19_v2_mha_alpha_sweep.md`
- Pre-launch reassessment: `experiments/gradient_free/REASSESSMENT_2026_05_25.md`
- Submissions: `submissions/mha_alpha{00,03,05,07}/{submission.py, result.json, run.log}`
- Kernel smoke test (run on Modal A100-80GB before the sweep): `submissions/mha_alpha05/test_kernel.py`
- Source paper: Masumura & Taki 2025 "On the Role of Hidden States of Modern Hopfield Network in Transformer" (arXiv 2511.20698)
- Original critique that prompted the experiment: cross-check finding "the 0.729 finding [in `hopfield_layer`] can't be attributed to the named mechanism."
