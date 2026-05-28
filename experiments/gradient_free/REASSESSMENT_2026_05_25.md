# Severity reassessment of cross-check findings — 2026-05-25 ~18:00Z

Cross-check originally produced ~30 design-defect findings against the
unimplemented experiment portfolio. Between the cross-check and this
reassessment, five experiments were submitted to Modal. The live evidence
changes the severity ranking of several findings.

## Live-run evidence summary

| Submission | Status | Outcome | Cross-check prediction |
|---|---|---|---|
| `rf_mlp_block2` (exp_06) | finished 231 s / 48.9 kJ, eval ~0.72 | tracking PASS | predicted "minor severity" — **confirmed** |
| `ff_pretrain_then_sgd` (exp_08) | finished 218 s / 48.2 kJ, eval ~0.71 | marginal PASS expected; FF Stage-1 dead | predicted "critical: random-corrupt negatives won't train" — **vindicated** (g_pos ≡ g_neg through 300 steps) |
| `hebbian_fw_block` (exp_07) v1 | DQ at 300.0 s / 36.5 kJ, step 500/2150 | wall-clock killed it | predicted both "normalization bug" and "time-cap risk" — **time-cap was the killer**; normalization stable but non-canonical |
| `mha_alpha05` (exp_19) | not yet submitted | n/a | predicted "wrong author attribution" — still present in v1 submission docstring |
| `noprop_terminal` (exp_09) | launched 17:37, in NVML calibration | n/a yet | predicted "label-embed collapse, KL fights orthogonality" — KL already down-weighted to 1e-3 (partial fix); label_embed still non-orthogonal |

## Severity reassessment for high-importance errors

| Finding | Original | Revised | Reason |
|---|---|---|---|
| exp_07 L2-vs-sum-norm | critical | **moderate** | v1 loss curve was monotonically decreasing (5.55 → 1.23), so non-canonical norm is at least stable. Wall-clock is the real critical bug. |
| exp_07 Python `for t` scan time-cap | (flagged "time-cap remains high risk") | **critical (vindicated)** | DQ at 300.0 s; only 500/2150 steps; ~0.59 s/step from inner loop. WY-Householder parallel scan is now mandatory. |
| exp_19 author attribution Tang→Masumura/Taki | factual | **still critical (pre-submission)** | Still in submission docstring; easy fix; reviewers will catch. |
| exp_19 sweep order (run α'=0.5 first) | (not flagged in cross-check) | **moderate** | α'=0 is the missing 4-layer baseline; running it first attributes everything else. |
| exp_09 label_embed orthogonal init | moderate | **still moderate** | v1 implements `normal(std=0.5)` — class crowding risk for 256 classes in d_label=128. KL down-weight to 1e-3 partially mitigates. |
| exp_09 KL fights orthogonal targets | moderate | **resolved by v1** | v1 multiplies KL by 1e-3, addressing the gradient-pressure concern. Still mathematically wrong (KL of two unit Gaussians = 0), but harmless. |
| exp_08 random-byte-corrupt negatives | critical | **critical (vindicated)** | g_pos ≡ g_neg throughout Stage 1; FF produced zero signal; v1 PASS (if any) is from Stage 2's body alone. Replacement with Mono-Forward / CwC is the right fix. |
| exp_08 Stage 2 step-count truncation | critical | **critical (vindicated)** | Stage 2 ran 1900/2150 = 88% steps; v1 eval at 0.71 is consistent with baseline-at-88%-steps. Confirms confound. |
| exp_06 frozen-MLP block savings | minor | **minor (confirmed)** | Run finished cleanly, accuracy on track. No revision needed. |

## Updated experiment design files (v2)

The four high-severity actionable findings now have written v2 designs:

- `experiment_07_v2_hebbian_fw_block.md` — WY-Householder parallel scan
  (axis A) + Schlag sum-norm (axis B1) + T=512 (axis C). Targets
  ≤ 0.18 s/step, finishes in 270 s wall.
- `experiment_19_v2_mha_alpha_sweep.md` — Masumura & Taki attribution
  fix throughout, mandatory α'=0 first (the missing 4L baseline), kernel
  pre-flight gate before any Modal spend.
- `experiment_08_v2_mono_forward_replacement.md` — replace random-corrupt
  FF with Mono-Forward (per-block CE on probe heads); add a probe-head
  val-acc gate that fails fast (v1's "gate ratio" was vacuously passing).
  Match Stage 2 step count to baseline for clean attribution.
- `experiment_09_v2_noprop_orthogonal_init.md` — equiangular tight frame
  label_embed init (256 classes in d_label=128), reconstruction loss
  against denoise-chain z_0 (not z_T_gt), drop the harmless-but-wrong
  KL term.

## Defects that remain critical but are NOT yet implemented

These are still in spec form only; my v2 files do not yet cover them but
the original cross-check critique stands:

- **spec_01 / exp_11 uMPS partition-function bug** for AR inference (needs
  transfer-matrix dominant eigenvector R_∞, not learned R).
- **exp_18 TT-HMM** fabricated "Cui 2016 IEEE TSP" citation.
- **exp_23 TTN / exp_24 MERA** AR-causality violation in symmetric tree
  structure.
- **spec_09 XGBoost** independent-bits assumption (need Bellard NNCP
  chained bit conditioning).
- **spec_10 CMA-ES** sample-budget gap (recommend cancel, not fix).
- **exp_14 / exp_15 ESN** Cholesky budget off by 6–10×, Triesch rule wrong
  form for tanh.

These should be revisited when the running experiments land and the
portfolio is repriotitized.

## Cross-references

- Original cross-check findings: agent reports from 2026-05-25 ~17:00Z
  (six parallel research-direction-explorer agents).
- Live submission logs:
  `submissions/{hebbian_fw_block, ff_pretrain_then_sgd, rf_mlp_block2,
   mha_alpha05, noprop_terminal}/run.log`
- v1 specs in `experiments/gradient_free/experiment_{06,07,08,09,19}_*.md`.
