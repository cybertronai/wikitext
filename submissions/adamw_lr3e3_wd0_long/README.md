# adamw_lr3e3_wd0_long — RUN 4 (budget extension)

**Paradigm:** Optimizer alternative (C11). Run 4 of the AdamW reopen
budget. Budget extended from 3 → 5 (per iterative-research SKILL.md
"substantial improvement" rule) because runs 1-3 showed a clear LR-axis
trajectory: lr=1e-3 → 0.625, lr=2e-3 → 0.633, lr=3e-3 wd=0.0 → 0.675
acc. Going from lr=1e-3 to lr=3e-3 wd=0.0 = +5pp acc, meets the
"substantial improvement" threshold.

**Mechanism:** Identical to `adamw_lr3e3_wd0` (the run 3 winner)
**except n_steps=4500 instead of 1500**. In run 3, loss was still
descending at step 1499 (1.16 with no plateau) and only used 60s out of
the 300s wall-clock cap. Adding 3× more training (1500→4500 steps,
~180s) should push loss further down. If loss-acc correlation holds
(run 3: loss 1.16 → acc 0.675), reaching loss ~1.05 should give acc
~0.70-0.72.

Same arch as E2 (d=256, L=4, bs=32, T=1024), same training loop, same
stable-then-decay schedule with cooldown_frac=0.7. AdamW for ALL
parameters at lr=3e-3, wd=0.0, betas=(0.9, 0.95).

**Why this is the right run 4:** Two candidates considered:
(a) push LR higher (lr=5e-3); (b) more steps at known winning recipe.
Option (b) is lower-risk — lr=3e-3 wd=0.0 is empirically validated, and
the loss trajectory shows no plateau. Option (a) risks divergence at
higher LR. If (b) clears 0.70, the paradigm reopens. If (b) plateaus
below 0.70, we'll have a definitive bound: "AdamW with proper LR + 3×
the training time still can't reach Muon."

**Expected joules:** ~42-45 kJ (3× more energy than run 3's 13.9 kJ).
**Expected accuracy:** if loss-acc holds → 0.69-0.72.

**Smoke test:** SAME as adamw_lr3e3_wd0; only delta is n_steps int.

**Stop condition update:** if this clears 0.70 → paradigm validated +
ship lr=3e-3 wd=0.0 + 4500 steps as the canonical AdamW recipe. If
plateaus at <0.69 → AdamW cluster definitively closed with the 4-point
trajectory: {1e-3/1500, 2e-3/1500, 3e-3/1500, 3e-3/4500}.
