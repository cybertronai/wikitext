# chunker_phase1_v2 — Schmidhuber chunker Phase 1, run 2 (DQ)

**Result:** DQ at 0.5621 acc / 13,936 J. Below floor by 13.8pp.

**Changes from v1:**
- `H_MODEL_DIM`: 192 → 256 (match alpha_06 NN capacity)
- `H_MAX_LEN`: 512 → 1024
- `H_N_STEPS`: 800 → 1200
- `TAU`: 0.30 → 0.15 (target p_s ~ 0.20; actual p_s = 0.3084)
- `ALPHA`: 0.50 → 0.60 (no surprise gating at inference)

**Hypothesis was wrong.** Capacity wasn't the limiter; the surprise-gated
inference mix WAS. Removing it destroyed the architecture: a larger NN
trained on a SMALLER subset (30.8% vs 43.5%) gets pushed toward
overfitting hard examples, and ALPHA=0.6 makes it dominant on easy bytes
where it has no training signal.

**Critical finding for the chunker paradigm:** v1's surprise-gated
inference mix (`if KN.max>=1-tau: 0.85*KN+0.15*NN else: 0.5*NN+0.5*KN`)
is essential.

**Status:** DQ. Stayed in adaptive-budget rule (3-run budget).
