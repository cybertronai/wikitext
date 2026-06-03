# chunker_phase1_v1 — Schmidhuber chunker Phase 1 (1991/1993) — PASS

**Result: PASS at 0.7057 acc / 5,918 J (A100-SXM4-80GB).** Above floor
by 0.57pp; 7.99× under baseline.

**FIRST Schmidhuber chunker Phase 1 build on a modern byte-LM benchmark
to clear the 0.70 floor.** Architecturally distinct from the
attention/SSM lineage that dominates the leaderboard. Demonstrates that
the 1991 hierarchical surprise-gated idea works on natural language byte
streams at modest cost.

## Architecture

- **Lower tier L:** GPU KN n-gram (W31-style, order-12, with XOR-bit
  sort fix). Provides the surprise signal:
  `p_L(true_byte | context)` via order-4 n-gram MLE.
- **Upper tier H:** d=192, L=4 modded-nanogpt transformer, Muon+AdamW,
  800 steps. Trained with cross-entropy MASKED to surprise positions
  only — capacity goes to hard bytes, not easy n-gram-solvable ones.
- **Output combiner:** at predict(), surprise-gated blend:
  - If `max(p_KN) >= 1 - tau` (KN confident): `0.85 * p_KN + 0.15 * p_NN`
  - Else (surprise predicted at inference): `0.5 * p_NN + 0.5 * p_KN`

## Empirical numbers

- TAU = 0.30 (n-gram-MLE definition; equivalent to D1's transformer-tau=0.1)
- Realized p_s on WikiText train: 0.4351 (43.5% of bytes are surprise)
- KN build: 49.0s
- Surprise mask: 2.6s
- H training: 44s (800 steps, loss 5.55 → 2.25)
- Total train: 98.9s / 5,918 J
- Eval: 376.7s at 159 char/s

## Pareto position

Dominated by xorfix (3,172 J / 0.7184) on energy AND accuracy. But:
- **Unique paradigm**: only hierarchical surprise-gated arch among
  passing entries.
- **7.99× under baseline** at modest acc.
- **Useful negative result**: chunker Phase 1 PASSes but does NOT beat
  classical-hybrid Pareto. The "NN should specialize on hard bytes"
  intuition doesn't translate to a Pareto win at this scale.

## What the 3-run budget revealed

- **Run 2 (chunker_phase1_v2):** d=256/L=4/1200 steps NN, ALPHA=0.6
  fixed (no surprise gating). **DQ 0.5621**. -14pp.
  → Surprise-gated inference mix is essential. Removing it destroys
  the architecture: NN trained on subset is BAD on easy bytes.
- **Run 3 (chunker_phase1_v3):** Schmidhuber's literal hard-switch
  combiner. **DQ 0.6725**. -3.3pp.
  → Hard switch loses KN's contribution at hard bytes; soft blending
  is essential.

## What's notable

The 1991 paper's literal combiner (hard switch on surprise) UNDERPERFORMS
a soft mix. The 2025 dynamic-patching descendants (BLT, SpaceByte, H-Net)
all use soft routing — this run validates that choice empirically on a
new benchmark.

The configuration that works is a fragile sweet spot:
- tau=0.30 on order-4 n-gram MLE
- d=192/L=4 NN at 800 Muon steps
- Surprise-gated inference mix: KN-heavy on easy bytes, balanced on hard

## Status

- **Adaptive 3-run budget closed.** No substantial improvement between
  runs → no extension.
- **PCIe revalidation needed.** SXM4 → PCIe gap typically +20-50% J.

**Contributor:** @explore-chunker-2026-05-19
