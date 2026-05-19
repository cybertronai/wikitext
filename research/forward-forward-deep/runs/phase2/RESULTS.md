# Phase 2 — Diagnostic Results

All five Phase-2 runs completed on Modal A100-80GB. Pass-2 baseline for comparison: **val 0.2792, energy 3,845 J, 105.7 s**.

## Headline table

| ID | What it measured | val_acc | energy (J) | train (s) | Δ vs pass-2 |
|---|---|---:|---:|---:|---|
| **P2-A** | Frozen-random 5×384 + ridge (no FF) | 0.2190 | 70 | 2.9 | −0.060 |
| **P2-B** | FF + per-layer ridge probes (best=L3) | 0.2658 | 3,898 | 107.1 | −0.013 |
| **P2-C** | FF width 1024, 5k steps | 0.3342 | 3,787 | 43.9 | **+0.055** |
| **P2-D** | FF K=64 context | 0.2073 | 4,957 | 77.1 | **−0.072** |
| **P2-E** | FF bigram-aware input | 0.3995 | 4,284 | 148.1 | **+0.120** |

## Layer-wise probe (P2-B)

Val accuracy of an independent ridge readout fit on each FF layer's activations alone:

| Layer | val_acc(20K val chars) |
|---:|---:|
| 1 (frozen random) | 0.2377 |
| 2 | 0.2465 |
| 3 | **0.2573** (peak) |
| 4 | 0.2506 |
| 5 | 0.2422 |

## Key findings

1. **FF rule only adds ~0.06 over random projection.** Random projection floor (P2-A) is 0.219; pass-2 FF was 0.279. Closing the 0.42 gap to 0.70 cannot rely on the FF rule alone.

2. **Hierarchy peaks at layer 3, degrades after.** Layers 4 and 5 are *worse* than layer 3 for next-byte ridge readout. FF over-trains its local objective in deep layers — the concat-of-layers ridge wins on feature *count*, not feature *quality*.

3. **Width is mildly load-bearing.** Width 1024 (P2-C) gives +0.055 even with step count cut 14k → 5k. Slope is non-trivial; Phase 7 P7-1..3 should prioritise width 1024+.

4. **K=64 actively hurts.** Context expansion with sparse one-hots took us *down* by 0.07. FF can't extract distant-context signal at this depth/width with this encoding. **Phase 7 P7-3 / P7-4 (K=128 sweeps) are dead unless paired with a structurally different encoding.**

5. **Input encoding is the dominant axis.** Bigram-aware input (P2-E) gives **+0.120** — biggest single-variant lift. Densifying the layer-1 input recovers signal that the K=24 one-hot was leaving on the floor. This implicitly explains P2-D's failure: K=64 makes the input *sparser*, not richer.

## Implications for Phases 3–7

- **Phase 3 (rule variants) priors weaken.** The FF rule is not the rate-limiter on the current setup; swapping the goodness function or loss is unlikely to lift > 0.02 unless paired with a better input encoding. Phase 3 will still execute as planned (Mono-Forward in particular is genuinely different — it eliminates contrastive loss entirely), but expectations are tempered.
- **Phase 4 (backbones) priors strengthen.** Conv-1D and dilated conv stems map directly onto the "structured-encoding helps" finding from P2-E. The conv stem effectively gives every layer a P2-E-style densified view of local n-gram structure.
- **Phase 7 (scale & combine) should start with `bigram_input × width_1024` compound** rather than `width × K`. The P2-D negative result downweights any K > 32 experiment.

## Reconstruction note

P2-B, P2-C, P2-D, P2-E result.json + nvml.json files were **reconstructed from the Modal streaming logs** after a mid-flight directory reorganisation broke submit.py's local-write path. The Modal compute itself completed normally; the val_char_accuracy, training_energy_J, and training_duration_s values are exactly as reported by run_eval.py. Reconstructed entries carry `"_reconstructed_from_log": true`. P2-A's result.json is original (it ran before the path break).
