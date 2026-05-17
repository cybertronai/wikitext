# New Research Directions

Six additional hypothesis-driven specifications, in the same format as `files.zip` (specs 1–6). These cover mechanisms identified in `../RESEARCH_DIRECTIONS.md` and the empirical signals from `../.survey/REPORT.md` that were not yet written as standalone, gated specs.

| # | Title | Source | Effort | Energy bet |
|---|------|--------|--------|------------|
| 7 | [PPM Context Tree + Tiny Neural Residual](spec_7_ppm_neural_residual.md) | Survey best result (0.63 / 633 J) | 2–4 days | Replace neural compute with count-table compute; patch residual gap |
| 8 | [Fast-Weight Programmer with Delta Rule](spec_8_fwp_delta_byte_lm.md) | RESEARCH_DIRECTIONS A2 | 2–3 days | No KV-cache; constant per-token compute |
| 9 | [LWTA Drop-in for modded-nanogpt](spec_9_lwta_dropin.md) | RESEARCH_DIRECTIONS B1 | 2–6 hours | 1/k structural sparsity in MLP |
| 10 | [Forward-Forward on Byte-Level Text](spec_10_forward_forward_bytes.md) | RESEARCH_DIRECTIONS A3 | 1–2 days | No backward sweep; no activation stash |
| 11 | [Neural Bucket Brigade](spec_11_neural_bucket_brigade.md) | RESEARCH_DIRECTIONS B9 | 1d diag + 3–5d port | Truly gradient-free local rule |
| 12 | [Chunker + FWP-Delta Hybrid](spec_12_chunker_fwp_hybrid.md) | RESEARCH_DIRECTIONS H1 | 3–5 days | Step-count × per-step-memory compound savings |

## Dependency graph

- Spec 12 depends on Spec 3 (from files.zip) and Spec 8.
- Spec 9 is a near-zero-cost sanity that informs whether LWTA is worth including in any hybrid.
- Specs 7, 10, 11 are standalone.

## Recommended order

1. **Spec 9** (LWTA) — 2 hours. Cheapest signal in the program.
2. **Spec 7** (PPM + residual) — anchored on the survey's strongest empirical result; highest probability of crossing the 0.70 floor.
3. **Spec 8** (FWP-delta) — cleanest "different paradigm" story; required prerequisite for Spec 12.
4. **Spec 10** (FF) — informative regardless of outcome; layer-local learning at byte scale is its own question.
5. **Spec 11** (NBB) — Day-1 diagnostic decides the multi-day port.
6. **Spec 12** (hybrid) — only if Specs 3 and 8 both pass.

Each spec contains a hypothesis, a single first experiment with a go/no-go gate, measurements to record, and what positive and negative outcomes mean for the broader research program. All specs assume the 300-second / A100-80GB / NVML-joule harness defined in the project README.
