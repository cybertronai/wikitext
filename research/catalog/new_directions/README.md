# New Research Directions

Hypothesis-driven specifications in the same format as `files.zip` (specs 1–6). Specs 7–12 cover gradient-free mechanisms from `../RESEARCH_DIRECTIONS.md`. Specs 13–16 cover Manning-collaborator-graph candidates (Hyena, Mamba, Pointer-Sentinel) and the standalone chunker (`RESEARCH_DIRECTIONS.md` § A1) that wasn't previously written as its own spec.

| # | Title | Source | Effort | Energy bet |
|---|------|--------|--------|------------|
| 7 | [PPM Context Tree + Tiny Neural Residual](spec_7_ppm_neural_residual.md) | Survey best result (0.63 / 633 J) | 2–4 days | Replace neural compute with count-table compute; patch residual gap |
| 8 | [Fast-Weight Programmer with Delta Rule](spec_8_fwp_delta_byte_lm.md) | RESEARCH_DIRECTIONS A2 | 2–3 days | No KV-cache; constant per-token compute |
| 9 | [LWTA Drop-in for modded-nanogpt](spec_9_lwta_dropin.md) | RESEARCH_DIRECTIONS B1 | 2–6 hours | 1/k structural sparsity in MLP |
| 10 | [Forward-Forward on Byte-Level Text](spec_10_forward_forward_bytes.md) | RESEARCH_DIRECTIONS A3 | 1–2 days | No backward sweep; no activation stash |
| 11 | [Neural Bucket Brigade](spec_11_neural_bucket_brigade.md) | RESEARCH_DIRECTIONS B9 | 1d diag + 3–5d port | Truly gradient-free local rule |
| 12 | [Chunker + FWP-Delta Hybrid](spec_12_chunker_fwp_hybrid.md) | RESEARCH_DIRECTIONS H1 | 3–5 days | Step-count × per-step-memory compound savings |
| 13 | [Hyena Hierarchy](spec_13_hyena.md) | Manning REPORT.md branch 2 | 1–2 days | Sub-quadratic sequence mixer; FFT-based long conv replaces attention |
| 14 | [Mamba (Selective SSM)](spec_14_mamba.md) | Manning REPORT.md branch 2 | 1 day | Constant-state recurrence; no KV cache |
| 15 | [Pointer-Sentinel at Char Level](spec_15_pointer_sentinel_char.md) | Manning REPORT.md branch 6 | 2 days | Small backbone + char-pointer absorbs repetition |
| 16 | [Schmidhuber Chunker (standalone)](spec_16_chunker.md) | RESEARCH_DIRECTIONS A1 | 2 days + D1 gate | Surprise-gated 2-level architecture: skip predictable bytes |
| 17 | [DiffusionBlocks AR](spec_17_diffusionblocks.md) | Sakana AI, ICLR 2026 ([arxiv:2506.14202](https://arxiv.org/abs/2506.14202)) | 3–5 days | One-block-at-a-time gradients via EDM denoising; activation-memory headroom may enable wider models inside the 300 s budget |

## Dependency graph

- Spec 12 depends on Spec 3 (from files.zip) and Spec 8.
- Spec 9 is a near-zero-cost sanity that informs whether LWTA is worth including in any hybrid.
- Spec 16 has a Phase-0 D1 surprise-rate diagnostic that gates the full implementation.
- Specs 7, 10, 11, 13, 14, 15 are standalone.

## Recommended order (post-LWTA)

1. **Specs 13, 14, 15, 16** — Manning branch + chunker, dispatched in parallel as of 2026-05-18. Each beats `lwta_k2` (46.1 kJ / 0.7146, current leader) at its energy target.
2. **Spec 7** (PPM + residual) — anchored on the survey's strongest empirical result.
3. **Spec 8** (FWP-delta) — cleanest "different paradigm" story; prerequisite for Spec 12.
4. **Spec 10** (FF) — `research/forward-forward-deep/` has Phase 2 done; Phases 3–7 pending.
5. **Spec 12** (hybrid) — only if Spec 8 and Spec 16 both pass.

Each spec contains a hypothesis, a single first experiment with a go/no-go gate, measurements to record, and what positive and negative outcomes mean for the broader research program. All specs assume the 300-second / A100-80GB / NVML-joule harness defined in the project README.
