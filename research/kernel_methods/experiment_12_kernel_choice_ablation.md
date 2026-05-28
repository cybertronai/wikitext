# Experiment 12: Kernel Choice Ablation on Best Working Configuration (Dot vs Cosine vs RBF vs Arc-Cosine)

## Hypothesis
Once a kernel-LM configuration clears the gate (likely exp 04 or exp 05 from this portfolio), swapping the kernel/feature map among {linear-dot, cosine, RBF, arc-cosine-order-2, NTK-RFF} does NOT materially change accuracy at fixed compute — confirming that on learned text embeddings, kernel choice is a second-order effect, validating the `finding_rbf_text_isotropy.md` note that RBF is not specially advantaged on text.

## Motivation
Empirical skepticism (per agent instructions): the literature on "RBF for X" often omits the comparison to a plain dot kernel on the same features. If dot kernel ties RBF here, that's a published-finding-verification result worth keeping in memory for future surveys; it also closes the "should I sweep 5 RBF variants" question definitively.

Cross-references: `finding_rbf_text_isotropy.md` (the prior reason to expect this); `survey_kernel_methods_2026_05.md` (kernel-family taxonomy).

## Method
On whichever of exp 04 (Linear-Tx with elu+1), exp 05 (DeltaNet), or exp 07 (Falkon hybrid) cleared the gate at lowest energy, sweep the kernel/feature map only:

For Linear-Tx (exp 04) family:
| Variant | φ(x) |
|---|---|
| baseline (exp 04) | elu(x) + 1 |
| dot-positive | softplus(x) |
| cosine | (x / ‖x‖)·√c where c is a learned per-head scale |
| RBF-RFF | (1/√k)·cos(ωᵀ x + b), ω ~ N(0, σ⁻² I) |
| arc-cosine-RFF | Han/Avron 2021 NTK-RFF |
| polynomial p=2 | [x ⊗ x] (outer-product-flatten) |

For Falkon hybrid (exp 07):
| Variant | Kernel |
|---|---|
| baseline | linear (cosine after normalization) |
| RBF | exp(-‖a-b‖²/2σ²) |
| arc-cosine | (1/π)‖a‖‖b‖(sin θ + (π-θ) cos θ) |
| polynomial | (aᵀb + c)^d, d=2, c=1 |
| Matérn-3/2 | (1 + √3 r/σ) exp(-√3 r/σ) — verify it adds anything over RBF |

Run all variants at fixed compute (same n_steps for SGD variants, same M for Nyström variants).

## Memory-Movement Analysis
- All variants have similar arithmetic intensity (within 2×); the differences are constant-factor in the feature-map cost
- RFF-based variants need extra ω-matrix storage but it's small (d × k × 2 bytes ≈ 100 KB)
- **Total energy of this experiment ≈ 5 × baseline-variant energy.** Run all 5 in a single Modal submission via a `mode=` env var.

## Setup
- Anchor: whichever of exp 04 / 05 / 07 cleared 0.70 at lowest energy. If multiple cleared, pick exp 04 (cheapest).
- Other config matches the anchor exactly
- Hardware: 1 × A100-80GB per variant, OR all 5 variants in one Modal run via sequential train calls
- Baseline (for this ablation): the anchor experiment's own result
- Metric: val char-acc and energy per variant

## Procedure
1. Take the anchor's `submissions/X/submission.py`.
2. Parameterize the kernel/feature map via an env var: `KERNEL ∈ {elu, softplus, cosine, rbf, arccos, poly}`.
3. Submit each variant separately (5 submissions). Use `--name` flag to disambiguate result folders.
4. Compile a small table of val char-acc and energy across variants.

## Success Criteria
- **Hypothesis confirmed:** max(val) - min(val) across variants ≤ 0.02 (i.e., within noise) → kernel choice is not the bottleneck; RBF is not specially helpful on text. Record in agent memory.
- **Hypothesis refuted (interesting):** one variant clearly beats others by >0.05 → opens a follow-up. RBF winning would be a surprise per the isotropy concern; arc-cosine winning would confirm Cho/Saul.
- **Numerical issues isolate one variant:** if RBF blows up due to bandwidth tuning issues, note as known issue rather than treat as evidence.

## Failure Modes & Diagnostics
- **Bandwidth σ varies wildly across variants:** for each kernel, do a quick median-heuristic computation on the first 4K embeddings before training (cheap).
- **One variant won't fit in 300 s:** the polynomial-p=2 feature map has dim d² — for d=64 head_dim, 4096 features per head per token. Skip if it OOMs.
- **The anchor variant doesn't clear the gate:** then there's no result to ablate on. Fall back to comparing all variants at *whatever* accuracy they reach, and report relative differences.

## Estimated Cost
- 5 Modal A100 runs (or 1 chained run), ~30 min total wall, expected energy 150-300 kJ total
- ~$2.00

## References
- Cho & Saul 2009 "Kernel Methods for Deep Learning" — arc-cosine
- Han & Avron 2021 "Random Features for the Neural Tangent Kernel" (arXiv 2104.01351)
- Rahimi & Recht 2007 "Random Features for Large-Scale Kernel Machines" — RBF-RFF
- Mu & Viswanath 2018 "All-but-the-Top: Simple and Effective Postprocessing for Word Representations" — text-embedding anisotropy reference cited in finding_rbf_text_isotropy.md
