# Experiment 19 v2: Hopfield-Coupled Attention (MHA) — corrected attribution + sweep ordering

## Status

v1 spec: `experiment_19_hopfield_coupled_attention_mha.md` (2026-05-25 17:25Z).
v1 submission: `submissions/mha_alpha05/submission.py` (2026-05-25 17:22Z).
**Not yet submitted to Modal.** Pre-flight kernel test exists
(`submissions/mha_alpha05/test_kernel.py`). Default config is
`alpha_prime = 0.5`.

## What v1 got wrong

1. **Critical (factual): paper authors are wrong throughout.** Both the v1
   spec and `submissions/mha_alpha05/submission.py` docstring attribute
   "Modern Hopfield Attention" to "Tang & Kopp, NeurIPS 2025." The actual
   authors of arXiv 2511.20698 are **Tsubasa Masumura and Masato Taki
   (Rikkyo University)**. A technician trying to look up the paper via the
   docstring authors will fail; reviewers will discount the spec. Easy fix,
   high embarrassment if not fixed.

2. **Moderate (procedural): sweep order is backwards for attribution.** v1
   plan: run α' = 0.5 first as the headline, then α' ∈ {0, 0.3, 0.7} as
   the sweep. But the **stated motivation** in v1 §"Motivation" is that
   α' = 0 is the missing 4-layer-Muon-without-Hopfield control — the
   baseline whose absence makes every other Hopfield experiment in the
   portfolio unfalsifiable. Running α' = 0.5 first means the first Modal
   spend produces an ambiguous result (could be from depth, the Muon
   optimizer, the kernel correctness, OR from Hopfield coupling). The
   α' = 0 run must come first so that subsequent α' > 0 cells have a
   matched-stack reference.

3. **Minor: flex_attention warmup is unbudgeted.** `torch.compile` of
   `flex_attention` has a documented multi-second first-call warmup that
   v1's wall-clock estimate doesn't carve out. At 4 layers, 2150 steps,
   even a 100 ms-per-layer warmup at first step is < 1% of budget — small
   but specify it explicitly so it's not conflated with a per-step
   regression. **The custom kernel itself is the right design** — see
   Fix C below for why we keep it across the whole α' sweep including
   α' = 0.

4. **Minor (cross-cutting): `HopfieldCoupledAttention.alpha_prime` is a
   per-layer attribute set to the same value across all layers.** v1
   correctly treats α' as a single scalar hyperparameter, but the
   per-layer storage is a footgun — a future "alpha-per-layer" variant
   should be Phase 2, not accidentally enabled via per-layer mutation.

## v2 design

### Fix A — author attribution (mandatory; trivial)

`submissions/mha_alpha05/submission.py` line 4 and v1 spec lines 4, 13:
replace "Tang & Kopp" with "Masumura & Taki" everywhere. The reference list
in the spec is correctly pointing at arXiv 2511.20698 — only the human-
readable byline is wrong.

### Fix B — sweep ordering (mandatory)

Run the four α' values in this order, each as a separate Modal submission
with directory copy (`cp -r submissions/mha_alpha05 submissions/mha_alpha{0,03,05,07}`):

1. **α' = 0.0** — *the attribution baseline*. Produces the 4-layer-Muon
   reference accuracy that every other Hopfield experiment is missing.
2. **α' = 0.5** — *the headline cell* (matches v1 default).
3. **α' = 0.3** — fill in the curve if (1) and (2) diverge by > 0.005 acc.
4. **α' = 0.7** — stretch arm; only if (2) > (1).

If (1) ≥ 0.73 acc on its own (i.e., the missing baseline is essentially the
6L baseline result), then steps (2)-(4) become high-prior failure cells and
worth running only as the published-already result. If (1) < 0.71, the
α' = 0 cell is itself a substantive finding ("4-layer Muon does not match
6-layer Adam") that resets the portfolio.

### Fix C — keep the custom flex_attention kernel; gate it, don't replace it

**Default path is the v1 submission's `torch.nn.attention.flex_attention`
with the captured-`h` `score_mod`.** Reasoning (v1 submission.py line
138–142, kept verbatim in v2):

> At α'=0 the EMA collapses to identity (h_n = scores_n) and the layer is
> mathematically equivalent to vanilla attention. We still go through the
> FlexAttention kernel (not SDPA) so **timing is comparable across the
> α' sweep — energy differences between α' = 0 and α' > 0 are then
> attributable to the EMA mechanism rather than kernel choice**.

This is the right call. Reverting to SDPA at α'=0 would confound the
α'-sweep energy comparison with a kernel-substitution effect. Keep flex
all the way through.

Run `test_kernel.py` locally before any Modal spend and verify:

- **Correctness**: `flex_attention(α'=0)` matches an explicit
  `softmax(Q·K^T)·V` math-attention reference within 1e-3 max-abs and
  1e-4 mean-abs at fp32 (bf16 expected slightly worse).
- **Throughput**: a forward+backward at (B=8, H=6, T=512, d_head=64)
  completes in ≤ 200 ms after the first `torch.compile` warmup call.
  If > 200 ms, **do not fall back to SDPA** (that breaks sweep
  attribution). Instead drop `n_steps` from 2150 to 1800 uniformly
  across all four α' cells so each cell still finishes inside 290 s
  and the comparison stays clean.
- **Gradient flow**: non-zero, non-NaN through the cross-layer `h`
  chain on a 4-layer toy net. This catches a flex `score_mod`-captured-
  tensor bug that broke gradient flow in early torch 2.5 builds.
- **Streaming path**: at `T_query = 1` the v1 submission switches to
  explicit-math attention (the score tensor is cheap there). Verify
  the math path produces identical predictions to the flex path on
  a 64-step rollout against a stored reference.

### Fix D — α'-per-layer is one global value

Add an `assert` at construction to verify all `HopfieldCoupledAttention`
modules in the stack share the same `alpha_prime`:

```python
assert len({blk.attn.alpha_prime for blk in self.blocks}) == 1, \
    "α' must be uniform across layers in this experiment"
```

Prevents an accidental per-layer mutation from producing an uninterpretable
result.

### Hypothesis (revised)

- **α' = 0** lands within 0.005 acc of `hopfield_layer` (0.7293) — the
  missing-baseline gap is small, and Hopfield's external memory is
  contributing essentially zero. Energy: ~40–42 kJ (4-layer Muon stack
  without the M=4096 retrieval bank).
- **α' = 0.5** lifts acc by 0.005–0.010 over α' = 0 at matched energy,
  reproducing the Masumura/Taki GPT-2-small improvement on WikiText-103
  (22.87 → 20.70 PPL) at small-scale char-LM.

If both hold: **first energy-attributable Hopfield-mechanism contribution
demonstrated in the portfolio.**

## Success criteria (revised, sweep-aware)

- **Attribution success** (independent of headline result): α' = 0 cell
  completes with non-DQ wallclock and any val acc. The number it produces
  is what was missing.
- **Strong PASS**: α' = 0.5 ≥ 0.735 at energy ≤ 42 kJ (recovers 6-layer
  accuracy under 82% of its energy).
- **Weak PASS**: α' = 0.5 > α' = 0 by ≥ 0.005 acc at matched energy.
- **Refutation**: |α' = 0.5 − α' = 0| < 0.003 with α' = 0.7 also flat —
  published GPT-2 win does not transfer to small char-LM. This is a
  **publishable negative**.

## Failure modes & diagnostics

- **flex_attention API change between Modal image torch versions.** Modal
  image is torch 2.5.1+cu124 per v1 spec. `flex_attention` is documented
  beta in 2.5.x; if the Modal container differs, fall back to the
  explicit-math path (already in the `HopfieldCoupledAttention.forward`
  branch).
- **`h.abs().max()` blows up at high α'.** Diagnostic logged at layer L−1,
  step 1000. Mitigation in v1 spec (LayerNorm on `h`) stands.
- **Per-α' submissions write to overlapping directories.** Use
  `cp -r submissions/mha_alpha05 submissions/mha_alpha{0,03,05,07}` once
  before the sweep so each Modal submission has its own `result.json`.

## Cost

4 Modal runs × ~5 min ≈ $0.40 total.

## References

- **Masumura & Taki 2025** (corrected), "On the Role of Hidden States of
  Modern Hopfield Network in Transformer", arXiv 2511.20698.
- Ramsauer et al. 2020 "Hopfield Networks Is All You Need" arXiv
  2008.02217 — the Hopfield-as-attention identity that MHA generalizes.

## Cross-references

- `submissions/mha_alpha05/submission.py` — the v1 implementation
  (correct mechanism, wrong byline).
- `experiments/gradient_free/experiment_19_hopfield_coupled_attention_mha.md`
  — v1 spec (correct mechanism, wrong byline, wrong sweep order).
- `submissions/modded_nanogpt/submission.py` — 6L baseline.
- `submissions/hopfield_layer/submission.py` — 4L+external-memory PASS
  (0.7293 / 40.2 kJ).
