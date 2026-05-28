# Results 03 — Polynomial / TensorSketch + closed-form ridge

**Status: DQ — under-capacity at the proposed configuration.**

## Numbers

| | value |
|-|-|
| val char-acc          | **0.3085** |
| training energy (NVML)| **2,522 J** |
| training wall-clock   | 14.7 s |
| GPU                   | A100 80GB PCIe |
| Config                | K=16, m=8192, p=3, λ=1.0, N=5×10⁶ positions |

Floor for leaderboard is 0.70 → disqualified. Threshold not reached, so
the energy figure is reported as the partial-budget measurement of what
the method actually consumed before solve.

## Comparison

- **Baseline modded_nanogpt:** 51,704 J, 0.7374 acc → poly-TS used 20×
  less energy but landed 0.43 acc below the gate.
- **Spec predicted:** 30–90 s wall, 5–15 kJ energy, char-acc 0.55–0.70.
  We blew past the energy estimate (came in at half the lower end at
  2.5 kJ) and significantly *under-shot* the accuracy estimate.
- **Closest non-NN reference:** GPU-n-gram (W31 k11) hits 0.7050 at
  1,333 J. Pure counting Kneser-Ney crushes degree-3 polynomial ridge
  on this task.

## Implementation notes

- TensorSketch built as expected: P=3 independent CountSketches on the
  K·256-dim positional one-hot, then rFFT-product-irFFT chain.
- Streamed Gram accumulation in fp32 with TF32 matmul. 5 M positions
  processed in 13.6 s sustained → 366 K pos/s. Cholesky on m=8192
  finished in 0.1 s.
- Per-char predict: 3 small FFTs of length 8192 + a 1×8192 @ 8192×256
  matmul → ~3.7 K char/s eval throughput. Eval wall 16 s, not budgeted.

## Did it beat the spec's energy estimate?

Yes — easily. 2.5 kJ vs the 5–15 kJ predicted, **6× under** the
midpoint. The spec underestimated both the rFFT throughput (length-8192
batched FFTs are ~1 µs/sample on A100) and the speed of streamed Gram
accumulation in TF32.

## Arithmetic intensity — did it land compute-bound?

Effectively no. The whole training pass took 14.7 s of which ~3 s is
overhead/data-load and ~10 s is steady-state. With m=8192 fp32 the
Phi.T@Phi step *should* dominate at ~5×10¹¹ FLOPs/chunk × 1.2 K chunks
≈ 6×10¹⁴ FLOPs → ~4 s on A100 TF32 (≈156 TF/s). Observed actual matmul
time is consistent. But sketch-construction (scatter_add for the
CountSketches + rFFT) is bandwidth-bound on the (P, B, M) scratch
buffer, so we're paying real time there too. Net: matmul-bound for the
Gram, BW-bound for the sketch — roofline-mixed, not the "deeply
compute-bound" the spec claimed.

## What surprised me

The accuracy. 0.31 char-acc at m=8192, p=3, K=16 is *below* even a
unigram floor on most reasonable embeddings. Several plausible
explanations:

1. **Variance of TensorSketch with only K=16 active features.** With
   sparse input and P=3 hashes onto m=8192, each rFFT product is
   dominated by sign-aliasing noise; the signal-to-noise ratio of phi
   is poor for short contexts. Pham-Pagh's theory assumes dense or
   high-density input.
2. **Positional one-hot kills cross-position interactions** in a way
   that hurts degree-3 polynomial expansion. The degree-3 monomials
   (i,j,k) at distinct positions are exactly the n-gram features you'd
   want, but they're spread across a 4096³ ≈ 7×10¹⁰ space and
   m=8192 hashes them densely.
3. **Ridge head onto 256 classes** is a hard target with this feature
   distribution; the cross-entropy / one-hot regression substitution
   is known to give poor classifiers on long-tail discrete outputs.

## Worth a second-round investment?

Marginal. The compute cost is so low (2.5 kJ) that it's tempting to
sweep (p∈{2,4}, m∈{16384, 32768}), but Pham-Pagh's variance bound
suggests m needs to scale like d^p/ε² for fidelity — at p=3 and
d=4096 that's effectively impossible. The honest read is that
*explicit* polynomial features (no sketch) at p=2 with dense input
embedding would be a fairer test — that's a different method, see
spec_04 (Falkon Nyström) or the krr_ngram sibling submission.

Bottom line: clean negative. Polynomial TensorSketch on byte-positional
one-hots does not clear the char-LM floor; the kernel-machine-replaces-
model paradigm at this scale is in the "negative result" column.

## Review (post-hoc audit)

**Validity for discarding polynomial-TensorSketch + ridge on char-LM:** *Valid for the tested configuration; narrow scope.*

**Core limitations:**
- **Positional-one-hot input map is hostile to TensorSketch.** As the writeup itself flags, Pham-Pagh's variance bound assumes dense / high-density inputs; on a K=16 byte-positional one-hot the sketch signal-to-noise is poor at p=3. A dense byte embedding + p=2 polynomial expansion (no sketch) is a different and untested method.
- **Ridge head onto 256-class one-hot targets** has the same conditional-mean → marginal-mode failure mode as `result_07` (Nyström). Quadratic loss on a high-entropy discrete target is structurally weak; the experiment cannot distinguish "polynomial features are too narrow" from "MSE-on-one-hot is too weak".
- **Compute was not budget-saturated** (14.7 s of 300 s, 2.5 kJ of ~50 kJ). m=16 384 / 32 768 would not change the verdict on accuracy (variance bound argues against), but the spec does not justify *why* the budget was left on the floor.

**Verdict:** The "kernel-machine-replaces-the-LM" paradigm at byte/one-hot/p=3 is fairly discarded. The Pham-Pagh argument generalises so the verdict can be read as "no degree-3 TensorSketch over sparse byte inputs will clear 0.70". Does not bound polynomial features with dense embeddings or higher degrees with adaptive sketches.
