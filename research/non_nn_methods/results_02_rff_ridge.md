# Results — Spec 02: Random Fourier Features + closed-form ridge LM

**Status.** Disqualified (val char-acc 0.3641 < 0.70 floor). Real Modal A100-80GB PCIe run.

## Headline numbers

| Metric                | Value          |
|-----------------------|---------------:|
| val char-acc (60K)    |  **0.3641**    |
| training energy       |  **2,565 J**   |
| training wall-clock   |  11.8 s        |
| GPU                   |  A100 80GB PCIe |
| Modal run             |  ap-O9XK75RR0hxTuuOlE5GmAg |
| Baseline comparison   |  20× lower energy than modded_nanogpt (51.7 kJ) — but DQ, so off-leaderboard |
| Spec energy estimate  |  3–8 kJ → **beat the lower bound** at 2.6 kJ |

## Config

K=16 byte context, frozen random per-byte embedding d_byte=64 → d_in=1024,
RFF m=8192, gamma=0.3, lambda=1e-2. Streamed N=8,000,000 train positions in
65,536-position chunks. One Cholesky on the 8192×8192 normal-equations matrix.
Single seed, no hyperparameter sweep — per task instructions, no design iteration.

## What worked

- Compute budget was a non-issue. 11.8 s wall (~4 % of 300 s budget) — Phi
  streaming, Phi^T Phi, and Cholesky all hit Tensor Cores cleanly in bf16
  with fp32 accumulator. Cholesky on m=8192 took 0.20 s.
- Throughput: 754,000 train positions/s sustained including I/O. Roofline
  predicted ~5 s for the dominant Phi^T Phi at m=4096; we ran at m=8192
  (4× the FLOPs) in ~11 s — consistent with strong compute-bound behavior.
- Submission infra path was clean; only fix needed was dropping `numpy`
  (the registry image does not ship it — pure-torch is sufficient).

## What surprised me

- Accuracy ceiling is far below what the spec's "0.50–0.65 plausible" range
  suggested. 0.36 is barely above bigram-class performance and well below
  the PPM baseline (0.63 at 633 J). The RFF readout learned *something*
  (the val-char unigram floor is ~0.18 for English text), but K=16 of
  byte-embed features through an RBF map is apparently a very poor
  representation of the conditional distribution.
- Cholesky succeeded without numerical issues at lambda=1e-2 — no jitter
  retry. Conditioning of the m=8192 Gram was not the bottleneck.

## Did it beat the spec's energy estimate?

Yes — 2.6 kJ < 3 kJ lower bound, despite running m=8192 (2× spec m=4096)
and N=8M (1.6× spec N=5M). The closed-form solve really did land
strongly compute-bound. But the energy efficiency is moot because the
accuracy is nowhere near the gate.

## Was the closed-form solve compute-bound?

Yes, by the FLOP/sec measurement.
- Phi^T Phi: 8M × 8192² × 2 ≈ 1.07e15 bf16 FLOPs → 11.8 s × ~250 W
  → ~91 TFLOPs sustained. A100 bf16 peak is 312 TFLOPs; we hit ~29 %,
  which is consistent with a streaming gemm at this aspect ratio
  (skinny B=64K × m=8192). Plenty of headroom; m=16384 or m=32768
  would still fit in budget.
- Cholesky at 0.20 s is negligible (~1.7 % of train wall).

## Verdict

Paradigm-A "kernel-machine-replaces-LM" fails on byte-level WikiText
even with generous m=8192 and full 8M training positions. This matches
`finding_kernel_two_paradigms.md`'s prediction that the kernel-LM
scaling story is weak — the RKHS spanned by 8192 random Gaussian-kernel
features on a 16-byte window is too restricted a hypothesis class to
encode the conditional structure of natural language past trigram-ish
range. Calibrated, publishable negative result.

## Worth a second-round investment?

Not on the energy axis — the accuracy gap to 0.70 is too large for
hyperparameter tuning alone (m, gamma, K, lambda) to close. A
generous reading: the closed-form readout pattern *does* shine when
paired with a richer feature map, which is exactly what spec_03
(polynomial / TensorSketch) and spec_04 (Falkon Nyström, data-adaptive
landmarks) propose. The RFF-with-frozen-random-features ceiling
measured here is the right baseline against which to judge those.

## Artifacts

- `submissions/rff_ridge_v1/submission.py` — implementation
- `submissions/rff_ridge_v1/result.json` — Modal run output
- `submissions/rff_ridge_v1/run.log` — full Modal log
- `submissions/rff_ridge_v1/nvml.json` — NVML probe summary

## Review (post-hoc audit)

**Validity for discarding RFF + closed-form ridge on byte-level char-LM:** *Valid.*

**Core limitations:**
- **Budget was actually saturated on the right axis** (compute-bound, ~29 % of A100 peak) and N=8 M, m=8 192 are roomy for the claim. Pham-Pagh-style variance scaling implies m would have to grow exponentially in K to meaningfully change the ceiling, so the m-sweep is a real diminishing-returns argument and not a cop-out.
- **K=16 byte context is short** — the experiment bounds RFF-ridge in the short-context regime, not RFF-ridge in general. A K=64 or K=128 variant on the same compute budget is a genuinely different question and would change the verdict scope to "RFF-ridge cannot encode language structure at *any* K we can afford here".
- **Frozen random per-byte embedding (d_byte = 64).** The kernel is being asked to do all representation work over a non-learned input map. A learned 16-byte embedding (paradigm B) is a different method — and is correctly punted to the Falkon spec.

**Verdict:** Adequate to discard "RFF over frozen byte embeddings + Gaussian-kernel closed-form ridge at K ≤ 16" on this benchmark. Should not be cited as a general statement about RFF for LM.
