# 01 · Krotov Dense Associative Memory with Energy-Descent Inference

## Mechanism

Replace the entire LM with a single Krotov-Hopfield Dense Associative Memory
(DAM). Storage is by Hebbian outer products; retrieval is by an energy
descent over the visible state. The DAM stores N (context-vector,
next-byte-onehot) pairs as outer products `Ξ = Σ_i ξ_i ξ_iᵀ` (with the
exponential interaction function from Demircigil 2017 / Ramsauer 2020 making
storage capacity exponential in `d`).

Prediction: given a streaming byte context, encode it to a query vector `q`
using a *frozen* random projection of a sliding window (no learned encoder).
Run T_descent ≤ 3 energy-descent steps on the DAM energy

    E(σ) = - lse(β · Ξ σ) + (1/2) σᵀσ + ...                  (Ramsauer 2020)

clamped on the context bytes, free on the 256-d "next-byte slot". Read out
the next-byte distribution from the converged free coordinates via softmax.
No gradient anywhere — both writes and the energy descent are differentiable
in principle, but we never compute that derivative; both reads and writes
are local outer-product / softmax operations.

This is *radically* simpler than the existing `hopfield_layer` submission:
no transformer body, no projections, no learned anything. The DAM is the
model.

## Seed papers

- Krotov & Hopfield, *Dense Associative Memory for Pattern Recognition*,
  NeurIPS 2016 (arXiv 1606.01164). Polynomial / rectified-polynomial
  interaction lifts capacity from O(n/2 log n) to O(n^k).
- Demircigil et al., *On a Model of Associative Memory with Huge Storage
  Capacity*, J. Stat. Phys. 2017 (arXiv 1702.01929). Exponential
  interaction → exponential capacity in n.
- Ramsauer et al., *Hopfield Networks Is All You Need*, ICLR 2021
  (arXiv 2008.02217). Continuous DAM with attention-equivalent single-update
  retrieval.
- Krotov, *A new frontier for Hopfield networks*, Nature Reviews Phys. 2023.
  Survey of DAM as a unifying framework; explicitly proposes DAM as a model
  family separate from "attention + DAM hybrid".

## Why it could work here

- **No SGD anywhere.** Writes are Hebbian outer products over the train
  stream, executed once. Reads are softmax-attention against the stored bank.
- **Linear in N for writes, O(M·d) per inference step.** With N = 540 MB
  and a sub-sampled bank M ≈ 16 K patterns at d = 256, the entire write phase
  is one bf16 matmul (`(M, d_ctx) @ (d_ctx, d_out)` outer product accumulation).
  Highly arithmetic-intense; A100 should run the whole train phase in seconds.
- **Memory profile is essentially "the data".** No optimiser state, no
  activations stored across layers. Live RAM ≈ Ξ ≈ M·d_ctx·d_out floats.
- **Streaming-natural.** `observe()` just refreshes the sliding window encoding;
  `predict()` is one (or T_descent) softmax(βqᵀΞ).

## Threshold of plausibility

To clear 0.70 char-acc, the Krotov DAM needs to encode "context → modal
next-byte" with enough specificity that:

- The encoder (random projection of `ctx_len = 256` bytes) maps semantically
  similar contexts to nearby query vectors. **This is where the design is
  fragile** — random projections of byte sequences have no language structure.
  A learned encoder would put us right back to needing backprop somewhere.
- The DAM has enough capacity (β large enough, M large enough) that close
  matches return the modal byte rather than a memorized byte. Demircigil
  capacity scales as O(exp(d)), so this is genuinely plausible at d ≥ 128.
- The fp32 outer-product accumulation over M patterns is numerically stable.
  Schlag warned about exactly this in the v1 → v2 transition.

The benchmark hopes implicit: a context window of 256 bytes carries
~6 bits/byte entropy ≈ 1500 bits = exp(1500) raw possibilities, but the modal
next-byte conditional distribution is concentrated. If a *random* projection
of context preserves modal-byte-cluster structure with enough fidelity, DAM
will recover it. Most likely scenario: clears 0.45–0.55 but not 0.70.

This is therefore a **capability demo** (per the research framing memory).
If it does clear 0.70 it is the simplest joules-positive submission in the
benchmark's history.

## Failure modes

- **Random projection has no language structure** → DAM stores noise around
  modal bytes → recall returns near-uniform → bumps off the unigram floor
  but stalls at 0.30. This is the most likely failure.
- **Stochasticity filter:** the DAM's natural read mode is "return the
  stored pattern matching the query best", which is a hard WTA over byte
  ids. Must use *soft* read (Ramsauer 2020 high-β softmax, not single-step
  Hopfield argmax) to escape the NBB failure mode documented in
  `project_nbb_diagnostic.md`. Hard-decision Krotov DAM = NBB-class dead.
- **β tuning is the only free knob,** and the original Demircigil result
  requires β = O(log M). Wrong β → either underfitting (uniform output) or
  catastrophic interference (single nearest neighbour wins).
- **Energy descent might not converge in 3 steps** when β is too high.
  Mitigation: cap T_descent and accept the suboptimal soft read.
- **Modal-byte capture fails on rare contexts** — DAM stores ALL examples,
  so a context that appeared once with byte 'q' and a thousand times with
  byte ' ' may still pull strongly to 'q' depending on β. The Ramsauer
  high-β regime acts as nearest neighbour, which is exactly the wrong mode
  here. Want low-β (≈ 1/√d).

What would falsify it: with M = 65 K, d_ctx = 512, β ∈ {0.5/√d, 1/√d, 2/√d,
4/√d}, the BEST configuration ≤ 0.45 char-acc → confirmed paradigm-A
representation ceiling for energy-based methods. Reject.

## Smallest first experiment

Build a "tiny DAM" submission:

1. **Encoder:** random Gaussian matrix `R: (256·ctx_len, d)` with `d = 256`,
   `ctx_len = 64`. Encode a byte window by one-hot embedding the bytes,
   flattening, multiplying by R, RMSNorm-ing. **Frozen.** No gradient.
2. **Memory build (one shot):** sample M = 16 K (window, next_byte) pairs
   from the 540 MB train stream. Compute query vectors q_i = R · onehot(win_i)
   and target vectors t_i = onehot(next_byte_i) ∈ R^256 (treated as
   pattern in the "next-byte half" of the DAM). Accumulate
   `Ξ = (1/M) Σ_i (q_i ⊕ t_i)(q_i ⊕ t_i)ᵀ` as a (d+256, d+256) matrix. Store.
3. **Inference (`predict`):** for current 64-byte window, compute query
   `q ∈ R^d`, append a free zero-vector in the next-byte coordinates,
   compute scores `s = β · Ξ_top · q` (only the (256, d) cross-block of Ξ;
   no fix-point iteration needed for the soft Hopfield read), softmax → 256-d
   distribution. Done.
4. **Sweep β ∈ {0.25, 0.5, 1, 2, 4} / √d** and `ctx_len ∈ {32, 64, 128, 256}`,
   one submission each.

Streaming wrapper: maintain a deque of the last `ctx_len` bytes, recompute
`q` on every `observe()`. No state to carry. No reset hazard. No backprop.

Expected total submission count: 1–5. Expected wall-clock: ≤ 30 s
(everything is one outer-product accumulation + one matmul per byte at
inference time).

## Memory-movement analysis

Train: one pass over `M·ctx_len` bytes for encoding (`M·d·ctx_len·256` flops
≈ 10^10), one outer-product accumulation `M·(d+256)²` ≈ 10^9 elements
matmul → A100 Tensor Cores will saturate. **Compute-bound on the train side.**
Inference: per byte, one (256, d) matvec → 10^4 flops × 60 K bytes = 6·10^8
flops total. **Trivially fast.**
Param count is dominated by Ξ ≈ (d+256)² ≈ 263 K floats ≈ 1 MB. **Tiny.**

## References

- Krotov & Hopfield, NeurIPS 2016: <https://arxiv.org/abs/1606.01164>
- Demircigil et al., J. Stat. Phys. 2017: <https://arxiv.org/abs/1702.01929>
- Ramsauer et al., ICLR 2021: <https://arxiv.org/abs/2008.02217>
- Krotov, Nature Rev. Phys. 2023 survey: <https://arxiv.org/abs/2306.03209>
