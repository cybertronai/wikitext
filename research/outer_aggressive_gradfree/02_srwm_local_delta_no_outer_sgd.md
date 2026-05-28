# 02 · Self-Referential Weight Matrix with Local Delta-Rule Slow Weights

## Mechanism

Irie/Schlag/Csordás/Schmidhuber 2022's Self-Referential Weight Matrix (SRWM)
already uses outer-product fast-weight writes and delta-rule reads. The novel
move: **also update the so-called slow weights — q, k, v, β, projection — by
a local rule, not by backprop**.

Specifically, the SRWM W is updated by

    W_t = W_{t-1} + β_t (v_t - W_{t-1} k_t) k_tᵀ

where (q_t, k_t, v_t, β_t) are projections of x_t through *another* SRWM Y.
In the original paper Y is trained by SGD on a global loss. Here Y itself is
also a "fast" weight that updates by the **same delta rule** against a
*self-supervised reconstruction target*: at each token, after producing a
prediction, the actual next byte's embedding is used as v-target and Y is
hit with a delta update along the same outer-product axis.

End result: NO global backprop. NO chain rule across the network. The full
stack is

    layer ℓ:  W^ℓ_t = W^ℓ_{t-1} + β_t (v_t - W^ℓ_{t-1} k_t) k_tᵀ
              Y^ℓ_t = Y^ℓ_{t-1} + α (e(byte_{t+1}) - Y^ℓ_{t-1} φ(x_t)) φ(x_t)ᵀ

where φ is a frozen random feature map (RFF or ELU+1 sum-norm). All matrices
are updated by Hebbian outer products only.

The **prediction head** is also fast-weight: a final readout fast-weight matrix
W^out is updated by the same delta rule against the true next byte after each
emission, and its read produces the 256-d softmax distribution.

## Seed papers

- Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight
  Programmers*, ICML 2021 (arXiv 2102.11174). Establishes the delta-rule
  fast-weight reading of linear attention; v.2 of `hebbian_fw_block`
  implements its sum-norm.
- Irie, Schlag, Csordás, Schmidhuber, *A Modern Self-Referential Weight
  Matrix That Learns to Modify Itself*, ICML 2022 (arXiv 2202.05780). SRWM:
  one matrix that produces its own update rule. The slow/fast distinction
  collapses into a single self-modifying matrix. Currently trained by SGD.
- Schmidhuber, *Reducing the ratio between learning complexity and number
  of time-varying variables in fully recurrent nets*, ICANN 1993. The
  original FWP paper — outer-product updates inside an RNN.
- Yang, Wang, Zhang, Shen, Kim, *Parallelizing Linear Transformers with the
  Delta Rule over Sequence Length*, NeurIPS 2024 (arXiv 2406.06484).
  DeltaNet at 1.3B params beats Mamba — but with backprop. We re-use only the
  WY-Householder chunkwise scan for compute efficiency.

## Why it could work here

- **All updates are outer products.** Compute pattern matches Tensor Cores
  perfectly. Per-token cost is O(d²) for each layer instead of O(d²·T) for
  attention. At d = 384, T = 1024, batch 32, that's ~1.5 × 10^10 flops
  per step — well within A100 budget for 2000+ steps in 300 s.
- **No backward pass means no need to keep activations.** Activation memory
  is O(d) per layer per token (just the current keys/values). Compared to
  the existing hebbian_fw_v2 at ~22 kJ, eliminating backward should give
  another 30–50% energy reduction if the model still trains.
- **Self-supervised target signal is the next-byte embedding itself.**
  No special objective design — every byte gives every layer a target.
- **Streaming-natural:** Y and W are exactly the kind of state that gets
  carried across `observe()` calls. No special "training-vs-eval mode".

## Threshold of plausibility

The hard question: do unsupervised Hebbian writes on q/k/v/β/proj actually
learn anything *useful* at d ≥ 256 on byte text in 2000 update steps, when
backprop normally does this job?

The Storkey-rule capacity argument suggests yes — n/√(2 ln n) ≈ 60 patterns
per dimension cleanly, which at d = 256 is ~15 K linearly-independent
contexts. The Demircigil exponential interaction lifts this further. With
2000 minibatch updates of size 32 × 1024 = 64 K tokens per step, the
"effective dataset" hitting Y is ~10^8 (context, next-byte) tuples — far
more than needed to saturate.

The unknown: whether the *propagated* Hebbian signal through 4 layers of
delta-rule writes carries enough credit to learn deep features. The SRWM
paper used SGD because they couldn't make purely local rules work on
RL/few-shot benchmarks at the time. This experiment is the missing data
point: same architecture, fully Hebbian.

Realistic estimate: 0.55–0.70. If it clears, it is the strongest result in
this portfolio.

## Failure modes

- **Local-rule plateau at 0.30–0.45** — Hebbian outer products do something
  but cannot escape the paradigm-A ceiling because they cannot represent
  task-adapted features that require a non-local credit signal. This is the
  null result.
- **W and Y blow up.** Outer-product writes have no built-in normalization;
  Schlag's sum-norm fix in `hebbian_fw_block_v2` is for W but Y needs the
  same. Mitigation: aggressive RMSNorm + clip W norms after each chunk.
- **Catastrophic interference** during streaming. Each new byte overwrites
  W via the delta rule. With T = 60 K val chars, we run 60 K delta updates
  during inference. If the rule isn't a contraction (i.e., decay < 1), W
  drifts away from train-time. Mitigation: bf16 sum-norm + decay = 0.97
  rather than 1.0.
- **Stochasticity filter:** the readout is naturally a soft distribution
  (softmax over the readout fast-weight matrix output), so this passes.
- **Wall-clock blow-up.** Two sets of fast weights to update per token
  doubles the chunkwise-scan work. At T = 512 and 4 layers, this stays
  within budget; at T = 1024 it may breach 300 s.

What would falsify it: with 4 layers of SRWM at d = 256, T = 512, batch = 32,
n_steps = 2000, val acc ≤ 0.40 → confirmed that Hebbian-only training cannot
learn task-adapted projections at this scale. Reject. (Compare to the
0.37 paradigm-A ceiling — anything north of that is interesting.)

## Smallest first experiment

Build `srwm_local_delta_v1`:

1. **Architecture:** 1 SRWM block (no transformer body at all). Inputs are
   byte embeddings projected by a frozen random Gaussian. d = 256, T = 512,
   chunk_size = 64, batch = 32, n_steps = 1500.
2. **Inside the block:** two fast-weight matrices W and Y, both (B, d, d).
3. **Per chunk:** apply WY-Householder parallel scan (re-use existing code
   from `hebbian_fw_block_v2`) to compute (q, k, v, β) = Y · x for each token,
   then standard delta-rule scan for W. After the chunk, compute the
   self-supervised loss e(byte_{t+1}) - Y · φ(x_t) per token and apply ONE
   delta-rule update to Y per chunk (NOT per token — that would serialize).
4. **No optimizer.** No `loss.backward()`. No SGD parameters at all. The model
   has zero `nn.Parameter` — only buffers.
5. **Readout fast-weight W^out (B, 256, d).** Updated by delta rule against
   the true next byte's one-hot at training; produces softmax(W^out · x_t)
   at inference.
6. **Sanity check:** at step 0, val acc should be unigram (≈ 0.18 for the
   modal byte ' '). At step 100 it should clear unigram. At step 1500 we hope
   to see ≥ 0.50.

Add as a follow-up: **2 layer** version. If 1-layer plateaus at 0.40 and
2-layer reaches 0.55, deeper is interesting; if 2-layer is also 0.40, the
ceiling is hit.

## Memory-movement analysis

Per chunk (size C = 64, d = 256): matmul cost is O(B·C·d²) ≈ 32·64·65 K
≈ 130 M flops per chunk; 8 chunks per sequence → 1 G flops per minibatch
per layer. At 2000 steps × 2 layers = 4 × 10^13 flops total. A100 fp16 peak
is 312 TF → < 1 s of pure compute. Wall-clock will be dominated by
HBM traffic for the W writes (B·d² floats = 32·65 K = 2 MB per layer per
chunk, very write-heavy). **Bandwidth-bound on writes.** Mitigation:
keep W in shared memory / Tensor Core fragments via custom kernel if needed
(but a stock PyTorch impl should also fit in 300 s).

## References

- Schlag et al., ICML 2021: <https://arxiv.org/abs/2102.11174>
- Irie et al., ICML 2022: <https://arxiv.org/abs/2202.05780>
- Yang et al., NeurIPS 2024 (delta-rule WY scan): <https://arxiv.org/abs/2406.06484>
- IDSIA modern-srwm code: <https://github.com/IDSIA/modern-srwm>
