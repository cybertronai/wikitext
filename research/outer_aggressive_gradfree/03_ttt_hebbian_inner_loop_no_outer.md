# 03 · Test-Time Training Layers with a Hebbian Inner Loop and Frozen Outer Features

## Mechanism

Sun, Li, Patil, Wang, Zhang, Guestrin et al. 2024 (*Learning to (Learn at
Test Time)*, arXiv 2407.04620) introduce TTT layers whose hidden state IS a
learnable model — specifically, the weights of a linear projection (TTT-Linear)
or 2-layer MLP (TTT-MLP) that gets updated by SGD on a self-supervised
reconstruction loss *while predicting test tokens*. The TTT-Linear update rule

    W_{t} = W_{t-1} - η · ∇_W L_SSL(W_{t-1}; x_t, x_t)

is mathematically a gradient step on the per-token loss, which for an MSE
self-supervised objective collapses to a **delta-rule outer product**.

In their paper, the *outer* parameters — the projections that produce
(input_t, target_t) and the readout — are trained by standard backprop on
the language modeling loss. **This experiment removes that.** The outer
projections become **frozen random feature maps** (RFF or learned-once
ELU+1 sum-norm). The TTT inner-loop SGD step is interpreted as a Hebbian
delta-rule write. There is no global backward pass at any point.

Concrete computation at step t:

    z_t = φ_rand(x_t)                     # frozen random feature map
    pred_t = W_{t-1} z_t                  # READ: linear projection
    target_t = ψ_rand(byte_t)              # frozen random target embedding
    W_t = W_{t-1} + η (target_t - pred_t) z_tᵀ   # WRITE: delta rule

Readout for next-byte: a frozen random matrix R: (256, d) projects pred_t
back to 256-d logits. Train on the cross-entropy gradient ONLY of R — and
even R can be replaced by a closed-form ridge solve on the (W_t z_t,
onehot(byte_{t+1})) pairs sampled during training.

This is the most aggressive simplification possible: every "learning" is a
single outer product. No deep stack, no chunked scan, no parameters that
move under any optimizer state. The model IS the in-context fast-weight
matrix W.

## Seed papers

- Sun, Liu, Wang, Patil, Zhang, Guestrin et al., *Learning to (Learn at Test
  Time): RNNs with Expressive Hidden States*, arXiv 2407.04620 (NeurIPS 2024).
  Establishes TTT-Linear / TTT-MLP. Shows context utilization rivaling
  Transformers with constant per-token latency.
- Rahimi & Recht, *Random Features for Large-Scale Kernel Machines*, NIPS
  2007. Random feature map construction used as the frozen φ here.
- Schlag, Irie, Schmidhuber, ICML 2021 (arXiv 2102.11174). Establishes the
  Hebbian/delta-rule = SGD-on-MSE equivalence used to reinterpret the TTT-Linear
  step as a Hebbian outer product.

## Why it could work here

- **The TTT-Linear inner loop, written as a Hebbian outer product, is
  exactly the kind of operation A100 Tensor Cores are designed for.** Per
  token: one (d, d) read, one (d, d) write — single op.
- **No optimizer state.** No Adam moments. Just W and the latest x_t.
- **Constant per-token latency** at inference (TTT's headline claim).
  Perfect fit for the streaming `CharModel.observe()` contract.
- **The TTT paper already showed scaling laws comparable to Transformers
  at WikiText.** That's with backprop on the outer projections. The
  experiment is whether *frozen random* outer projections give up most or
  little of that performance — which is the central open question this
  portfolio asks.

## Threshold of plausibility

The fundamental question: how much of TTT-Linear's performance comes from
(a) the fast inner-loop adaptation vs (b) the well-trained projections?

If most of the win is (a), this could clear 0.70: the inner fast-weight
adaptation captures local context in a way attention doesn't, and the
frozen projections are only used to provide a meaningful similarity
function. RFF features have provable kernel-approximation guarantees, so
the similarity function is "Gaussian kernel on byte-window embeddings" —
not great but not zero.

If most of the win is (b), the experiment exposes paradigm-A's ceiling
under a new name: 0.35–0.45 plateau.

The TTT paper's own ablation (replacing q/k/v projections with identity)
gave a ~30% perplexity hit but not catastrophic failure on a 125M-param
model. Extrapolating to byte-level and to a much smaller model: expect
0.50–0.60. Whether it crosses 0.70 is genuinely uncertain.

This experiment is therefore the **most direct test of "is outer-aggressive
Hebbian fast-weight enough?"** in the entire portfolio. Highest expected
information per joule.

## Failure modes

- **Frozen RFF projections cannot represent byte-level char-LM context
  well enough.** Most likely outcome at 0.35–0.45.
- **One single (d, d) fast-weight matrix has insufficient capacity** for
  the full conditional distribution of byte ∣ context. Multi-layer TTT
  (TTT-MLP) is heavier but might be needed. Mitigation: stack 2–4 TTT-Linear
  layers with frozen-random projections between them.
- **Streaming W vs train-time W:** during training we sample independent
  64-byte windows; at inference we stream sequentially. The W state evolves
  differently. Mitigation: train with fw_state carried across mini-windows
  within an epoch.
- **The closed-form readout solve has a stochasticity problem** if it sees
  multiple bytes for the same prediction state. Use cross-entropy minimizing
  ridge regression (i.e., regress onto one-hot targets with soft-max
  inversion) rather than vanilla L2 ridge. This converts the readout into
  a soft-distribution problem and dodges the WTA failure mode.
- **bf16 outer-product accumulation precision.** Schlag warned. Use fp32
  for W; outer products in bf16 → cast → accumulate in fp32.

What would falsify it: TTT-Linear with frozen random RFF projections and 4
stacked layers, d = 256, on full WikiText-103, fails to beat 0.45 →
confirmed that the win in Sun 2024 came from the projections, not the inner
loop. Reject in current form.

## Smallest first experiment

`ttt_hebbian_frozen_v1`:

1. **Outer-projection:** frozen RFF feature map `φ: byte_window → R^d`,
   d = 512. Use the byte window's last 64 bytes' one-hot vectors,
   concatenated, projected by a random Gaussian RFF — `cos(Rx + b)` style.
2. **Inner fast-weight W: (B, d, d).** Zeros at start of each minibatch
   sample.
3. **Per token t in a chunk:** compute z_t = φ(window_t),
   target_t = embed_random(byte_{t+1}) ∈ R^d, then apply
   `W_t = W_{t-1} + η · (target_t - W_{t-1} z_t) z_tᵀ` under no_grad.
   Use the WY-Householder parallel scan for chunkwise parallelism.
   No `nn.Parameter` involved. η is a constant.
4. **Readout:** a frozen random matrix `R: (256, d)`. Logit at step t is
   `R · (W_{t-1} z_t)`. *Or*: train R once at end of training via ridge on
   collected (W·z, one_hot(byte)) pairs from the last 100 K tokens. **Both
   variants gradient-free.**
5. **Single config sweep:** η ∈ {0.01, 0.1, 1.0}, ctx_len ∈ {32, 64, 128},
   d ∈ {256, 512}, n_layers ∈ {1, 2}. 6 submissions max.

For multi-layer: stack via x^(ℓ+1) = φ_ℓ(W^ℓ z^ℓ_t) where φ_ℓ is a frozen
random feature map between layers. Fast weights at each layer are
independent and updated locally.

## Memory-movement analysis

Per token: 2 × (d, d) bmm read + 1 (d, d) outer-product write =
3·d²·B flops per token. At d = 512, B = 32, T = 1024, n_steps = 1500
→ 4 × 10^13 flops total. A100 fp16 peak ~312 TF → < 1 s pure compute.
HBM traffic: W is hot (B·d² · 4 bytes = 32 MB per layer). Reads and writes
hit L2/HBM repeatedly; this will be the bottleneck. **Bandwidth-bound on W.**
Mitigation: process longer chunks (chunk_size = 128) so each chunk's writes
amortize one HBM round-trip.

Expected wall-clock: 20–40 s for a 1-layer training; 60–120 s for 4-layer.
Energy: should land well below `hebbian_fw_v2`'s 22 kJ because there is no
backward pass and no optimizer state. Ballpark 8–15 kJ if it converges.

## References

- Sun et al., NeurIPS 2024 (TTT): <https://arxiv.org/abs/2407.04620>
- Rahimi & Recht, NIPS 2007 (RFF): NeurIPS proceedings; see also
  <https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf>
- Schlag et al., ICML 2021 (delta=SGD-MSE equivalence): <https://arxiv.org/abs/2102.11174>
