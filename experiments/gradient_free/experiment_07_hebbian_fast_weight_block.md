# Experiment 07: Hebbian Fast-Weight Block (Delta-Rule, Local Plasticity) Inside SGD-Trained Transformer

## Hypothesis
Replacing one transformer block with a Schlag-Schmidhuber fast-weight layer — where the fast-weight matrix W is updated by the **delta rule** (write the *residual* between the new value and the retrieved old value, gated by a learned β) and **never receives backprop gradients on those updates** — within an otherwise SGD-trained 5-layer stack, matches val acc ≥ 0.72 at energy ≤ 42 kJ. Tests whether an explicitly local-Hebbian component can play the same paradigm-B role as Hopfield.

## Motivation
The Schlag-Schmidhuber 2021 identity (arXiv 2102.11174): a linear transformer's attention output equals a query against a fast-weight matrix W_fast(t) updated by outer-product writes. The 2021 paper's contribution over Katharopoulos 2020 was replacing the **purely additive** update `W_t = W_{t-1} + v_t ⊗ φ(k_t)` with a **delta rule**:

```
v_bar_t = W_{t-1} φ(k_t)                        # retrieve old value at this key
W_t = W_{t-1} + β_t (v_t - v_bar_t) ⊗ φ(k_t)    # write the residual, gated by β_t
```

This fixes additive-without-forgetting blow-up at finite memory capacity. DeltaNet (Yang et al. 2024, arXiv 2406.06484) is the parallelized-over-sequence reformulation of this same rule.

**Both** Schlag 2021 and DeltaNet differentiate through the W recurrence at training time. The **gradient-free variant** introduced here treats the W updates as a non-differentiable in-place hidden state — a true local Hebbian rule with no backprop into W. Only the q, k, v projections and the β gate (and the surrounding net) are SGD-trained. This is the experimental contribution; it is **not** a re-implementation of Schlag 2021.

The Hopfield win pattern says *frozen retrieval* works. This pattern says *online-written, gradient-free retrieval* might work. Both fit paradigm-B.

## Method
One block replaced with a `HebbianFastWeightBlock` implementing the Schlag delta rule with a no-grad W:

```python
class HebbianFastWeightBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.beta = Linear(dim, 1)            # per-step learned write gate (Schlag §3.2)
        self.proj = Linear(dim, dim)

    def forward(self, x):  # x: (B, T, d)
        # Schlag recommends L2-normalized q,k over the elu+1 feature map
        # used by Katharopoulos; the sum-of-keys denominator (Katharopoulos)
        # is dropped — L2-normalized keys make it unnecessary and harmful in
        # combination with the delta rule.
        q = F.normalize(self.q(x), dim=-1, eps=1e-6)
        k = F.normalize(self.k(x), dim=-1, eps=1e-6)
        v = self.v(x)
        beta = torch.sigmoid(self.beta(x)).squeeze(-1)   # (B, T)

        B, T, d = x.shape
        W = torch.zeros(B, d, d, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            # READ: gradient flows through q_t and (via no-grad W) only through
            # the q/k/v/beta projections of *prior* steps via the W state's
            # detached values — i.e., not at all. W is a non-differentiable
            # hidden state. Output gradient flows back through q_t only.
            o_t = torch.einsum('bde,be->bd', W, q[:, t])
            outs.append(o_t)

            # WRITE (delta rule, gradient-free): retrieve old value at k_t,
            # write the residual scaled by beta_t. This is Schlag-Schmidhuber
            # 2021 Eq. ~ "delta rule update", NOT Katharopoulos additive update.
            with torch.no_grad():
                v_bar = torch.einsum('bde,be->bd', W, k[:, t])      # retrieve
                residual = v[:, t] - v_bar                           # residual
                W = W + beta[:, t, None, None] * torch.einsum(
                    'bd,be->bde', residual, k[:, t])                 # write

        out = torch.stack(outs, dim=1)  # (B, T, d)
        return self.proj(out)
```

The block thus has **trainable** q/k/v/β/proj projections but a **gradient-free fast-weight state W**. Note: there is no sum-of-keys denominator — Schlag argued (and DeltaNet confirms) that with L2-normalized keys the explicit normalizer is unnecessary and degrades the delta rule.

## Memory-Movement Analysis
- W is (B, d, d) = 32·384·384·2 B = 9.4 MB. Lives in L2. The T-step causal loop is the killer: T=1024 sequential einsums (with the delta rule adding one extra einsum per step for v_bar retrieval).
- **THIS IS A TIME-CAP RISK.** The sequential causal scan is exactly the FAVOR+ failure pattern that DQ'd exp 03; the delta rule's read-then-write structure does NOT trivially fuse (unlike pure additive), making the parallel-scan mitigation harder.
- Mitigation 1: naive cumsum on outer products is impossible here — the delta rule's `v_bar = W_{t-1} k_t` makes the recurrence non-linear in the writes. (Additive linear-attention's cumsum trick does NOT apply.)
- Mitigation 2: block-causal chunks of size 64 with within-chunk delta scan and cross-chunk carry. Reduces serial steps from T=1024 to T/64=16.
- Mitigation 3: DeltaNet's WY-Householder reparameterization (Yang et al. 2024, arXiv 2406.06484) parallelizes the delta rule over sequence length without materializing W explicitly. If chunked causal scan DQs on time, this is the canonical fallback — but note it's a substantial rewrite.
- Mitigation 4: use this only at block 4 or 5 (after most layers), and at reduced inner T_chunk to bound serial cost.
- Per-step expected compute: ~2× one attention layer at d² cost (read + write each cost d² per step). Serial scan structure burns wall-clock.
- **Honest assessment**: this is the experiment most likely to DQ on time. Include it but rank as medium-risk. If chunked scan DQs by < 30 s, DeltaNet reparameterization is the second attempt.

## Setup
- 5-layer modded_nanogpt body + 1 HebbianFastWeightBlock at block index 4 (last block).
- Mitigation 2 (chunked causal scan, chunk=64) implemented from the start.
- Baseline: `modded_nanogpt` (6L, 51.7 kJ / 0.7374).

## Procedure
1. `cp -r submissions/modded_nanogpt submissions/hebbian_fw_block`
2. Implement `HebbianFastWeightBlock` with chunked causal scan.
3. Replace `self.blocks[5]` with `HebbianFastWeightBlock`. Adjust the rest to handle the block returning no KV cache (it has its own state, the fast-weight matrix W, which during eval must be carried across `observe()` calls).
4. For inference / streaming `predict()`: W is part of the model state and accumulates as bytes are observed. Reset at `reset()`.
5. Train.

## Success Criteria
- **Strong**: val ≥ 0.73, energy ≤ 42 kJ.
- **Pass**: val ≥ 0.70, time ≤ 290 s.
- **Likely failure mode**: time-cap DQ — interpret as confirming that gradient-free fast-weight scans share Performer's wall-clock failure mode at T=1024.

## Failure Modes & Diagnostics
- Time DQ at chunk=64: try chunk=128 (less serial overhead, more per-chunk compute). Or move the block earlier where T_effective is shorter (no — T_effective is constant in a transformer).
- W blows up numerically: log ‖W‖_F per step; if growing unboundedly, add explicit decay W ← γ W with γ ≈ 0.999 (Ba 2016 fast weights with decay).
- Per-token streaming inference loses access to training-time W: this block's per-byte `observe()` must apply exactly one delta-rule write step (retrieve v_bar, write residual; cheap, no scan).

## Estimated Cost
1 Modal run, ~10 min, ~$0.40. High DQ probability on time — budget a second run if first DQ's by 30 s or less.

## References
- Schlag, Irie, Schmidhuber 2021 "Linear Transformers Are Secretly Fast Weight Programmers" (ICML 2021, arXiv 2102.11174) — source of the delta-rule update used above.
- Katharopoulos, Vyas, Pappas, Fleuret 2020 "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (ICML 2020) — the *additive* linear-attention update Schlag's delta rule replaces.
- Yang, Wang, Zhang, Shen, Yu, Kim 2024 "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" / DeltaNet (NeurIPS 2024, arXiv 2406.06484) — parallel-over-sequence reformulation of the same delta rule; fallback if the chunked scan DQs on time.
- Ba et al. 2016 "Using Fast Weights to Attend to the Recent Past" (NeurIPS)
- Schmidhuber 1992 (original fast-weight programmer)
- Note: this design intentionally differs from `delta-net` style (which IS differentiated through) — the no-grad wrapper around fast-weight writes is the gradient-free contract and is the experimental contribution of this submission.
