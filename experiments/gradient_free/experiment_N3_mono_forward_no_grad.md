# Experiment N3 — Mono-Forward all-the-way-down (closed-form, zero-SGD)

## Strict ablation from `mono_forward_v2`

`mono_forward_v2` (0.7346 / 46.2 kJ) trains each transformer block by CE
on a *per-block probe head*; the block weights still see SGD via the
probe-head gradient. N3 removes **all** backprop and tests whether the
layer-local byte-target supervision alone can carry depth.

## Algorithm (one paragraph)

A stack of `L` "blocks". Each block ℓ has:
- a **fixed random featurizer** `φ_ℓ : R^d_in → R^d_feat` (random projection of a
  sliding byte-context window, then ReLU);
- a **closed-form ridge classifier head** `R_ℓ : R^d_feat → R^256` fit by
  Cholesky on `(Φ_ℓᵀ Φ_ℓ + λI) W = Φ_ℓᵀ Y_ℓ`, where `Y_ℓ` is the *residual
  target* — the next-byte one-hot minus the cumulative logits from blocks
  0..ℓ-1, projected back to one-hot logit space.

That makes the stack **gradient-boosted ridge readouts** over fixed
random nonlinear features of a byte-context window. Inference logits at a
position are `sum_ℓ R_ℓ(φ_ℓ(x))`. We pass log-prob residuals between layers,
which is what makes the depth meaningful: each new layer fits whatever the
previous layers got wrong. The accumulator across layers escapes the
Paradigm-A ≈0.37 ceiling for random-feature + linear-readout: that ceiling
applies to a *single* readout against the raw target, not to a stack of
readouts against successive residuals.

## Why this is a strict ablation

- Same per-block-CE-on-byte-target structure as mono_forward_v2.
- Probe head's SGD is replaced by **closed-form ridge** (Cholesky on
  `Φᵀ Φ`, accumulated incrementally over minibatches).
- Block weights' SGD is replaced by **random projection + ReLU** (fixed).
- Block-to-block handoff: where mono_forward_v2 passes detached hidden
  states, N3 passes **residual targets** in logit space.

## Why it survives the three structural findings

1. **Stochasticity filter**: ridge fits an expectation of `Φᵀ Y` over the
   whole training distribution; no one-shot WTA. CE-equivalent.
2. **Paradigm-A ceiling**: defeated by additive boosting over layers — the
   residual target changes between layers, so the L-layer composite is
   strictly more expressive than any single fixed-feature ridge.
3. **n-gram floor**: byte-context window is exactly the input n-gram
   sufficient statistic, so the lower bound is at least competitive with
   small-window KN.

## Risk

If even boosted random-feature ridge cannot clear 0.70 on byte windows,
this is **the falsification** of "layer-local closed-form supervision is
enough" — a high-information negative.
