# Experiment N1: Predictive Coding Networks (PCN) for byte LM

## Hypothesis

Whittington & Bogacz (2017) and Millidge et al. (2020) show that a stack of
hierarchical predictive-coding layers, with the top layer clamped to a target
and inner-loop activity updates run to convergence, implements a strictly
**local Hebbian** learning rule whose weight updates approximate backprop.
Two structural properties make PCN a high-information candidate at byte-LM:

1. The update `ΔW_l = -η · e_l · x_{l+1}^T` is driven by an *expectation of
   error*, not a one-shot WTA. **Escapes the stochasticity filter** that
   killed Softhebb / NBB on byte targets.
2. Iterative energy descent over activities `x_l` makes the features adapt
   *during* inference. **Escapes the Paradigm-A ≤ 0.37 ceiling** that caps
   frozen-feature + linear-readout methods.

No PCN result exists for byte-level WikiText; the only nearby data point in
the repo is `mono_forward_v2` (0.7346 / 46.2 kJ) — a layer-local SGD-on-probe
method, the only deep gradient-free PASS to date.

## Architecture

Byte-level (vocab=256). MLP-only — PCN through self-attention is open
research and outside the budget.

```
x_0 (one-hot/embedded K-byte context window)
   └─ W_1 ─→ x_1   d=512, ReLU
              └─ W_2 ─→ x_2   d=512, ReLU
                         └─ W_3 ─→ x_3   d=512, ReLU
                                    └─ W_out ─→ x_top  one-hot next-byte target (clamped)
```

- `K = 64` byte context window, flattened to a `K · 256 = 16384`-dim
  one-hot input (sparse-multiplied — cheap on GPU).
- 3 hidden PCN layers + 1 output layer.
- `g = ReLU` for hidden layers; output `μ_top = W_out x_3` matched against
  one-hot target (no nonlinearity at top — std PCN).
- ~5–8M params.

## PCN inference + update

For each minibatch (B random K-byte windows):

1. Forward init: `x_l ← g(W_l x_{l+1})` walking from bottom to top.
   `x_top` is **clamped** to the one-hot next-byte target.
2. Inner loop (T=6): update hidden activities in fp32 to descend energy
   `F = Σ_l ||e_l||²`:
   `x_l ← x_l - α · (e_l - g'(μ_l) ⊙ (W_{l+1}^T e_{l+1}))`,
   where `e_l = x_l - μ_l`, `μ_l = W_l x_{l+1}`, `g'(μ_l) = (μ_l > 0)`.
   Layer 0 and top are clamped — no inner update.
3. Local Hebbian weight update (no autograd):
   `ΔW_l = η · e_l · x_{l+1}^T` (and bias `Δb_l = η · e_l`).

All steps inside `torch.no_grad()`. No `loss.backward()` anywhere.
Per-weight update depends only on adjacent-layer activities — fully local.

**T choice.** T=1 reduces to a single Hebbian step (= generalised target
prop, no signal-flow through depth). T → ∞ recovers backprop but kills
the budget. Salvatori et al. 2022 use T ∈ {16, 32}; we set T=6 so the
T × forward-backward-flavoured-pass cost fits the 250s training budget,
and document e_l norms in the log so a non-converging inference is
diagnosable.

## Initialization

Hidden layers: He-normal `std = sqrt(2/fan_in)`. Output W_out small
(`std = 0.01`) so initial `μ_top` is near zero and the clamp creates a
strong driving error on step 0. Activity init from forward pass; no
warm-starting across minibatches (different windows, no temporal
continuity at training time).

## Supervision

Target = one-hot next byte (vocab 256) at the centre of each window.
Clamping `x_top` = target is the standard PCN supervision; it makes the
top error `e_top = x_top - μ_top` push the network toward predicting
that byte, exactly the byte-target signal that `mono_forward_v2`
already established escapes the stochasticity filter.

## Eval

Per-byte streaming: take last K bytes, single forward pass, softmax
over `μ_top`. No inner loop needed at eval (only at training).

## Budget

- Training: B=512 windows × T=6 inner iter × (3 layer matmuls fwd +
  3 backward-flavoured) × d=512 ≈ 7.5 ms/step on A100 → ~30k steps in
  225 s.
- Eval: small MLP, single forward per byte, ~0.3 ms/byte → 18s / 60k.

## Risks

a) Inner loop fails to converge in T=6 → updates are noise. Mitigation:
log `||e_l||` per layer per N steps; if errors don't decrease over T,
diagnose and increase T.
b) 5M-param MLP undershoots 0.70. Real risk; floor is high. We accept
this — it would still be the first PCN byte-LM number.
c) Clamping creates a numerically large e_top initially. We scale the
top precision (effective lr on output) down.
