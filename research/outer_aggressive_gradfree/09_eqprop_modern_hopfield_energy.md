# 09 · Equilibrium Propagation on a Modern Hopfield Energy

## Mechanism

Equilibrium Propagation (Scellier & Bengio 2017, arXiv 1602.05179) is a
gradient-free *contrastive* learning rule for energy-based models. Given an
energy function E(σ; W, x) over state σ with weights W and clamped input x:

- **Free phase:** let σ relax to ground state σ_free of E with x clamped to
  context bytes.
- **Nudged phase:** add a small target-pulling term β · L_target(σ); let σ
  relax to ground state σ_nudge.
- **Local update rule:** ΔW = (1/β)(∂E/∂W|_nudge - ∂E/∂W|_free).

For energy E that is a quadratic form in W and a sum of two-neuron
interactions, ∂E/∂W is a single outer product `σ σ^T`. So EqProp's
weight update is a **Hebbian outer-product difference**: nudged outer
product minus free outer product. **Fully gradient-free in the
backprop sense.**

The new direction: take the **Ramsauer 2020 modern continuous Hopfield
energy** as the E function and apply EqProp to train its slow weights
(the (k, v) memory bank itself).

    E(σ; ξ_1...ξ_N, β) = - β^{-1} lse(β · Ξ^T σ) + (1/2) σ^T σ + ...
    Ξ = [ξ_1, ..., ξ_N]   ∈  R^{d × N}

Setting "σ ground state" via the Hopfield relaxation: σ* = Ξ softmax(β · Ξ^T σ).

For the LM task, σ is partitioned into a context block (clamped to last
T bytes' features) and a target block (the byte_{T+1} one-hot, free in
the free phase, soft-clamped in the nudged phase). EqProp's update rule
then writes new patterns into Ξ via outer-product Hebbian, but **with the
sign of the nudged-vs-free difference as the modulation**.

This is a *contrastive* version of the Hopfield write — closer to the
Boltzmann machine training of the 1980s than to Schlag's delta-rule fast
weights. Crucially, it has the "anti-Hebbian + Hebbian" structure that
prevents the runaway of standard Hebbian rules.

## Seed papers

- Scellier & Bengio, *Equilibrium Propagation: Bridging the Gap between
  Energy-Based Models and Backpropagation*, Frontiers Comp. Neurosci. 2017
  (arXiv 1602.05179).
- Laborieux et al., *Scaling Equilibrium Propagation to Deep ConvNets by
  Drastically Reducing Its Gradient Estimator Bias*, Frontiers 2021
  (arXiv 2101.05536). Practical scaling tricks.
- Ramsauer et al., ICLR 2021 (arXiv 2008.02217). Modern Hopfield energy.
- Bal & Sengupta 2024 / Lin, Bal, Sengupta 2024 — recent attempts to scale
  EqProp to sequence learning (arXiv 2508.15989 and IJCAI 2023 sequence-EP
  paper). Has shown some success at CIFAR-scale; LM is open.
- *Textual Equilibrium Propagation*, ICLR 2026 (arXiv 2601.21064) — text
  variant of EqProp, but for compound AI systems with prompt-level
  inference. Not directly applicable but adjacent.

## Why it could work here

- **Contrastive Hebbian writes have built-in stability** — they don't run
  away. This solves one of the known problems with #1 (Krotov DAM with
  runaway capacity).
- **The Ramsauer Hopfield energy is *exactly* what `hopfield_layer` uses
  for read.** Using the same energy for write makes the entire model a
  single energy-based system, which is conceptually clean.
- **The free→nudged relaxation can be done in 2–3 iterations** under
  Ramsauer high-β (each step is exactly one softmax-attention).
- **A100 Tensor Cores handle softmax-attention well.** The compute is
  matched to hardware.

## Threshold of plausibility

EqProp has historically been hard to scale because the contrastive signal
is small (β must be small for the linear-response approximation) and
gradient estimation is noisy. At LM scale this typically caused trouble.
Laborieux 2021 showed a fix on CIFAR-CNN, and Lin/Bal/Sengupta 2024 showed
some sequence-modeling success at ~10M params. But neither was at
60K-val-byte char-LM with 300 s wall-clock.

The 2026 *Textual Equilibrium Propagation* paper is for compound
prompt-based systems, not a from-scratch LM, so it doesn't directly inform
our case.

Realistic estimate: 0.30–0.50. The per-step relaxation iteration is a
known time-cap risk (per `finding_kernel_round1_results.md`, Performer
blew the cap for similar reasons). With only 2-3 relaxation steps and a
single Hopfield energy this should not blow the cap, but the weight update
is noisy enough that 2000 train steps may not suffice.

Lower confidence than other portfolio items. Included for genuine novelty:
**no one has applied EqProp to char-LM with a modern Hopfield energy as
far as my search shows.**

## Failure modes

- **Linear-response approximation breaks down** at large β. Use β → 0 in
  EqProp and accept slow convergence (more train steps needed).
- **Free-phase / nudged-phase imbalance** — the modal-byte target pulls
  the system in a meaningful direction only if the energy basin around
  the target is shallow enough. Stochastic targets are the right kind
  of target for EqProp, since modal-byte is the "average" target across
  multiple soft-clamping. **Should pass the stochasticity filter.**
- **Iterative relaxation blows the 300 s cap.** Cap relaxation at 2 steps
  even at the cost of slower convergence.
- **The (free - nudged) update has small magnitude** at small β, so the
  effective learning rate must compensate. Easy to set wrong; sweep.
- **Update sign cancellation** can cause stalls. Use the "Holomorphic
  EqProp" variant (Laborieux 2022) that uses two β's to reduce variance.

What would falsify it: 2-layer EqProp-Hopfield, β = 0.1, 2-step
relaxation, 2000 train steps — val acc ≤ 0.40 → confirms EqProp does
not scale to char-LM under our constraints. Reject.

## Smallest first experiment

`eqprop_hopfield_v1`:

1. **Architecture:** a single energy E(σ; Ξ) with σ = (context_features ∈
   R^d_c, byte_target ∈ R^256), Ξ a (d_c + 256, M) pattern matrix,
   M = 4096.
2. **Pattern matrix init:** small random Gaussian.
3. **Frozen feature extractor:** RFF over last 256 bytes → R^d_c, d_c = 512.
4. **Per training batch (size B):**
   - Compute context features φ.
   - Free phase: σ_free = lse-softmax-Hopfield relaxation with target block
     unclamped, 2 steps from φ-init.
   - Nudged phase: same with target block soft-clamped toward true
     byte_{t+1} one-hot (β_nudge = 0.5).
   - EqProp update: Ξ += η · (1/β_nudge) · (Ξ_grad_nudge - Ξ_grad_free)
     where Ξ_grad is a single outer-product term σ · attention_softmax^T.
5. **No SGD, no optimizer state. Pure outer-product writes scaled by
   contrastive difference.**
6. **Inference (`predict`):** clamp context features φ, run 2-step free
   relaxation, read out σ-target → softmax → 256-d.

Sweep: M ∈ {1024, 4096, 16384}, β_nudge ∈ {0.1, 0.5, 1.0}, n_steps ∈
{500, 2000}.

## Memory-movement analysis

Per training step: 2 free + 2 nudged relaxation passes × softmax-attn
cost = 4 × (B, M) × (M, d) = 4 × B × M × d matmul. At B = 32, M = 4096,
d = 768 → 4 × 10^8 flops per step. 2000 steps = 8 × 10^11 flops. < 5 s
compute. HBM traffic dominated by Ξ writes (M × d = 3 M floats per
update) which adds bandwidth pressure but stays under 10 GB/s × 5 s
= 50 GB total traffic, well under A100's 2 TB/s.

Inference: 2 relaxation iters × softmax-attn per byte. (1, M) × (M, d) ~
3 M flops per byte; 60 K bytes = 200 G flops total = ~ 1 s.

Energy estimate: 5-15 kJ if it converges.

## References

- Scellier & Bengio EqProp 2017: <https://arxiv.org/abs/1602.05179>
- Laborieux et al. 2021: <https://arxiv.org/abs/2101.05536>
- Ramsauer Hopfield 2021: <https://arxiv.org/abs/2008.02217>
- Sequence-EP IJCAI 2023:
  <https://www.ijcai.org/proceedings/2023/0329.pdf>
- Scalable EP 2024: <https://arxiv.org/abs/2508.15989>
- Textual EqProp ICLR 2026: <https://arxiv.org/abs/2601.21064>
