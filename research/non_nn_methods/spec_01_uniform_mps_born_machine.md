# Spec 01 — Uniform Matrix Product State (uMPS) Born-machine LM

## 1. Method & mechanism

A uniform Matrix Product State (uMPS) represents the joint distribution over a length-T byte
sequence as the squared magnitude of a contracted tensor network:

    P(x_1, ..., x_T) = |<L| A[x_1] A[x_2] ... A[x_T] |R>|^2 / Z

where `A` is a single (D x V x D) core tensor (V = 256 byte symbols, D = bond dimension), `|L>`
and `|R>` are D-dim boundary vectors, and `Z` is the partition function — itself a small
D^2 x D^2 power-iteration / transfer-matrix solve. Training is by maximum likelihood with
density-matrix renormalization-group (DMRG)-style local sweeps that **directly solve a small
linear system** at each site (Stoudenmire & Schwab 2016; Miller, Rabusseau, Terilla 2021).
No backpropagation through the network.

Conditional next-byte prediction: P(x_{t+1} | x_1..x_t) is one matrix-vector product against
the running left-environment vector L_t = <L| A[x_1] ... A[x_t]>, normalized over the 256
possible next symbols.

## 2. Why not a neural network / not backprop

uMPS is a tensor network, not an MLP/transformer/RNN. The optimization is DMRG: pick a site,
form a local environment, solve a small generalized eigenvalue or least-squares problem for
the new core, sweep. There is no chain-rule gradient through a layer stack. Modern variants
also support Riemannian-manifold optimization or moment-matching closed-form updates — both
are gradient-free in the backprop sense.

## 3. Universal approximation status

**Proven for finite alphabets:** any probability distribution over a length-T sequence over a
finite alphabet can be represented exactly by an MPS with bond dimension D up to V^(T/2).
For long sequences the meaningful bound is on truncated approximations: the error of best
bond-D approximation decays with the singular-value spectrum of the unfolded distribution.
For a stationary stochastic source on V symbols, asymptotic representable-distribution class is
captured by uMPS as D grows (Glasser et al. 2019; Miller 2021).

## 4. Discrete categorical fit

Native. The output is `|<L| A[x_t+1] R_t>|^2 / sum_v |<L| A[v] R_t>|^2` — a length-V probability
vector by construction, where R_t is the right-environment vector pre-cached during prefill.
No softmax over logits; the Born rule provides the normalization.

## 5. Autoregressive applicability

Yes. Miller, Rabusseau, Terilla 2021 (the u-MPS paper) explicitly demonstrate sequence
modeling with O(log T) depth conditional sampling on synthetic context-free language data.
uMPS at WikiText scale is **a novel application** — to my knowledge no published uMPS result
is at byte-level WikiText-103 with a 0.70 char-acc-class metric. This is a capability demo.

## 6. Roofline analysis

Dominant kernel: one A100 step is a batched matmul of a (B x D) state by an (D x V x D) core,
tensor-contracted along D, producing a (B x V x D) result. For B=128, D=256, V=256:

    FLOPs / step = 2 * B * D * V * D = 2 * 128 * 256 * 256 * 256 ~= 4.3e9 FLOPs
    bytes moved  = (B*D + D*V*D + B*V*D) * 2 (bf16) ~= 34 MB read+write

    Arithmetic intensity = 4.3e9 / 3.4e7 ~= 125 FLOPs/byte

On A100, ridge = 156 FLOPs/byte — uMPS at D=256 sits *just under* the ridge (slightly
bandwidth-bound for the core read). At D=512 we get ~250 FLOPs/byte — comfortably
compute-bound. Both fit Tensor Cores. **Verdict: compute-bound at D >= 384.**

DMRG sweep cost per epoch: O(T * D^3) for the optimal-core solve at each position. For
D=256 and T=2e6 positions, that is ~3.4e13 FLOPs ~= 5 s on A100 at 70% utilization — fits.

## 7. Top references

1. Miller, Rabusseau, Terilla 2021, "Tensor Networks for Probabilistic Sequence Modeling", AISTATS.
   <http://proceedings.mlr.press/v130/miller21a/miller21a.pdf>
   *u-MPS sequence model with O(log T) sampling. Synthetic CFL only.*
2. Han, Wang, Fan, Wang, Zhang 2018, "Unsupervised Generative Modeling Using Matrix Product States",
   Phys. Rev. X 8, 031012. <https://arxiv.org/abs/1709.01662>
   *MNIST and Bars-and-Stripes; DMRG sweeps; bond dimension D=300 in their largest run.*
3. Stoudenmire & Schwab 2016, "Supervised Learning with Tensor Networks", NeurIPS.
   <https://arxiv.org/abs/1605.05775>
   *Original DMRG-trained MPS classifier; the lineage all the LM variants trace to.*
4. Glasser, Pancotti, Cirac 2019, "From Probabilistic Graphical Models to Generalized
   Tensor Networks for Supervised Learning". <https://arxiv.org/abs/1806.05964>
   *Representation-theoretic expressivity bounds.*
5. Wall, Bevilacqua, Carleo 2025, "Initialization and training of matrix product state
   probabilistic models". <https://arxiv.org/abs/2505.06419>
   *Recent training-stability tricks; relevant for fitting in 300 s.*

## 8. Limitations / failure modes

- DMRG sweeps are **inherently sequential** along the chain. Wall-clock per epoch is set by
  D^3 * T, hard to parallelize past a single sweep direction. Mitigation: small D (256–384),
  one-shot moment-matching init (Wall 2025), single epoch.
- uMPS represents only correlations the *bond dimension* can carry. English at byte level has
  long-range syntactic dependencies; published MNIST-scale uMPS results suggest decent local
  statistics but distant-correlation capture is harder than transformers.
- The Born rule's |...|^2 normalization can be numerically unstable for long sequences; running
  log-norm carry is required.
- **No published byte-level WikiText result exists** for uMPS. Plausible char-acc range from
  the literature: 0.45–0.65; clearing 0.70 would be a capability surprise.
- Char-LM scoring contract: predict() must return a P(c) vector. uMPS gives this natively, no
  wrapper needed.

## 9. Experiment spec

**Setup.**
- Architecture: single uMPS core, V=256, bond dim D=384, fp32 cores (numerical stability matters).
- Boundary vectors `|L>`, `|R>` learned.
- Training: 5 left-to-right DMRG sweeps over ~50–80 MB of train (whatever fits in 300 s wall).
  At each site, solve the local effective-tensor least-squares update with one Cholesky.
- Alternative: stochastic-gradient-on-cores using natural-gradient (no chain-rule activations)
  — fall back to this only if DMRG cannot reach 0.50 acc.
- Initialization: small-random + identity bias (Wall 2025 prescription).

**CharModel translation.**
- `predict()`: contract running left-environment vector L_t against `A[v]` for each v in 0..255,
  return normalized squared magnitudes — one D^2 x V matmul per predict call (fast).
- `observe(c)`: update L_t ← L_t @ A[c] / norm.
- `reset()`: L_t ← |L>.

**Energy budget.** Aim for one DMRG sweep ~= 60 s at D=384. Total 300 s for 4–5 sweeps.
Predicted energy 25–45 kJ (compute-bound, full GPU utilization). **Will not beat 51.7 kJ
on energy unless training converges in 2 sweeps.** Primary goal: demonstrate the mechanism
clears 0.70 at all.

**Char-acc ceiling estimate.** 0.55–0.65 from the MPS-on-text precedent literature. Reaching
0.70 likely requires either D=768 (busts the 300 s budget) or a uMPS+ridge-head hybrid.

## 10. Verdict — **Tier B**

Promising as a *new* mechanism on the benchmark — no uMPS result exists for byte-level
WikiText. Compute-bound roofline. DMRG training is gradient-free by construction. Realistic
ceiling 0.65 puts it at risk of DQ on the 0.70 floor; the value is the capability demo,
not the energy number. Run after the Tier-A baselines have landed.
