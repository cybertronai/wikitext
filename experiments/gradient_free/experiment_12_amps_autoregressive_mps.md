# Experiment 12: Autoregressive MPS (AMPS) — per-position normalized conditionals

## Hypothesis
An Autoregressive MPS (AMPS), in which each position has its own core tensor and the cores are constructed so that summing over the local index yields 1 *by construction* (Liu et al. PRL 2021, arXiv:2106.12974), is more numerically stable than a Born-rule uMPS at the same bond dimension and matches or exceeds val char-acc of experiment_11 at equal energy. AMPS eliminates the long-chain |ψ|² normalization that drove the Born MPS overflow problem (arXiv:2510.00382). Reaching 0.65–0.70 is plausible because (a) each conditional is *exactly* normalized — no partition function ever computed — and (b) gradients of log-likelihood are tractable, well-conditioned, and computed *site-by-site* (still backprop-free in the chain-rule sense).

## Motivation
Born-machine MPS conditional P(c_t | c_<t) is computed by *ratio of squared norms* — both numerator and denominator can underflow on a long sequence. AMPS replaces this with `P(c_t | c_<t) = |<L_t | A_t[c_t]>|² / sum_v |<L_t | A_t[v]>|²` where the normalization is **local to position t** and never accumulates across t. This is the cleaner AR factorization and the one most likely to actually clear 0.70 on bytes. We run both (experiment_11 = uMPS Born, experiment_12 = AMPS) to disentangle the two distinct failure modes: AR formulation vs. bond-dim ceiling.

## Method
A non-uniform MPS with one core per position is too large for our context — we instead use a **block-translation-invariant AMPS**: one shared core for every position, but each `predict()` *projects* the resulting D-vector onto a normalized categorical over V=256 via a local Born rule:

```
For each position t:
  ket_t      = L_t        ∈ R^D           # running left-environment
  scores_v   = ket_t · A[:, v, :] · R_norm    for v in 0..V-1, where R_norm is a learned (D,) head
  logits_v   = scores_v^2                  # Born rule on the head only
  P(c_t = v | c_<t) = logits_v / sum_v logits_v
```

`R_norm` is **not** a global wavefunction boundary — it is a *learned head vector* that maps the running bond vector to V categorical logits. Because the squared-magnitudes are normalized over V at each position, no long-chain norm carries; AMPS exactness comes from this local Born step. State update after observing the true c_t:

```
L_{t+1} = L_t · A[:, c_t, :] / ‖L_t · A[:, c_t, :]‖
```

Training: maximize sum_t log P(c_t = c_t* | c_<t*) by SGD on cores A and head R_norm. Crucially, the gradient w.r.t. A at position t depends only on (L_t, L_{t+1}, R_norm) — **the chain rule does not propagate through positions** because L_t is renormalized at each step (treated as a stop-gradient anchor for the gradient computation of A *at* step t). This is a backprop-free local update *in the same sense that DMRG is*: gradient is local to one site, not a global chain through nonlinearities.

## Memory-Movement Analysis
- **Per-step training FLOPs**: per-position, per-window: one (1, D) × (D, V, D) contraction = 2·D²·V FLOPs = 2·384²·256 = 75 MFLOPs. For B=512 windows, T=256 positions, the per-batch cost is B·T·2·D²·V = 512 · 256 · 75 M = 9.8 GFLOPs. At 5 % of A100 peak (compute-bound for these batched matvecs), one batch ≈ 1.3 ms; 5 epochs over 50 MB / (B·T) = 380 batches → 5 · 380 · 1.3 ms = 2.5 s of pure GEMM. Add 10× overhead for backward + optimizer → ~25 s total for 5 epochs. **Massive headroom.**
- **Memory**: core A is 150 MB fp32; gradient buffer another 150 MB; AdamW state 600 MB. Total ~1 GB — trivial on 80 GB A100.
- **Arithmetic intensity**: per-step contraction is (D, V) read against (1, D) → 2·D·V FLOPs over (D·V + D) bytes ≈ V ≈ 256 FLOPs/byte. Above the A100 ridge — **compute-bound**. Batched across B and T, this is one big GEMM.
- **Inference**: same as experiment_11 — single (D × V × D) · (D,) contraction = 75 MFLOPs/char, ~30 µs/char.

## Setup
- Dataset: WikiText-103 train split as bytes, V=256.
- Architecture: AMPS with D=384, fp32 core. Head vector R_norm ∈ R^D learned. Total params: 384·256·384 + 384 = 37.7 M params.
- Training: 5 epochs of SGD on 50 MB of training text in windows of length T=256, batch B=512. Loss = mean per-position NLL. Optimizer: AdamW lr=3e-3, wd=0, bf16 forward / fp32 master weights. Cores re-orthogonalized at the bond axis every 50 steps (numerical hygiene; cheap SVD on D×D matrix).
- Baseline comparisons: experiment_11 (uMPS Born); `hopfield_layer` (40.2 kJ / 0.7293).
- Reference: Liu et al. 2021 AMPS report "competitive with state-of-the-art neural networks" on MNIST and Fashion-MNIST as discrete distributions.

## Procedure
1. `cp -r submissions/umps_born_d384 submissions/amps_d384`. Strip the DMRG sweep code.
2. Implement the AMPS forward as a single batched contraction `einsum('btD,DvE,E->btv', L_t, A, R_norm)` where L_t is the running batched left-environment of shape (B, T, D), then square and normalize over v.
3. Compute the running L_t with `L_{t+1} = L_t · A[:, c_t, :] / norm`, **detached** from the gradient computation of A *at position t* (this is what makes it backprop-free in the chain-rule sense).
4. Optimize NLL with AdamW. Track per-epoch val NLL on a held-out 50K-char slice.
5. Wrap `CharModel`: same as experiment_11 — running L_t, predict via R_norm.
6. `python submit.py submissions/amps_d384 --yes`.

## Success Criteria
- **Capability**: val char-acc ≥ 0.55, demonstrating AMPS as a viable AR mechanism on bytes.
- **Surprise**: val char-acc ≥ 0.70 — clears the floor with a true AR tensor-network LM. First in repo.
- **Negative result**: val char-acc ≤ uMPS Born (experiment_11) — local-Born head does not buy expressivity over the global Born rule at D=384.

## Failure Modes & Diagnostics
- **Backward through detached L_t propagates anyway via PyTorch autograd subtleties**: explicitly `L_t = L_t.detach()` after every position update. Diagnostic: `print(L_t.grad_fn)` should be None during the position-update step.
- **AdamW lr too aggressive for a single-core network**: at lr=3e-3, fp32 core may diverge on the first 100 steps. Diagnostic: monitor ‖A‖_F; if it grows >2× in 100 steps, halve lr.
- **Acc plateau at 0.5 because R_norm dominates** (the trainable head is doing all the work): zero-init R_norm and confirm acc starts at 1/256 = 0.004; if it jumps directly to plateau by step 100, the core is not contributing.
- **Per-position normalization makes gradient computation O(V) instead of O(1) per token**: this is intentional (compute-bound is the goal). If wall-clock blows, drop V to a 192-byte working alphabet (UTF-8 prefix of WikiText is ~95 % ASCII).

## Estimated Cost
1 Modal A100-80GB run × ~5 min wall ≈ $0.10. A D=512 follow-up if v1 lands in [0.55, 0.70]: another $0.10.

## References
- Liu, Li, Zhang, Zhang 2021, "Tensor networks for unsupervised machine learning" (AMPS), Phys. Rev. Lett.; arXiv:2106.12974 — exact normalization + autoregressive sampling.
- Hou, Li, You 2023, "Sequential Learning on a Tensor Network Born Machine with Trainable Token Embedding", arXiv:2311.05050 — AR sampling and trainable embeddings.
- Glasser et al. 2019, "Expressive Power of Tensor-Network Factorizations", NeurIPS — Born-MPS strictly more expressive than HMM.
- Wall 2025, "Initialization and training of matrix product state probabilistic models", arXiv:2505.06419.
- Companion: experiment_11 (uMPS Born).
