# Spec 02 — Random Fourier Features + closed-form ridge LM

## 1. Method & mechanism

A purely shallow, gradient-free language model: hash a fixed-length byte context window
x_t = (c_{t-K+1}, ..., c_t) into a high-dimensional explicit feature map phi(x_t) approximating
a Gaussian (RBF) kernel by Rahimi-Recht random Fourier features, then fit a closed-form ridge
regression that maps phi(x_t) to a 256-dim one-hot of the next byte. At inference, P(next | x_t)
is softmax( W phi(x_t) ) (or just argmax — the harness only needs argmax).

    phi(x): R^d -> R^m, phi_i(x) = sqrt(2/m) cos(omega_i^T x + b_i)
    omega_i ~ N(0, gamma^2 I),  b_i ~ U[0, 2 pi]

    W = (Phi^T Phi + lambda I)^{-1} Phi^T Y     (closed-form, one Cholesky)

Input encoding for x_t: byte-positional one-hot of size 256*K, or a learned but frozen
random projection of one-hot into d-dim before phi (controls input scale).

## 2. Why not a neural network / not backprop

This is a classical kernel machine (paradigm A in `finding_kernel_two_paradigms.md`).
phi(.) is fixed at initialization (random sampling from the kernel's Fourier dual).
W is solved in closed form — one Cholesky on the m x m Gram-style normal equations.
There is no chain-rule backprop and no SGD on phi or W. The only iteration is the
Cholesky factorization itself.

## 3. Universal approximation status

**Proven.** RBF kernels are universal kernels on R^d (Steinwart & Christmann 2008,
Thm 4.63). RFF with m → ∞ converges to the exact RBF kernel; for finite m, function-class
expressivity is bounded by the RKHS spanned by phi. For categorical regression on text
features, the universal-approximation guarantee is asymptotic in m.

## 4. Discrete categorical fit

Output is a 256-vector of real scores W phi(x). Two options:
- **Hard:** argmax — one-hot, harness scores correctly.
- **Soft:** softmax(W phi(x)) — for energy-based ranking / partial credit if we extend
  to a hybrid with a second method. Both fit the CharModel API.

This is a **soft** kernel method by construction (output is a real-valued vector over
classes, not a single hard winner), so it survives the stochasticity filter
(`finding_kernel_stochasticity_filter.md`).

## 5. Autoregressive applicability

Standard: fixed window K, slide one byte at a time. predict() = matmul of cached
phi(x_t) with W (m x 256). observe() = shift window, recompute phi for the new window.
Recompute of phi can be partial if the projection is incremental (RFF doesn't compose
across shifts; full recompute per char, but cheap).

Has this been used for AR sequence modeling? RFF for text classification yes
(Falkon YELP benchmark, n=1.5e6 docs). For *autoregressive char-LM* this is novel
application territory — RFF has not been published at the byte-level WikiText scale.

## 6. Roofline analysis

Dominant kernels:

**Training-time normal-equations.** Build Phi (N x m), N ~ 5e6 training positions, m=4096.
   - Phi @ Phi.T: 2 * N * m^2 = 2 * 5e6 * 4096^2 ~= 1.7e14 FLOPs ~= 0.5 s on A100 bf16.
   - Cholesky on (m, m): 1/3 * m^3 = 2.3e10 FLOPs — negligible.
   - HBM traffic for Phi @ Phi.T: streamed read of Phi once = 5e6 * 4096 * 2 = 40 GB.
   - **Arithmetic intensity = 1.7e14 / 4e10 = 4200 FLOPs/byte — deeply compute-bound.**

**Phi materialization.** N * m * d = 5e6 * 4096 * 256 = 5.2e12 FLOPs ~= 17 ms on A100.

**Inference (predict).** One m x 256 matmul per char: 2 * m * 256 = 2.1e6 FLOPs / char.
For 60K val chars: 1.3e11 FLOPs total — negligible.

Verdict: **strongly compute-bound** at m=4096. All the parameters fit comfortably in
HBM (4096*256*4 bytes for W = 4 MB). The single Cholesky is the only sequential pass;
everything else is matmul-friendly.

## 7. Top references

1. Rahimi & Recht 2007, "Random Features for Large-Scale Kernel Machines", NeurIPS.
   <https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf>
   *Original RFF paper.*
2. Le, Sarlós, Smola 2013, "Fastfood: Approximating Kernel Expansions in Loglinear Time",
   ICML. <https://proceedings.mlr.press/v28/le13.html>
   *Structured RFF with log-linear sample/feature cost — drop-in replacement if m=4096 is too slow.*
3. Avron, Kapralov, Musco, Musco, Velingker, Zandieh 2017,
   "Random Fourier Features for Kernel Ridge Regression: Approximation Bounds and Statistical Guarantees".
   <https://arxiv.org/abs/1804.09893>
   *Precise sample-complexity bounds for RFF-KRR — guides choice of m.*
4. Yu, Suresh, Choromanski, Holtmann-Rice, Kumar 2016, "Orthogonal Random Features", NeurIPS.
   <https://papers.nips.cc/paper/2016/hash/53adaf494dc89ef7196d73636eb2451b-Abstract.html>
   *Variance-reduced RFF with structured-orthogonal omega — typically 2-4x sample efficient.*
5. Rudi, Carratino, Rosasco 2017, "FALKON: An Optimal Large Scale Kernel Method", NeurIPS.
   <https://papers.nips.cc/paper/6978-falkon-an-optimal-large-scale-kernel-method.pdf>
   *Companion Nyström+CG variant — see spec_04 for the Falkon-specific version.*

## 8. Limitations / failure modes

- **Capacity ceiling at m=4096.** For an N=5e6 dataset, this is in the under-parameterized
  regime relative to a transformer with millions of params. Plausible char-acc 0.50–0.65;
  reaching 0.70 may require m=16384 (16x cost — Phi @ Phi.T becomes 27 TFLOPs, still <100s).
- **K (context window) matters enormously.** Long K → high-dim input → RBF curse-of-dim;
  K too small → no signal. Sweet spot is typically K=8–16 for byte-level (one BPE chunk).
- **Idle byte hash.** A naive one-hot encoder gives 256*K dim input — wasteful. Better: a
  fixed random projection of one-hot to d=128 before phi.
- **Numerical conditioning** of the normal equations at m=4096 with N=5M needs jitter
  (lambda) tuning. Cholesky may fail without it.
- **Stochasticity not a concern** — output is a soft score per class.

## 9. Experiment spec

**Setup.**
- Context window: K=16 bytes.
- Input embedding: one-hot * frozen random projection R ∈ R^{4096 -> 256}, output d=256.
- Random Fourier Features: m=4096; gamma swept ∈ {0.1, 0.3, 1.0} (3 settings).
- Closed-form ridge with lambda swept ∈ {1e-4, 1e-2, 1.0}.
- All in bf16 for matmul, fp32 for Cholesky.

**Training procedure.**
1. Stream N=5e6 byte positions from `wiki.train.raw`.
2. For each position t, compute phi(x_t) and add to Gram accumulator Phi.T @ Phi and
   cross-correlation accumulator Phi.T @ Y. Both can be done in batched chunks of 64K
   positions to keep memory bounded.
3. Cholesky-solve W = (Gram + lambda I)^{-1} Phi.T @ Y.
4. Total wall ~30–60 s. Energy estimate: 3–8 kJ at A100 sustained ~250 W.

**CharModel translation.**
- `predict()`: 1 matmul of cached phi(x_t) by W → 256-d vector; argmax.
- `observe(c)`: append c to ring buffer of length K, recompute phi(x_{t+1}) — one O(m*d) matmul.
- `reset()`: clear ring buffer.

**Success criteria.**
- *Strong:* >= 0.70 char-acc on val 60K. Mechanism demonstration + leaderboard entry; energy
  estimated at ~5 kJ would be 10x better than baseline.
- *Useful negative:* < 0.50 confirms paradigm-A kernel-machine-replaces-model fails on byte LM
  at this m, validating `finding_kernel_two_paradigms.md`'s "scaling story is weak" claim.

**Diagnostics to log.**
- Train and val char-acc as a function of m.
- Gram-condition number (post-jitter).
- Per-position phi compute time vs Cholesky time vs predict time.

## 10. Verdict — **Tier A**

Cheapest possible kernel-machine baseline. Single Cholesky, fits in <60 s. Strongly
compute-bound. Resolves an open question in the kernel-LM literature (no published
RFF result at byte-level WikiText). Even a negative result (fails to clear 0.70)
is publishable as the first calibrated measurement.
