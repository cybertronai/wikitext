# Spec 04 — Falkon (Nyström + preconditioned CG) kernel ridge LM

## 1. Method & mechanism

Falkon (Rudi, Carratino, Rosasco 2017) is an algorithmic recipe for *exact-quality*
large-scale kernel ridge regression at near-linear cost. Pick M = O(sqrt(N)) Nyström
landmarks {x_{c_1}, ..., x_{c_M}} from the training set; form the M x M kernel matrix
K_MM and the N x M cross-kernel K_NM; precondition by L = chol(K_MM + reg I); solve
the M-dim system

    (L^-T K_MN K_NM L^-1 + lambda L L^T) alpha = L^-T K_MN Y

via preconditioned conjugate gradient (typically 10–30 iterations). Predict at a new x
by f(x) = sum_i alpha_i k(x, x_{c_i}).

For byte-LM: x = byte n-gram context window of length K, encoded as a one-hot vector
or hashed feature vector; kernel = RBF or polynomial or *string kernel* (Lodhi 2002)
on the window. Y = 256-dim one-hot of next byte; vector-valued KRR fits one
column of alpha per output class. The full implementation is in the `FalkonML/falkon`
GitHub library — multi-GPU PyTorch, GP-accelerated via KeOps.

## 2. Why not a neural network / not backprop

Pure kernel ridge regression with Nyström approximation. No layers, no SGD on
hidden representations, no backprop. The only iteration is the preconditioned CG
on a M-dim system, where M is small (10K–100K). Closed-form (modulo CG tolerance).

## 3. Universal approximation status

**Proven** for universal kernels (RBF, Laplace) on a compact domain. Falkon comes with
matching upper bounds on the estimation error for the *full* kernel ridge regressor
with high probability — i.e. Nyström doesn't sacrifice asymptotic statistical
accuracy at M = O(sqrt(N) log N) (Rudi et al. 2017, Theorem 3).

For string kernels (Lodhi 2002 spectrum/subsequence kernels) the relevant question
is whether the feature space is rich enough to encode English n-gram distributions
— empirically yes for text classification, untested for autoregressive char-LM.

## 4. Discrete categorical fit

Multi-output ridge regression to a 256-d one-hot target. Output is a 256-d vector
of real scores per input. Argmax for predict(). Same stochasticity-filter profile
as spec_02.

## 5. Autoregressive applicability

Same sliding-window mechanic as spec_02. The autoregressive structure is in the
streaming input, not the model — Falkon itself is treating each (context, next-byte)
pair as an iid sample.

**Falkon has not been published on autoregressive char-LM.** Its closest published
application is the YELP text classification benchmark with 6.52e7-dim binary
n-gram features at 1.5e6 training samples (FALKON paper, Section 5). Adapting to
char-LM is novel; the throughput / scaling story carries.

## 6. Roofline analysis

Hyperparameters: N=5e6, M=20000, K=16 byte context, d=256*16=4096.

**Cross-kernel K_NM (N x M):** dominant kernel evaluation cost. For RBF on x in R^d,
   - Per-entry cost: ~3d FLOPs = 12K FLOPs.
   - Total: N * M * 3d = 5e6 * 2e4 * 12e3 ~= 1.2e15 FLOPs ~= 4 s on A100 bf16.
   - HBM traffic: M*d (landmarks, persistent) + N*d streamed = ~80 GB streamed.
   - **Arithmetic intensity = 1.2e15 / 8e10 = 15000 FLOPs/byte — strongly compute-bound.**

**M x M kernel decomposition:** M^3 / 3 = 2.7e12 FLOPs — sub-second on A100.

**CG iterations:** each iter is O(N*M) for the K_NM @ vec product = 1e11 FLOPs each;
20 iters = 2e12 FLOPs ~= 1 s.

Falkon's published throughput on a single GPU: ~1e7 training points + 1e4 landmarks
fit in seconds. Our N is 5x larger but M is 4x smaller — fits comfortably.

## 7. Top references

1. Rudi, Carratino, Rosasco 2017, "FALKON: An Optimal Large Scale Kernel Method", NeurIPS.
   <https://papers.nips.cc/paper/6978-falkon-an-optimal-large-scale-kernel-method.pdf>
   *Original.*
2. Meanti, Carratino, Rosasco, Rudi 2020, "Kernel methods through the roof: handling
   billions of points efficiently", NeurIPS. <https://arxiv.org/abs/2006.10350>
   *Multi-GPU; the actual `falkonml/falkon` library implementation.*
3. Meanti, Carratino, De Vito, Rosasco 2022, "Efficient Hyperparameter Tuning for
   Large Scale Kernel Ridge Regression". <https://proceedings.mlr.press/v151/meanti22a/meanti22a.pdf>
   *Auto-tuning of lambda and kernel hyperparams — relevant since we cannot afford a
   manual sweep in 300 s.*
4. Lodhi, Saunders, Shawe-Taylor, Cristianini, Watkins 2002, "Text Classification using
   String Kernels", JMLR. <https://www.jmlr.org/papers/volume2/lodhi02a/lodhi02a.pdf>
   *String kernel — alternative to RBF on one-hot for raw text. Mentioned for completeness;
   evaluating string kernels at scale is its own engineering project.*
5. Avron, Sindhwani, Yang, Mahoney 2016, "Quasi-Monte Carlo Feature Maps for
   Shift-Invariant Kernels". <https://www.jmlr.org/papers/volume17/14-538/14-538.pdf>
   *Lower-variance alternative to vanilla RFF — applies to the kernel choice here.*

## 8. Limitations / failure modes

- **Falkon library has Python dependencies** (KeOps, PyTorch). Need to verify they
  install cleanly in the Modal image (it has tiktoken+torch already; install Falkon
  in the submission's setup).
- **Choice of kernel + bandwidth dominates.** RBF on one-hot byte windows may not
  capture n-gram interaction efficiently; consider polynomial or arc-cosine kernel.
- **Landmark selection** by uniform random can miss rare contexts; consider
  k-means++ on a subset for better coverage if first pass fails.
- **N x M kernel evaluation is the wall-clock dominant cost** at our scale; if the
  Modal A100 is busy with image-pull at startup we may have <250 s effective compute,
  which is tight.
- **String kernels are quadratic in window length** — avoid them on K=16 byte windows
  unless engineering bandwidth allows.
- **Same stochasticity-filter pass as spec_02** (soft outputs).

## 9. Experiment spec

**Setup.**
- Context window: K=16 bytes, one-hot encoded → 4096-d sparse-binary vector.
- Kernel: RBF (Gaussian) with bandwidth gamma swept ∈ {0.1, 0.5, 1.5}.
- Nyström landmarks: M=20000, uniform random sample of training positions.
- CG iters: 20.
- lambda swept ∈ {1e-6, 1e-3}.

**Implementation.**
- `pip install falkon` in submission setup. If install latency exceeds ~30 s on
  Modal cold start, vendor a pinned wheel.
- Stream-load N=5e6 contexts; build N x M cross-kernel by Falkon's blockwise GPU
  kernel evaluator.
- Solve with `falkon.Falkon` class out of the box.
- Per-class fit by vector-valued KRR (Falkon supports multi-output natively).

**CharModel translation.**
- `predict()`: evaluate kernel of current 16-byte window against all 20K landmarks
  → 20K-d vector → dot with alpha (20K x 256) → 256-d scores. Cost: 20K * (4096
  byte-window encoder) + 20K * 256 = ~5M ops per char. ~10 ms / char on GPU; fine.
- `observe()`: shift window, recompute landmark kernel.
- `reset()`: zero context buffer.

**Energy budget.** 60–120 s for kernel build + CG + Cholesky; cold-start ~30 s for
Falkon install. ~15–35 kJ total.

**Char-acc ceiling estimate.** Same as spec_02 / spec_03 — 0.55–0.70 plausible.
The library and method are validated at this scale on YELP text classification;
adapting to AR char-LM is the new uncertainty.

## 10. Verdict — **Tier B**

Most rigorous paradigm-A kernel-machine implementation in the portfolio (statistical
guarantees, GPU-accelerated library, published scalability). One step heavier than
spec_02 because of the dependency / library install cost. Run after RFF (spec_02)
as a "if RFF cleared 0.65, scale up with Nyström" follow-up.
