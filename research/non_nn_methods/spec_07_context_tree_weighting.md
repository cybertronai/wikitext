# Spec 07 — Context-Tree Weighting (CTW) bit-level LM

## 1. Method & mechanism

Context-Tree Weighting (Willems, Shtarkov, Tjalkens 1995) is the Bayesian-optimal
universal source code for memoryless tree-source models. It maintains a context tree
of depth D where each node holds a *weighted mixture* of (a) the KT (Krichevsky-Trofimov)
estimator P_e at that node and (b) the product of weighted estimators of its children:

    P_w(node) = (1/2) * P_e(node) + (1/2) * P_w(left_child) * P_w(right_child)   (bit-level)

For bytewise prediction on byte streams, the canonical approach (Begleiter & El-Yaniv
2004 survey) is to run CTW bit-by-bit on the binary representation of bytes — 8 CTW
predictions per output byte, multiplied to get the byte-level distribution.

Training: one streaming pass over the train corpus updates the per-node KT counts
n_0, n_1 along the active context path of depth D. Prediction at a new context: walk
the tree to depth D, accumulate the weighted-mixture probability along the way.

## 2. Why not a neural network / not backprop

Pure Bayesian counting. No parameters trained by gradient descent. No layers. The KT
estimator P_e(node) = (n_0 + 0.5) / (n_0 + n_1 + 1) for bits 0/1 is a closed-form
Laplace-Bayesian update. The mixture weights {1/2, 1/2} are *fixed* by the Bayesian
prior over tree sources (Willems et al. 1995, Theorem 1) — no learning needed for them.

## 3. Universal approximation status

**Proven optimal** for the class of finite-memory tree sources up to depth D
(Willems et al. 1995, Theorem 2: redundancy = O(|S| log N / N) where |S| is the number
of leaves in the true tree source). This is *the* universal approximation theorem for
discrete sequence prediction; the model class is finite-context tree sources, which
captures every finite-order Markov model up to depth D.

For long-memory non-tree sources (English text), the optimality degrades but CTW
still typically *outperforms PPM* in published compression benchmarks (Begleiter &
El-Yaniv 2004).

## 4. Discrete categorical fit

Native at the bit level. Byte level is the product of 8 bit-conditionals:
P(byte = b | context) = prod_{i=0..7} P(bit_i | context, bits 0..i-1 of byte b).
This is a 256-vector for the CharModel.predict() API; argmax for hard, soft otherwise.

## 5. Autoregressive applicability

Native. The streaming context-tree update is literally the definition of an online
autoregressive predictor. CTW is most famously used in *cmix* and *paq* family
compressors which are bit-level autoregressive models of text.

The paq mixer in `submissions/paq_mixer_v3` is in the same family but uses *neural*
gradient mixing; CTW uses Bayesian-optimal mixing — distinct mechanism.

## 6. Roofline analysis

CTW is **bandwidth-bound, irregular-access** — the dominant cost is the random-access
walk down the context tree per bit. There is no Tensor Core opportunity.

For depth D=24, byte-level training over N=5e8 bytes:
- 8 bits / byte * N = 4e9 bit predictions.
- Per bit: walk D=24 nodes, ~5 floating-point ops at each node + 24 random-access reads
  ~= 120 ops + 1 KB streamed from HBM/DRAM per bit.
- Total ops: ~5e11; total HBM traffic: ~4 TB.
- Arithmetic intensity: ~0.1 ops/byte. **Deeply bandwidth-bound.**

At A100 HBM BW 2 TB/s, the wall-clock for 4 TB streamed is ~2 s. But this assumes
contiguous access — the actual access pattern is pointer-chasing through a sparse
trie. Realistic throughput on A100 for unstructured tree walks is closer to ~100
MB/s effective, yielding ~40 s wall time. The PPM survey (`research/gradfree-survey/`)
hit the same wall on pure-Python — CTW will need a C/CUDA implementation.

This is **the wrong hardware** for CTW; the algorithm should ideally run on CPU or a
custom DRAM-streaming kernel. Energy estimate on A100: 5–20 kJ for the streaming pass
(GPU idles at ~150 W while waiting for HBM, * 60-120 s).

## 7. Top references

1. Willems, Shtarkov, Tjalkens 1995, "The Context-Tree Weighting Method: Basic
   Properties", IEEE Trans. Inf. Theory. <https://ieeexplore.ieee.org/document/382012>
   *Original. Proves universality + redundancy bound.*
2. Begleiter, El-Yaniv 2004, "On Prediction Using Variable Order Markov Models", JAIR.
   <https://www.jair.org/index.php/jair/article/view/10394>
   *Comparative survey; CTW vs PPM on text.*
3. Sadakane, Okazaki, Imai 2000, "Implementing the Context Tree Weighting Method for
   Text Compression", DCC.
   <https://www.researchgate.net/publication/3844192>
   *Practical impl notes; bytewise CTW reaches ~2.0 bpc on text.*
4. Veness, Ng, Hutter, Bowling 2010, "A Monte-Carlo AIXI Approximation", JAIR.
   <https://arxiv.org/abs/0909.0801>
   *CTW used as the universal sequence model in MC-AIXI.*
5. Mahoney 2005, "Adaptive Weighing of Context Models for Lossless Data Compression".
   <http://mattmahoney.net/dc/dce.html>
   *PAQ context-mixing review — clarifies relation between CTW and the existing
   PAQ-mixer submission.*

## 8. Limitations / failure modes

- **Bandwidth-bound on GPU.** The natural substrate is CPU. The A100 wall-clock will
  not be dramatically better than a fast CPU implementation. **But energy is the
  metric, and idle-A100 is 50 W; if we can finish in 60 s the energy is sub-3 kJ.**
- **Tree memory.** Depth D=24, alphabet 256: worst-case nodes = (256)^24 — infeasible.
  In practice, the tree is sparsely populated and a hash-trie keeps memory at
  ~5-20 MB per 10 MB of train (empirical from PPM logs in `research/gradfree-survey/`).
  Mitigation: cap nodes, prune by min-count.
- **No published WikiText-103 byte-level CTW result with timing on modern hardware.**
  Published CTW bpc on enwik8 is ~1.9 — corresponding to char-acc ~0.71.
- **Eval-time update.** The CharModel API allows `observe()` to keep updating counts
  during eval; this is allowed and helps for non-stationary text (Wikipedia per-article
  vocabulary).

## 9. Experiment spec

**Setup.**
- Bit-level CTW with byte-aligned context depth D=24 bits (= 3 bytes of byte context
  at the deepest, but the contributing weighted mixtures span deeper bit-level
  contexts via the bit-tree structure).
- Hash trie with cap of 2e7 nodes; LRU-eviction on rare nodes.
- Pure CUDA kernel for the streaming pass; the tree itself lives in GPU HBM as a
  hash-array struct-of-arrays.
- Online updates during eval (allowed by CharModel.observe()).

**Implementation.**
- Reference Python CTW: github:fumin/ctw.
- For Modal: port to CuPy + Numba or hand-rolled CUDA. **First pass should use a
  fast C implementation via ctypes** — accept the bandwidth-bound penalty, target
  60-90 s wall on the 500 MB train set.
- Actual byte-level prediction: 8 bit-CTW evaluations per output byte, multiplied
  to get the byte distribution.

**CharModel translation.**
- `predict()`: walk current 24-bit context, accumulate weighted mixture per the 8
  candidate next bits, multiply into 256-byte distribution. ~5K ops per char on
  CPU/A100; ~3-10 us per char.
- `observe(c)`: update KT counts along 8 active contexts (one per bit of c).
- `reset()`: zero context; tree state persists across reset (it's the model).

**Energy budget.** Training: 30–90 s, 1–5 kJ. **This is the lowest-energy submission
plausibly hitting the floor.**

**Char-acc ceiling estimate.** Published CTW bpc 1.85–2.0 on enwik8 maps to char-acc
0.65–0.73. **Plausibly clears 0.70.** Strong outcome since the cost is sub-5 kJ.

## 10. Verdict — **Tier A — Run this first**

Cheapest plausible submission. Closest cousin (PPM order-7) cleared 0.63 at 633 J in
the gradfree-survey; CTW's Bayesian-optimal mixing gives strictly higher per-byte
compression than PPM (Begleiter 2004 survey). The energy could be sub-1 kJ if a
streaming C implementation lands within 300 s wall.

The risk is engineering: writing a fast bit-CTW kernel for A100 (or running it on
CPU and accepting the idle-W penalty for the joule count). **Treat as the cheap
fast-failure / capability demo run.**
