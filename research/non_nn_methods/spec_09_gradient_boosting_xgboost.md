# Spec 09 — Gradient-boosted decision trees (XGBoost / LightGBM) next-byte LM

## 1. Method & mechanism

Train a multi-class gradient-boosted forest with 256 output classes (one per byte
value). Features are byte-context-window encodings — typically n-gram presence
indicators, hashed n-grams, or position-byte one-hots — produced by a fixed hashing
pipeline (no learned embedding). XGBoost (Chen & Guestrin 2016) or LightGBM (Ke 2017)
fits an additive ensemble F(x) = sum_k f_k(x) of regression trees, each chosen to
minimize a regularized softmax cross-entropy objective via Newton-Raphson updates on
the second-order Taylor expansion of the loss.

Mathematically: at boosting iteration k, given current F_{k-1}(x), fit a new tree f_k
to the negative gradient of the loss (the "pseudo-residual"), with the leaf values
set in closed form by the ratio of gradient sum to hessian sum per leaf — no
backprop, no SGD.

## 2. Why not a neural network / not backprop

The architecture is an ensemble of *decision trees*, not an MLP or any
backprop-trained network. Each tree is fit by greedy split-finding on histograms of
(feature, gradient, hessian) tuples — a purely combinatorial search. The boosting
"gradient" here is the Newton-Raphson Taylor expansion of the loss function w.r.t. the
model's *output*, **not** w.r.t. internal weights. There is no chain rule
backpropagation through layers.

## 3. Universal approximation status

**Proven.** Gradient boosting with decision-tree base learners is a universal
function approximator on R^d → R^V — any continuous function can be approximated to
arbitrary accuracy by a sufficiently large tree ensemble (Breiman 2001 random
forests UAT; the same argument carries to boosted ensembles).

## 4. Discrete categorical fit

256-class softmax loss in XGBoost / LightGBM is standard `multi:softprob` objective.
Output: 256-class probability vector. The CharModel `predict()` reads this directly
(or takes argmax for the hard version).

## 5. Autoregressive applicability

Standard sliding-window K. At each position, evaluate the boosted forest on the
K-byte (or n-gram-feature) context vector to get the 256-class distribution.

**Not commonly used for AR sequence modeling.** Decision-tree LMs are sparse in the
literature — the closest is XGBoost / LightGBM for next-word prediction with hand-
engineered n-gram features (e.g., LightGBM Kaggle competitions on text auto-complete
do this informally). **Novel application at byte-LM scale.** This is a real
capability-claim experiment: can a tree ensemble clear 0.70 on raw text bytes?

## 6. Roofline analysis

Decision-tree training and inference on a GPU is the awkward case — XGBoost has a
GPU histogram backend (`gpu_hist`) but its arithmetic intensity is lower than
matmul. Per-split: histogram aggregation across ~5e6 samples per feature, ~1000
features ~= 5e9 ops; per-tree ~30 splits = 1.5e11 ops; for 200 trees: 3e13 ops.

For comparison, a single bf16 matmul of (5e6, 4096) by (4096, 256) is 1e13 ops —
boosted trees do *more* work than the closed-form ridge of spec_02 but at lower
arithmetic intensity (~10–50 ops/byte vs ~4000 ops/byte for spec_02).

**Mixed: not Tensor-Core-friendly, but XGBoost's GPU hist implementation is reasonably
well-tuned. Expect ~30–50% of A100 peak elementwise FLOPs.**

Inference: forest of 200 trees, depth 8 = ~50K nodes. Per char: 200 trees * 8 levels
= 1600 conditional branches per char + a small log-sum-exp at the end. ~2e3 ops per
char on CPU; ~5e2 ops on GPU (if batched). 60K char eval: trivial.

## 7. Top references

1. Chen, Guestrin 2016, "XGBoost: A Scalable Tree Boosting System", KDD.
   <https://arxiv.org/abs/1603.02754>
   *XGBoost original.*
2. Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu 2017, "LightGBM: A Highly Efficient
   Gradient Boosting Decision Tree", NeurIPS.
   <https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html>
   *LightGBM, faster than XGBoost on big data.*
3. Mason, Baxter, Bartlett, Frean 1999, "Boosting Algorithms as Gradient Descent",
   NeurIPS. *Boosting-as-gradient-descent foundation.*
4. Friedman 2001, "Greedy Function Approximation: A Gradient Boosting Machine",
   Ann. Statist. <https://projecteuclid.org/euclid.aos/1013203451>
   *Gradient boosting framework.*

## 8. Limitations / failure modes

- **Feature engineering is the load-bearing decision.** A raw byte one-hot window will
  not work well — there are 2^(K*8) possible windows and the histogram-split learner
  needs continuous-valued features. Use **hashed n-gram counts** over the K-byte
  window: hash all 1-, 2-, 3-grams in the window to a 4096-bucket feature vector.
- **Time budget.** XGBoost with multi:softprob over 256 classes does *one tree per
  class per round*, so 200 rounds = 200 * 256 = 51200 trees. This is heavy. Consider
  one-vs-rest binarized or hierarchical (predict bit-by-bit, 8 bits = 8 binary
  classifiers, then multiply).
- **Sample size.** A100 GPU XGBoost can fit ~5e6 samples in a few minutes; tight
  for the 300 s budget.
- **No published WikiText-103 byte-level XGBoost result** — the experiment will
  produce a first calibrated number, regardless of outcome.

## 9. Experiment spec

**Setup.**
- Context window: K=16 bytes.
- Feature engineering: at each position t, build a 4096-dim sparse-binary feature
  vector by hashing all 1-, 2-, 3-grams in the K-byte window into 4096 buckets
  (with sign/feature-hashing trick to reduce collisions).
- Decompose output: 8 binary classifiers (one per bit) rather than 256-class softmax;
  multiply bit-probs to form byte distribution.
- XGBoost `gpu_hist` on A100, max_depth=8, learning_rate=0.1, n_estimators=200.

**Implementation.**
- `pip install xgboost` in submission (already in the Modal image's torch lineage, but
  verify). LightGBM also viable.
- Streaming feature extraction: 5e6 positions * 4096 features as sparse CSR ~= 800
  MB; fits in HBM.
- Fit each of the 8 binary classifiers in sequence; total training ~120 s on A100.

**CharModel translation.**
- `predict()`: extract features for current 16-byte window, score 8 binary models →
  multiply to 256-byte distribution. ~5 ms / char. 60K eval: 5 min.
  Risk: eval may need batching to fit in eval time budget.
- `observe(c)`: append to ring buffer.

**Energy budget.** Training: 60–150 s, ~10–30 kJ. Eval: separate, not metered.

**Char-acc ceiling estimate.** 0.50–0.65. Tree ensembles are not particularly
well-suited to long-range character dependencies; without learned representations
they will rely on whatever n-gram structure leaks through the hashed features.

## 10. Verdict — **Tier B**

Strong capability demo for "can off-the-shelf gradient boosting do byte LM at all?".
Mixed roofline; not the cheapest method but also not the most expensive. The
hashed-n-gram-feature pipeline doubles as a useful diagnostic for what features
*any* shallow learner needs to clear 0.70.
