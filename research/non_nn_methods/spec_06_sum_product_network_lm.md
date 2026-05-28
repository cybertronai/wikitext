# Spec 06 — Sum-Product Network / Probabilistic Circuit LM

## 1. Method & mechanism

A Sum-Product Network (SPN) is a rooted DAG of three node types: **leaves** = univariate
distributions over a single byte position; **sum nodes** = convex combinations of children
(latent mixtures with learned weights); **product nodes** = factorizations across disjoint
scope sets. The joint probability over a K-byte window is the value at the root computed
by single forward pass — every probabilistic query is linear in the network size.

For a char-LM:

- Window of K bytes (c_{t-K+1}, ..., c_{t+1}).
- Leaf distributions: univariate categorical over 256 bytes per position.
- Structure: a hierarchical mixture (LearnSPN, Gens & Domingos 2013) or a learned
  region graph (Peharz 2020 RAT-SPN; Liu, Peharz, Van den Broeck 2024 PyJuice).
- Training by **online EM** or **online closed-form moment matching** — no backprop.
- Conditional prediction P(c_{t+1} | c_{t-K+1}..c_t) by *evidence marginalization*: set
  the leaves of positions t-K+1..t to their observed values, marginalize over c_{t+1}
  to obtain its normalizer, evaluate at each candidate c_{t+1} for the conditional.

## 2. Why not a neural network / not backprop

SPN training canonically uses one of:
- **Online EM** (Hsu, Anandkumar 2012 method-of-moments; Poon & Domingos 2011 hard EM):
  closed-form M-step updates over sum-node weights given soft assignments from the E-step.
- **Hard EM with mini-batches** (Vergari, Di Mauro, Esposito 2015): integer counts per
  cluster, no gradient.
- **Structure learning** (LearnSPN, ID-SPN): recursive partition-and-cluster algorithm
  driven by mutual information on the data.

None of these involve chain-rule backprop. PyJuice (Liu 2024) does support gradient-based
fine-tuning but the *initial* fit is EM-based.

## 3. Universal approximation status

**Proven.** Any joint distribution over a finite set of discrete random variables can be
represented exactly by a sufficiently large SPN (Peharz 2015 thesis, Theorem 4.1).
Practical SPNs are bounded by the chosen structure / max scope size; with a sufficiently
deep / wide SPN over a K-byte window, conditional next-byte prediction is exact in the
limit.

## 4. Discrete categorical fit

Native. The leaf distributions are categorical over the 256-byte alphabet by construction.
Conditional P(c_{t+1} | context) is one forward pass per candidate next byte (256 forward
passes per char) — or, more efficiently, one forward pass with the candidate variable
left unobserved, then the marginal over its values is read directly from the network.

## 5. Autoregressive applicability

Cheng, Kok, Pham, Chieu, Chai 2014 published an SPN-LM evaluated on n-gram next-word
prediction with K=4; reported "took less time to train than RNNs" but the absolute
accuracy was not competitive with neural LMs. Dynamic SPNs (Melibari, Poupart, Doshi 2016
arxiv 1511.04412) generalize to sequences of arbitrary length via repeated template
networks — directly applicable to streaming next-byte prediction.

## 6. Roofline analysis

This is the spec where the **HBM-vs-compute tension is sharpest**.

A typical RAT-SPN (Peharz 2020) of depth D=6 with branching factor B=4 has
~B^D = 4096 product nodes at the root level, ~16K sum-node parameters, ~1M leaf
parameters. Training: one online EM pass over N=5e6 windows.

- Per-window forward: O(network size) ~= 1e6 ops, plus 256-way conditional eval.
- Per-window E-step: another forward pass.
- Total: N * 2 * 1e6 = 1e13 ops; A100 elementwise: ~3 s wall.

But the dominant cost is **non-contiguous memory access patterns** — SPN nodes are a
DAG, not a regular tensor; child indices are sparse and irregular. PyJuice's primary
contribution is a CSR-style layout that enables batched GPU evaluation (Liu 2024
reports 1-2 orders of magnitude speedup over previous SPN libraries).

Arithmetic intensity estimate for PyJuice-style layout: ~5 FLOPs/byte (per their paper
sect. 5). **Strongly bandwidth-bound** — every node value is a random-access from HBM.
This is the worst-fitting method in the portfolio for the A100 roofline.

Memory: ~10 MB params + activations; fits in HBM with ample room.

## 7. Top references

1. Poon & Domingos 2011, "Sum-Product Networks: A New Deep Architecture", UAI.
   <https://arxiv.org/abs/1202.3732>
   *SPN original.*
2. Cheng, Kok, Pham, Chieu, Chai 2014, "Language Modeling with Sum-Product Networks",
   Interspeech. <https://www.comp.nus.edu.sg/~skok/papers/is14.pdf>
   *The only published SPN-LM paper. Word-level, K=4, no SOTA.*
3. Liu, Peharz, Van den Broeck 2024, "Scaling Tractable Probabilistic Circuits: A Systems
   Perspective". <https://arxiv.org/abs/2406.00766>
   *PyJuice library, GPU-native, modern.*
4. Peharz, Lang, Vergari, Stelzner, Molina, Trapp, Van den Broeck, Kersting, Ghahramani
   2020, "Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits", ICML.
   <https://arxiv.org/abs/2004.06231>
   *EinsumNets — modern fast PC layout.*
5. Choi, Vergari, Van den Broeck 2024 review, "Building Expressive and Tractable
   Probabilistic Generative Models: A Review". <https://arxiv.org/abs/2402.00759>
   *State of the art as of 2024.*

## 8. Limitations / failure modes

- **Bandwidth-bound; will not approach modded_nanogpt energy.** The known-good
  PyJuice implementation is 1-2 OOM faster than legacy SPN code but still loses to
  dense Tensor-Core matmul methods on A100.
- **Capacity at small network size** is limited — Cheng 2014 word-level SPN-LM was
  not competitive with neural LMs even at its publication.
- **Structure learning is offline** — LearnSPN takes tens of minutes for nontrivial
  data; need to use a pre-fixed structure (RAT-SPN) and only learn parameters.
- **256-way per-position categorical leaves** are not standard PyJuice — need custom
  leaf implementation.
- **No published byte-level char-LM SPN result**; ceiling estimate is 0.40–0.55,
  unlikely to clear 0.70.

## 9. Experiment spec

**Setup.**
- RAT-SPN structure: K=16 byte window, depth=6, num_sum_layers=4, replicas=3.
- Categorical leaves over 256 bytes.
- Online EM (mini-batch hard EM, batch size 1024) for 3 epochs over 1M windows.
- PyJuice for GPU evaluation.

**Implementation.**
- `pip install pyjuice` in submission. Verify Modal install latency (<60 s expected).
- Define RAT region graph; instantiate categorical leaves; fit by `pyjuice.train` with
  EM updates.

**CharModel translation.**
- `predict()`: marginalize last position; for each candidate byte, evaluate forward
  pass conditioned on the K-1 observed bytes and the candidate. Cost: 256 * one forward.
  ~5 ms / char on A100.
- `observe(c)`: append to ring buffer.

**Energy budget.** Training: 60–120 s, ~15–30 kJ (bandwidth-bound). Inference: ~5 min
on 60K val chars — *risk of exceeding eval budget if not batched aggressively*.

**Char-acc ceiling estimate.** 0.40–0.55. Cheng 2014 word-level result mapped to bytes
suggests this neighborhood. **Likely DQ on 0.70.**

## 10. Verdict — **Tier C**

Bandwidth-bound, likely DQs at 0.70, has no published byte-LM result to verify against.
Run only if the higher-priority specs all clear; the deliverable is the first
PyJuice-on-WikiText capability number, useful for the PC community but unlikely to
inform the energy frontier on this benchmark.
