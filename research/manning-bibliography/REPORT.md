# Manning Collaborator Graph — Energy-Efficient LM Research Directions

**Date:** 2026-05-17
**Task:** WikiText-103 LM, A100/300s, val char-acc ≥ 0.70, ranked by NVML joules. Baseline: modded_nanogpt 51.7 kJ.
**Framing:** Capability demo, not leaderboard. Char-level is a universal scoring interface; word/BPE/phrase-level methods are first-class. Mechanism novelty + scaling story > incremental wins.
**Scope:** Manning's students, postdocs, frequent co-authors and their academic descendants.

## Summary

The collaborator graph yields stronger candidates than Manning's own bibliography. Three branches stand out:

1. **Retrieval-as-memory (Khandelwal+):** kNN-LM and RetoMaton — gradient-free at inference, scales trivially with stored data.
2. **State-space models (Hazy Research, Manning-adjacent via Stanford):** Hyena reports a direct **20% training compute reduction on WikiText-103**.
3. **EM-based unsupervised (Berg-Kirkpatrick, Klein-line):** featurized EM is the most "truly gradient-free" pretraining mechanism with real scaling behavior.

---

## Branch 1: Urvashi Khandelwal — retrieval-as-memory (HIGHEST PRIORITY)

### kNN-LM (Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis, ICLR 2020) — LOW-HANGING FRUIT
https://arxiv.org/abs/1911.00172

Linearly interpolate a parametric LM with a kNN model over a datastore keyed by the LM's left-context embedding and valued by the next token. The datastore can grow without retraining; the paper hit SOTA 15.79 perplexity on WikiText-103 with **zero additional training**.

- **Scaling:** excellent — perplexity improves monotonically with datastore size, no retraining
- **Energy angle:** train a small parametric LM cheaply, then amortize: build datastore once, query at inference. Capacity scales with disk, not training joules.

### RetoMaton (Alon, Xu, He, Sengupta, Roth, Neubig, ICML 2022) — LOW-HANGING FRUIT
https://arxiv.org/abs/2201.12431

Cluster kNN-LM datastore into automaton states with learned transitions; only invoke kNN search when needed.

- **83% reduction in kNN searches at iso-perplexity** OR 1.85 perplexity reduction at iso-cost
- This is the energy-aware version of kNN-LM — pairs perfectly with the joule metric

### RETRO (Borgeaud et al., DeepMind, ICML 2022) — RESEARCH-GRADE
https://arxiv.org/abs/2112.04426

Chunked cross-attention to retrieved neighbors at training time. GPT-3 quality with 25× fewer parameters when given 2T retrieval tokens — strongest scaling-with-data-not-compute claim in the literature.

---

## Branch 2: Hazy Research / state-space models (HIGH PRIORITY)

### Hyena Hierarchy (Poli et al., ICML 2023) — LOW-HANGING FRUIT
https://arxiv.org/abs/2302.10866

Replaces attention with implicit long convolutions + multiplicative gating. **Matches transformer quality on WikiText-103 with 20% less training compute** at seq length 2K. Direct claim of ~20% joule savings on our exact benchmark.

### Mamba (Gu, Dao, 2023) — LOW-HANGING FRUIT
https://arxiv.org/abs/2312.00752

Selective SSM with content-dependent ∆,B,C. 5× higher generation throughput than transformers. Off-the-shelf via `mamba-ssm` package. Mamba-3B matches transformers of 2× size.

### S4 (Gu, Goel, Ré, ICLR 2022) — RESEARCH-GRADE
https://arxiv.org/abs/2111.00396

Original SSM with HiPPO parameterization. O(N) memory vs attention's O(N²). 60× faster generation reported.

### H3 (Fu, Dao, Saab, Thomas, Rudra, Ré, ICLR 2023) — RESEARCH-GRADE
https://arxiv.org/abs/2212.14052

Diagnosed pure SSMs' weakness on associative recall, proposed hybrid H3+attention (2 attention layers suffice) matching transformer perplexity.

### FlashAttention (Dao et al., NeurIPS 2022)
https://arxiv.org/abs/2205.14135

Already standard in modded_nanogpt-style codebases — flagged so the baseline isn't double-counting it.

---

## Branch 3: Berg-Kirkpatrick / Klein-line EM-based unsupervised (HIGH NOVELTY)

### Painless Unsupervised Learning with Features (Berg-Kirkpatrick, Bouchard-Côté, DeNero, Klein, NAACL 2010) — RESEARCH-GRADE
https://aclanthology.org/N10-1083/

Each multinomial in a generative model becomes a small logistic regression over features. EM E-step unchanged; M-step is L-BFGS over feature weights. The only "gradient" is inside a convex per-multinomial M-step — **no global backprop, no BPTT**.

- **Mechanism novelty:** the most "gradient-free end-to-end" learning algorithm in this entire survey
- **Energy angle:** EM scales linearly in data, is embarrassingly parallel, dramatically cheaper than transformer training per token
- **Risk:** whether a feature-rich HMM can reach 0.70 char-acc on WikiText — but the framing rewards novel mechanism, not benchmark perfection

### Unsupervised Transcription of Historical Documents (Berg-Kirkpatrick, Durrett, Klein, 2013) — exemplar
Structurally-rich generative model trained completely unsupervised with EM. Demonstrates the methodology.

### Neural CRF Parsing (Durrett & Klein, ACL 2015) — MODERATE
Tiny neural net + exact structured DP inference. Compute dominated by structure, not net. Char-LM analog: latent-structure HMM with tiny neural emissions.

---

## Branch 4: Mitchell / model-editing (RESEARCH-GRADE NOVELTY)

### MEND (Mitchell, Lin, Bosselut, Finn, Manning, ICLR 2022)
https://arxiv.org/abs/2110.11309
Hypernetwork transforms standard fine-tuning gradient into a single-shot edit. One forward pass ≈ one "training step." If pretraining were reformulated as a sequence of learned edits, joule cost per example collapses.

### ROME / MEMIT (Meng, Bau, Andonian, Belinkov et al., 2022/2023)
https://arxiv.org/abs/2202.05262 / https://arxiv.org/abs/2210.07229
Closed-form rank-one updates to MLP weights as a key-value store. MEMIT scales to thousands of simultaneous edits. Speculative pretraining substrate but mechanically distinct from SGD.

### SERAC (Mitchell, Lin, Bosselut, Manning, Finn, ICML 2022)
https://arxiv.org/abs/2206.06520
Semi-parametric edits in external memory; bridge between model-editing and retrieval (Branch 1).

---

## Branch 5: John Hewitt — structural / non-parametric LM

### Backpack Language Models (Hewitt, Thickstun, Manning, Liang, ACL 2023 — Outstanding Paper)
https://arxiv.org/abs/2305.16765

Each vocab item has multiple non-contextual "sense vectors"; predicted word is a **non-negative weighted sum** of sense vectors over the sequence. Sparse, interpretable, intervention-friendly output structure.

- 170M Backpack matches 124M GPT-2; sense vectors *outperform* a 6B transformer's word embeddings on lexical similarity
- **Energy angle:** pretrain sense vectors cheaply via co-occurrence stats (GloVe-style); train only the contextual weighting net. Decomposable training.

### Truncation Sampling as Desmoothing (Hewitt, Manning, Liang, EMNLP Findings 2022)
https://arxiv.org/abs/2210.15191
Frames LMs as true distribution + smoothing prior. Useful framing for sparse-output LMs that match argmax with less softmax compute.

---

## Branch 6: Socher / Salesforce — the WikiText-native branch

### Pointer Sentinel Mixture (Merity, Xiong, Bradbury, Socher, ICLR 2017) — LOW-HANGING FRUIT
https://arxiv.org/abs/1609.07843

**Built for WikiText.** Output = mixture of (vocab softmax) + (pointer over recent context). Sentinel learns the gate. Wikipedia is rampant with repetition (proper nouns, formatting, citations) — pointer carries the long tail at near-zero parameter cost.

- **Char-level analog:** small char softmax + char-position pointer over windowed buffer
- **Energy angle:** saves on output projection for the long tail; smaller parametric backbone

### AWD-LSTM (Merity, Keskar, Socher, 2017) — MODERATE
https://arxiv.org/abs/1708.02182
Strongest documented LSTM training recipe — DropConnect + NT-ASGD averaging.

### QRNN (Bradbury, Merity, Xiong, Socher, ICLR 2017) — MODERATE
https://arxiv.org/abs/1611.01576
Convolutions over time + minimal recurrent pooling. 16× faster than LSTMs at iso-accuracy. Predecessor to Hyena's philosophy.

---

## Branch 7: Iyyer — "simpler is better"

### Deep Averaging Networks (Iyyer, Manjunatha, Boyd-Graber, Daumé III, ACL 2015) — MODERATE
https://aclanthology.org/P15-1162.pdf
Average word embeddings + small FFN matches RecursiveNN at tiny compute. Direct LM port is harder (order matters), but possibly useful as an encoder-side embedding step.

### RankGen (Krishna, Chang, Wieting, Iyyer, EMNLP 2022)
Contrastive prefix-continuation reranker. Less relevant for argmax-acc but a mechanism to study.

---

## Branch 8: Vinyals / Le-adjacent

### Pointer Networks (Vinyals, Fortunato, Jaitly, NeurIPS 2015) — MODERATE
https://arxiv.org/abs/1506.03134
Output is an index into the input. Conceptual ancestor of copy mechanisms. A pure pointer-LM over a dictionary buffer of common n-grams would be cousin to LZ77 compression — mechanically novel but speculative.

---

## Branch 9: Pennington — random matrix theory

### Dynamical Isometry (Pennington, Schoenholz, Ganguli, NeurIPS 2017)
Trains 10,000-layer vanilla nets via well-conditioned Jacobian initialization. For 300s budget, anything that improves convergence wall-clock is a direct joule save.

---

## Branches with less to offer

- **Drew Hudson (MAC, NSM):** visual reasoning, language port loses grounding
- **Hashimoto data-curation:** WikiText is single-source
- **Toutanova / Schuster / de Marneffe (UD):** parsing infrastructure, not LM mechanisms
- **Iyyer's recent long-form work:** quality reranking with frozen LMs

---

## Top 5 — Try first

### 1. RetoMaton — Automaton-Augmented kNN-LM (Branch 1)
Direct WikiText-103 substrate. kNN-LM's "scales with stored data" story + **83% fewer kNN searches** = the energy-aware version. Train a tiny char-LSTM for embedding extraction, build automaton over training data, interpolate at inference. Most joules go to building the index once. **Highest expected joule reduction × novelty.**

### 2. Hyena Hierarchy (Branch 2)
Subquadratic, drop-in attention replacement with a published **20% training-compute saving on WikiText-103 specifically**. Mature implementations exist. The least-risky "novel mechanism beats modded_nanogpt" submission.

### 3. Pointer Sentinel Mixture at char level (Branch 6)
Built *for WikiText*. At char level the pointer copies from a windowed buffer of recent chars — Wikipedia is extremely repetitive at the char level. Pointer branch has near-zero parameter cost; parametric model gets smaller.

### 4. Berg-Kirkpatrick featurized EM as pretraining (Branch 3)
Most mechanism-novel entry. Featurized HMM/PCFG over characters trained via EM — no backprop. Use either standalone or as a feature extractor / initialization for a small downstream gradient-trained head. **Submitting an EM-pretrained model that clears 0.70 with a tiny head is exactly the kind of capability demo the project wants.**

### 5. Mamba (Branch 2)
Off-the-shelf, mature, single-import. Mamba blocks in place of attention is a one-day experiment with credible 5× inference speedup. The "safe novelty" companion to the riskier mechanisms above.

---

## Sources

- [kNN-LM](https://arxiv.org/abs/1911.00172) · [RetoMaton](https://arxiv.org/abs/2201.12431) · [RETRO](https://arxiv.org/abs/2112.04426)
- [S4](https://arxiv.org/abs/2111.00396) · [H3](https://arxiv.org/abs/2212.14052) · [Hyena](https://arxiv.org/abs/2302.10866) · [Mamba](https://arxiv.org/abs/2312.00752) · [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Painless Unsupervised Learning](https://aclanthology.org/N10-1083/) · [Neural CRF Parsing](https://www.cs.utexas.edu/~gdurrett/papers/durrett-klein-acl2015.pdf)
- [MEND](https://arxiv.org/abs/2110.11309) · [ROME](https://arxiv.org/abs/2202.05262) · [MEMIT](https://arxiv.org/abs/2210.07229) · [SERAC](https://arxiv.org/abs/2206.06520)
- [Backpack LMs](https://arxiv.org/abs/2305.16765) · [Truncation as Desmoothing](https://arxiv.org/abs/2210.15191)
- [Pointer Sentinel](https://arxiv.org/abs/1609.07843) · [AWD-LSTM](https://arxiv.org/abs/1708.02182) · [QRNN](https://arxiv.org/abs/1611.01576)
- [DAN](https://aclanthology.org/P15-1162.pdf) · [Pointer Networks](https://arxiv.org/abs/1506.03134)
