# Forward-Forward Literature Survey (Phase 1)

**Audience.** Phase-3 design for the WikiText-103 byte-level char-LM competition (300 s on A100-80GB, val char-acc floor 0.70, ranked by training energy, no end-to-end backprop allowed).

**What this survey is.** A verified taxonomy of Forward-Forward (FF) variants from 2022 through 2025, plus a small set of sibling forward-only / local-learning families that papers commonly group with FF. Each entry below was sourced from an arxiv (or publisher) page I actually fetched; URLs are recorded inline.

**What this survey is not.** It is not a re-derivation. Algorithmic summaries are short and operational, written to answer "is it worth re-implementing on top of pass-2's `submission.py`?". Numbers are headline numbers as the papers reported them — no re-running.

**Scope.** 20 entries cap, ranked by *portability score* (1–5) for byte-level WikiText-103 char-LM under a 300 s wall budget. Portability is a working judgement, not a derived quantity. Two known facts colour it heavily:

- **Almost every FF paper benchmarks on vision (MNIST/CIFAR/ImageNet).** Of all 20 entries, **only one** (Gandhi & Gala, 2023) actually evaluates an FF variant on a text task at all, and even that is sentence-level sentiment, not character-level LM.
- **No paper in the 2022–2025 corpus reports char-level WikiText accuracy under any FF variant.** Confirmation that this is an empirically open question, not a closed-form re-application.

A score of 4–5 means the algorithmic core (negative generation / goodness function / readout) is cheap to swap into a 5-layer FC FF stack with one byte-context input encoding, and there is at least suggestive evidence the rule transfers to non-vision modalities. A score of 1–2 means the variant is tightly coupled to convolutional vision pipelines or to architectures (Vision Transformers, deep CNNs, SNNs) that don't fit a 300 s budget on bytes.

---

## Summary table (ranked by portability score, descending)

| # | Variant | Year | Citation | Modifies axis | Best reported result | Portability | Effort |
|---|---|---|---|---|---|---|---|
| 1 | SymBa-FF | 2023 | [arxiv:2303.08418](https://arxiv.org/abs/2303.08418) | loss, schedule | MNIST competitive w/ FF; faster convergence | **5** | small |
| 2 | Mono-Forward | 2025 | [arxiv:2501.09238](https://arxiv.org/abs/2501.09238) | goodness, loss, readout | Matches/exceeds BP on MLP MNIST/CIFAR; 41% less energy | **5** | small |
| 3 | Channel-wise Competitive FF (CwC) | 2023, AAAI 2024 | [arxiv:2312.12668](https://arxiv.org/abs/2312.12668) | negatives, goodness | CIFAR-10 78.1% (no negative data) | **4** | small |
| 4 | DeeperForward | ICLR 2025 | [openreview kOYnXVQCtA](https://openreview.net/forum?id=kOYnXVQCtA) | goodness, layer-norm | 17-layer CNN on CIFAR-10 (best FF) | **4** | small |
| 5 | Self-Contrastive FF (SCFF) | 2024 / Nat. Comm. 2025 | [arxiv:2409.11593](https://arxiv.org/abs/2409.11593) | negatives | CIFAR-10 SOTA local; extends to RNNs | **4** | small |
| 6 | Cascaded Forward (CaFo) | 2023 | [arxiv:2303.09728](https://arxiv.org/abs/2303.09728) | negatives, predictor, schedule | 4 image benchmarks; no negatives | **4** | medium |
| 7 | FF for NLP (Gandhi & Gala) | 2023 | [arxiv:2307.04205](https://arxiv.org/abs/2307.04205) | training schedule, threshold | First FF on IMDb sentiment; weak | **4** | small |
| 8 | Hyperspherical FF (HFF) | 2026 preprint | [arxiv:2605.00082](https://arxiv.org/abs/2605.00082) | goodness, loss, predictor | CIFAR-10 83.1% CNN; ImageNet >25% top-1 | **4** | medium |
| 9 | Plain Hinton FF (baseline) | 2022 | [arxiv:2212.13345](https://arxiv.org/abs/2212.13345) | (control) | MNIST 1.36%; CIFAR-10 41% | **3** | (already done) |
| 10 | Distance-Forward (DF) | 2024 | [arxiv:2408.14925](https://arxiv.org/abs/2408.14925) | goodness, loss | CIFAR-10 88.2%; <40% memory of BP | **3** | medium |
| 11 | Contrastive FF (ViT) | 2025 | [arxiv:2502.00571](https://arxiv.org/abs/2502.00571) | negatives, loss | +10% over FF baseline on ViT image cls | **3** | medium |
| 12 | FFCL (Ahmed) | MIDL 2023 | [arxiv:2305.02927](https://arxiv.org/abs/2305.02927) | training schedule, predictor | +3.69% on pneumonia X-ray | **3** | medium |
| 13 | Forward Target Propagation (FTP) | 2025 | [arxiv:2506.11030](https://arxiv.org/abs/2506.11030) | sibling: target prop | MNIST/CIFAR competitive; RNN sequential | **3** | medium |
| 14 | PEPITA | ICML 2022 | [arxiv:2201.11665](https://arxiv.org/abs/2201.11665) | sibling: error-modulated 2nd pass | CIFAR-10/100 close to BP | **3** | medium |
| 15 | FAUST (similarity-based FF) | 2025 preprint | [arxiv:2509.08697](https://arxiv.org/abs/2509.08697) | goodness, loss | CIFAR-10 56.2% MLP (close to BP 57.6%) | **3** | medium |
| 16 | Scalable FF (SFF) | 2025 | [arxiv:2501.03176](https://arxiv.org/abs/2501.03176) | backbone, goodness | MobileNetV3/ResNet18 vs BP | **2** | medium |
| 17 | Adaptive Spatial Goodness Encoding (ASGE) | ICASSP 2026 | [arxiv:2509.12394](https://arxiv.org/abs/2509.12394) | goodness, backbone | First FF on ImageNet (51.58% top-1) | **2** | large |
| 18 | Covariance-Aware Goodness (BiCovG) | 2026 preprint | [arxiv:2605.04346](https://arxiv.org/abs/2605.04346) | goodness | ImageNet-100 73.0%, VGG-16 | **2** | large |
| 19 | FF for SNNs (Ghader et al.) | 2025 | [arxiv:2502.20411](https://arxiv.org/abs/2502.20411) | backbone (SNN) | SHD, N-MNIST competitive w/ BP-SNN | **1** | large |
| 20 | Direct Feedback Alignment (DFA) | 2020 base, scaling 2020 | [arxiv:2006.12878](https://arxiv.org/abs/2006.12878) | sibling: random feedback | Transformer LM trained w/ DFA on 100M tokens | **2** | large |

Numbers in the "Best reported" column are exactly as reported by the cited paper; many are top-1 / test-error / accuracy and units are not normalised here.

---

## Detailed entries

### 1. SymBa-FF — Symmetric Backpropagation-Free FF

- **Citation.** Lee & Song, 2023. *"SymBa: Symmetric Backpropagation-Free Contrastive Learning with Forward-Forward Algorithm for Optimizing Convergence."* arxiv:2303.08418. URL fetched: https://arxiv.org/abs/2303.08418
- **Summary.** Three modifications on top of plain Hinton FF: (i) symmetric gradient balancing to fix the asymmetry between positive-pull and negative-push directions, (ii) explicit loss balancing between the two halves, and (iii) an Intrinsic Class Pattern (ICP) auxiliary that carries class info through the stack to prevent it being washed out by L2 normalisation. The rule itself stays sum-of-squares goodness with a logistic-on-(G − θ) loss; what changes is the optimisation geometry. Third-party reimplementations report it "converges more quickly to ~2.2% MNIST error after 60 epochs" vs. plain FF.
- **Axis modified.** Loss formulation; training schedule.
- **Reported result.** Faster convergence than FF on MNIST and CIFAR. Abstract claims improvement over both FF and BP. Exact numbers not isolatable from abstract; for verified MNIST ~2.2% number see the dah33 reimplementation (https://github.com/dah33/explore_forward_forward).
- **Code.** No official repo found; reimplementation exists at https://github.com/dah33/explore_forward_forward.
- **Portability (5).** SymBa is a drop-in modification of pass-2's loss function. No architecture change, no negative-generation change, no readout change. The ICP idea — pumping class info into the activation stream — is *especially* worth porting to char-LM, where the "class" is the next byte and goodness can be class-conditioned trivially.
- **Effort.** Small (≤1 day from pass-2's submission.py).

### 2. Mono-Forward — FF with cross-entropy goodness

- **Citation.** Gong, Li & Abdulla, 2025. *"Mono-Forward: Revisiting Forward-Forward through Objective-Locality Decomposition."* arxiv:2501.09238. URL fetched: https://arxiv.org/abs/2501.09238 (most recent version dated 2026 in the listing).
- **Summary.** Decomposes FF into "locality" and "goodness" components and argues empirically that locality is fine — it's the contrastive sum-of-squares goodness that holds FF back. Replaces goodness with **standard multi-class cross-entropy applied locally at each layer**, using a per-layer linear projection to logits. No positive/negative pass at all: each layer just does a local softmax-CE on labels using its own activations. The follow-up paper (Spyra & Dzwinel 2025, [arxiv:2509.19063](https://arxiv.org/abs/2509.19063) and [arxiv:2511.01061](https://arxiv.org/abs/2511.01061)) re-validates this with rigorous energy measurements using NVML — *exactly the methodology this competition uses* — and reports 41% less energy and 34% less time than tuned BP at matching accuracy.
- **Axis modified.** Goodness function; loss formulation; predictor/readout (collapses into one).
- **Reported result.** "Consistently matches or surpasses BP" on MLPs across MNIST, CIFAR-10, CIFAR-100, and several PathMNIST tasks. MLP-Mixer on PathMNIST beats BP with 31% of BP's memory. **No language-modelling results.**
- **Code.** No official repo verified in this survey.
- **Portability (5).** This is the most natural FF→char-LM port available. For char-LM the "class" is the next-byte ID (256 classes). Each FF layer becomes: `logits_l = W_l^head · LN(a_l)`, optimised with CE on the actual byte target. Closed-form ridge readout becomes redundant — every layer is its own classifier. The energy methodology in the follow-up paper aligns precisely with this competition's NVML-based ranking.
- **Effort.** Small. Pass-2's goodness loss replaced with per-layer cross-entropy plus a tied 256-way projection per layer.

### 3. Channel-wise Competitive FF (CwC) — AAAI 2024

- **Citation.** Papachristodoulou, Kyrkou, Timotheou & Theocharides, 2023. *"Convolutional Channel-wise Competitive Learning for the Forward-Forward Algorithm."* arxiv:2312.12668, AAAI 2024. URL fetched: https://arxiv.org/abs/2312.12668
- **Summary.** Partitions output channels of each layer into class-aligned groups; goodness becomes the activation energy *within the channel group corresponding to the true label*, normalised against other groups. Crucially this **eliminates the need for negative data**: the positive sample's wrong-class channel groups are the implicit negatives. Convolutional in the published form, but the channel-group idea is architecture-agnostic.
- **Axis modified.** Negative generation (none needed); goodness function (class-conditional channel partition).
- **Reported result.** MNIST 0.58% error, FashionMNIST 7.69%, CIFAR-10 21.89%, CIFAR-100 48.77% — competitive with BP on small images and ahead of every prior FF method as of late 2023.
- **Code.** https://github.com/andreaspapac/CwComp (Python, official, AAAI 2024).
- **Portability (4).** The "channel groups = classes" idea ports to char-LM if you partition each hidden vector into 256 sub-vectors and define goodness as energy of the sub-vector matching the next byte. This is a clean replacement of the external-unigram and hard-self negative-generation logic used in passes 1 and 2. Penalty for not yet being demonstrated on text.
- **Effort.** Small.

### 4. DeeperForward — ICLR 2025

- **Citation.** Sun, Zhang, He, Wen, Shen & Xie, 2025. *"DeeperForward: Enhanced Forward-Forward Training for Deeper and Better Performance."* ICLR 2025. URL fetched: https://openreview.net/forum?id=kOYnXVQCtA
- **Summary.** Two specific changes to plain FF that together let FF scale to deeper networks: (i) replace L2-normalisation between layers with **LayerNorm** (which preserves activation magnitude information in a controlled way), and (ii) replace `sum(a^2)` goodness with **mean(a^2)** so the goodness statistic doesn't blow up with layer width. Authors also propose a "model parallel" training schedule that uses the locality property.
- **Axis modified.** Goodness function (mean vs sum); layer-norm style.
- **Reported result.** 17-layer CNN on CIFAR-10 successfully trained; "significant advantages over existing FF-based algorithms" on MNIST/Fashion-MNIST/CIFAR-10. Exact scores not in the abstract but reported as best-of-class FF at ICLR 2025.
- **Code.** Not surfaced (review-anonymous note in the OpenReview submission).
- **Portability (4).** Two trivial changes to pass-2: swap L2 for LayerNorm (already used in pass-2's ridge input), swap sum for mean in the goodness statistic. Both are 1–2 line edits. The depth-unlock matters because passes 1/2 used 5–6 layers and the literature consistently shows plain FF stalls at this depth.
- **Effort.** Small.

### 5. Self-Contrastive Forward-Forward (SCFF) — Nature Communications 2025

- **Citation.** Chen, Liu, Laydevant & Grollier, 2024. *"Self-Contrastive Forward-Forward Algorithm."* arxiv:2409.11593; final version Nature Communications, July 2025. URL fetched: https://arxiv.org/abs/2409.11593
- **Summary.** Standard FF needs an external mechanism (or extra label-injection) to generate negatives. SCFF removes this by self-contrasting: a **positive example is the input concatenated with itself**, and a **negative example is the input concatenated with a different random sample from the batch**. The rest of FF (sum-of-squares goodness, logistic loss, layer-wise) is unchanged. Importantly, SCFF demonstrates the rule on **RNNs**, not just MLP/CNN — the only FF paper in this survey that does so.
- **Axis modified.** Negative generation.
- **Reported result.** MNIST 98.7% MLP; SOTA-for-local-methods on CIFAR-10, STL-10, Tiny ImageNet. RNN extension demonstrated but specific sequence-task numbers not in abstract. **No language modelling.**
- **Code.** https://github.com/neurophysics-cnrsthales/contrastive-forward-forward (official, Grollier group).
- **Portability (4).** Self-contrast removes the need for the external-unigram and hard-self negative samplers used in passes 1/2. For char-LM this could translate to: positive = K-context windows of the actual sequence; negative = a K-context window with one byte swapped from a different position. Negative generation is essentially free. RNN demonstration is a positive signal — FF can in principle handle stateful sequence models.
- **Effort.** Small.

### 6. Cascaded Forward (CaFo)

- **Citation.** Zhao, Wang, Li, Jin, Lang & Ling, 2023. *"The Cascaded Forward Algorithm for Neural Network Training."* arxiv:2303.09728; final version published in Pattern Recognition, 2024. URL fetched: https://arxiv.org/abs/2303.09728
- **Summary.** Each "cascaded block" outputs a full **label distribution** directly. No positive/negative passes. Each block is trained independently to minimise cross-entropy between its predicted label distribution and the true label, taking the previous block's output as input (no gradient flow back across blocks). The label-prediction head per block is a small linear layer with its own loss.
- **Axis modified.** Negative generation (none); predictor/readout (one per layer); training schedule (greedy layer-wise).
- **Reported result.** Significant accuracy improvement over plain FF on four image classification benchmarks (specific numbers in body text, not abstract).
- **Code.** https://github.com/Graph-ZKY/CaFo (PyTorch, third-party but referenced).
- **Portability (4).** CaFo is conceptually identical to "every layer is a next-byte classifier" — i.e., it's algorithmically very close to Mono-Forward (#2) but with a greedy training schedule rather than synchronous. For char-LM this gives a stack of 256-way classifiers, one per layer, trained sequentially. Choice between CaFo and Mono-Forward becomes "greedy schedule vs synchronous schedule"; portability of the *idea* is high but the greedy schedule may not fit a 300 s budget well (idle layers waste time).
- **Effort.** Medium — the schedule logic and per-layer head are both small but the orchestration differs from pass-2.

### 7. FF for NLP (Gandhi & Gala et al.) — first text result

- **Citation.** Gandhi, Gala, Kornberg & Sridhar, 2023. *"Extending the Forward Forward Algorithm."* arxiv:2307.04205. URL fetched: https://arxiv.org/abs/2307.04205
- **Summary.** Replicates plain Hinton FF on MNIST, then transports it to **IMDb movie-review sentiment classification** — the first published FF result on a non-vision task. Also introduces a "pyramidal" loss-threshold schedule (θ decreases across layers) which is reported to change test error by up to 8% in some configurations.
- **Axis modified.** Training schedule (θ schedule); domain (text classification, sentence-level).
- **Reported result.** Reports IMDb accuracy as a baseline — "the first instance of the algorithm's extension beyond computer vision." Mixed-to-weak compared to standard text classification baselines. **Note this is sentence/sentiment classification, NOT character-level language modelling.** Weight visualisations show 10–20x larger mean/variance for FF-trained weights vs BP.
- **Code.** Not surfaced in this survey.
- **Portability (4).** The most directly relevant *prior result* — it confirms FF runs on text at all. But the result is weak (the paper itself frames it as "baseline" rather than competitive), and IMDb sentiment is a single-label classification problem fundamentally easier than 256-way next-byte prediction. The pyramidal-θ schedule is a small, cheap tweak worth trying.
- **Effort.** Small (just the θ schedule).

### 8. Hyperspherical Forward-Forward (HFF)

- **Citation.** Sarode, Moser, Folz, Raue, Nauen, Frolov & Dengel, 2026 preprint. *"Hyperspherical Forward-Forward with Prototypical Representations."* arxiv:2605.00082. URL fetched: https://arxiv.org/abs/2605.00082
- **Summary.** Replaces FF's binary "goodness above/below θ" with a multi-class objective on the unit hypersphere: each class is a learned unit-norm prototype; the local objective at each layer is to align the layer's activation (also unit-normed) with the true-class prototype and away from others. **Inference becomes a single forward pass** (one cosine sim per class), eliminating FF's per-class re-rollout. Authors report ">40x inference speedup".
- **Axis modified.** Goodness function (cosine + prototypes); loss formulation; predictor/readout (single-pass argmax).
- **Reported result.** CIFAR-10 MLP 61.93%; CIFAR-10 CNN 83.08% — claimed new SOTA for local-loss methods on CIFAR-10. ImageNet-1k >25% top-1 (greedy), 65.96% w/ transfer learning.
- **Code.** Not surfaced — this is a recent preprint.
- **Portability (4).** Char-LM has 256 "classes" — a perfect fit for prototype-based goodness. Each layer activation is L2-normed (already done in FF passes) and projected to a 256-prototype basis via dot products; goodness is the dot with the true-byte prototype. Single-pass inference is also competition-relevant because inference time is not charged but greedy-argmax over 256 candidates *at every position* is the eval-step cost. HFF would replace passes 1/2's goodness-softmax-over-256-candidates with one matmul.
- **Effort.** Medium — prototypes are new state to maintain; if you tie them to the byte-embedding table you cut state.

### 9. Plain Hinton FF (the baseline)

- **Citation.** Hinton, 2022. *"The Forward-Forward Algorithm: Some Preliminary Investigations."* arxiv:2212.13345. URL fetched: https://arxiv.org/abs/2212.13345
- **Summary.** Two forward passes (positive data, negative data), per-layer sum-of-squares goodness, logistic loss against threshold θ, L2-normalised inputs to next layer to prevent goodness leaking up the stack. Negative samples in §3 are class-mismatched concatenations on MNIST; §4 sketches generative negatives but never benchmarks them at scale.
- **Axis modified.** (Control.)
- **Reported result.** MNIST ~1.36% error (FC); CIFAR-10 ~41% test error with 2–3 layers. The paper explicitly notes "Forward-Forward is somewhat slower than backpropagation."
- **Code.** Official MATLAB (Hinton); widely re-implemented in PyTorch: https://github.com/loeweX/Forward-Forward (Loewe, reimpl), https://github.com/mpezeshki/pytorch_forward_forward (Pezeshki).
- **Portability (3).** This is exactly what pass 1 implemented; pass 2 was a closely-related refinement. Score 3 because we already know the baseline lands at 0.235–0.279 char-acc — that *is* the gap we're trying to close. Listed for completeness as the reference point against which every other entry is measured.
- **Effort.** Already done.

### 10. Distance-Forward Learning (DF)

- **Citation.** Wu, Xu, Wu, Deng, Xu, Wen & Li, 2024. *"Distance-Forward Learning: Enhancing the Forward-Forward Algorithm Towards High-Performance On-Chip Learning."* arxiv:2408.14925. URL fetched: https://arxiv.org/abs/2408.14925
- **Summary.** Reformulates FF goodness as a **centroid-based metric-learning loss** (N-pair margin). Each layer maintains class centroids; positive examples are pulled toward the true-class centroid, negative examples toward a different class centroid, via a margin loss. Also adds a "layer-collaboration" update where each layer's gradient sees a small mix of the next layer's signal (sub-locally).
- **Axis modified.** Goodness function (distance/margin); loss formulation; some cross-layer coupling.
- **Reported result.** MNIST 99.7%, CIFAR-10 88.2%, CIFAR-100 59%, SVHN 95.9%, ImageNette 82.5%. <40% BP memory.
- **Code.** Not surfaced.
- **Portability (3).** Centroids work cleanly for byte-level prediction (one centroid per byte = 256 centroids per layer). But the cross-layer "layer-collaboration" coupling weakens the "strictly local" claim and may run afoul of competition rule 4 (no end-to-end BP across layers) depending on implementation detail — needs careful reading before adoption.
- **Effort.** Medium — N-pair margin is a 20-line module; centroid maintenance adds bookkeeping.

### 11. Contrastive Forward-Forward (Aghagolzadeh & Ezoji, ViT)

- **Citation.** Aghagolzadeh & Ezoji, 2025. *"Contrastive Forward-Forward: A Training Algorithm of Vision Transformer."* arxiv:2502.00571. URL fetched: https://arxiv.org/abs/2502.00571
- **Summary.** Replaces FF's goodness/badness with a supervised contrastive loss (SupCon) applied at each layer. Two augmented views per image; the within-layer loss pulls together same-class views and pushes apart different-class views. Stage 1 trains the encoder layers iteratively with local SupCon; later stages train a head. Targets Vision Transformers specifically.
- **Axis modified.** Loss formulation; negative generation (within-batch).
- **Reported result.** Up to 10% accuracy improvement over baseline FF on ViT; 5–20x faster convergence. Specific dataset/architecture not in abstract.
- **Code.** https://github.com/HosseinAghagol/ContrastiveFF (search-surfaced; not directly verified by fetching the README).
- **Portability (3).** SupCon at each layer requires labels (which we have: next byte = label). The within-batch contrastive idea is good in principle; cost is that batch size must be large enough to give meaningful contrast across 256 classes. ViT-specific implementation details don't transfer.
- **Effort.** Medium — SupCon is a known recipe but layer-wise application on a 5-layer FC is non-trivial.

### 12. FFCL — Forward-Forward Contrastive Learning (Ahmed)

- **Citation.** Ahamed, Chen & Imran, 2023. *"Forward-Forward Contrastive Learning."* arxiv:2305.02927, MIDL 2023. URL fetched: https://arxiv.org/abs/2305.02927
- **Summary.** Three-stage training: (i) local FF-style contrastive pretraining per block, (ii) global contrastive learning across blocks, (iii) final classification head trained with **regular backpropagation**. Targeted at medical imaging pretraining.
- **Axis modified.** Training schedule (multi-stage); predictor/readout.
- **Reported result.** +3.69% over ImageNet-pretrained ResNet-18 on pneumonia X-ray classification.
- **Code.** Not surfaced.
- **Portability (3).** The third stage uses BP, which would violate competition rule 4 if used across layers — adoption would require restricting BP to a single-layer readout head (which we already allow via the ridge readout). The local + global staging is interesting but the global stage looks like it might cross layers.
- **Effort.** Medium.

### 13. Forward Target Propagation (FTP)

- **Citation.** As-Saquib, Abeer, Chien, Yoon, Kumar & Yi, 2025. *"Forward Target Propagation: A Forward-Only Approach to Global Error Credit Assignment via Local Losses."* arxiv:2506.11030. URL fetched: https://arxiv.org/abs/2506.11030
- **Summary.** **Sibling family** (target propagation, not FF strictly). Replaces the backward pass with a second forward pass that propagates *targets* — i.e., desired activations for each layer — using only feedforward computations. Each layer trains itself to match its locally-assigned target. Notably evaluates on **RNNs and long-term dependency tasks**, which is more relevant to language than most vision-only papers.
- **Axis modified.** Sibling: target propagation.
- **Reported result.** Competitive with BP on MNIST/CIFAR-10/CIFAR-100; "effective modeling of long-term dependencies in sequential tasks" (RNN results); outperforms BP under quantised low-precision.
- **Code.** Not surfaced.
- **Portability (3).** Listed because (a) it has sequence/RNN results, (b) it's strictly forward-only, (c) it might literally satisfy competition rule 4. But it's *not* FF — the "goodness" axis doesn't apply, the negative-generation axis doesn't apply, and the implementation is genuinely different from pass-2's submission.py. Worth tracking; lower priority than the FF-proper variants because the engineering distance is larger.
- **Effort.** Medium.

### 14. PEPITA — Error-driven Input Modulation

- **Citation.** Dellaferrera & Kreiman, 2022. *"Error-driven Input Modulation: Solving the Credit Assignment Problem without a Backward Pass."* arxiv:2201.11665, ICML 2022. URL fetched: https://arxiv.org/abs/2201.11665
- **Summary.** **Sibling family.** First forward pass on a clean input. Compute output error. Second forward pass where the **input is perturbed by the error** projected back through a random matrix. The update is the difference of activations between the two passes, applied layer-wise. No backward pass at all.
- **Axis modified.** Sibling: error-modulated 2nd pass (no goodness/negatives axis at all).
- **Reported result.** Competitive with BP on MNIST, CIFAR-10, CIFAR-100. ICML 2022.
- **Code.** https://github.com/GiorgiaD/PEPITA (Python/PyTorch + NumPy, official).
- **Portability (3).** Architecturally cleaner than FF in some respects (no goodness threshold to tune, no negative generation, no layer-norm tricks needed), but uses the *output* error to drive learning — which for char-LM means we need a final softmax-CE head, and PEPITA's update rule needs to propagate the error through random projections to each layer. The two-pass structure fits naturally in a wall-clock budget. Note: the Srinivasan et al. 2023 paper [arxiv:2302.05440](https://arxiv.org/abs/2302.05440) shows PEPITA and FF share "the same learning principles" — implementing one approximates testing the other family.
- **Effort.** Medium.

### 15. FAUST — Forward-Forward with Similarity-Based Tuplet Loss

- **Citation.** Gong, Luo, Wang, Ge, Li, Marattukalam & Abdulla, 2025 preprint. *"Reshaping the Forward-Forward Algorithm with a Similarity-Based Objective."* arxiv:2509.08697. URL fetched: https://arxiv.org/abs/2509.08697
- **Summary.** Replaces the FF goodness/threshold scheme with a **similarity-based tuplet loss** (effectively an N+1-tuple metric loss). Eliminates the need for multiple forward passes at inference time. Closely related to Distance-Forward and HFF.
- **Axis modified.** Goodness function; loss formulation.
- **Reported result.** CIFAR-10 56.22% (MLP), within 1.4 points of BP MLP (57.63%). MNIST and FashionMNIST also reported as competitive.
- **Code.** Not surfaced.
- **Portability (3).** Tuplet loss is well-understood (FaceNet etc.); applied per-layer with one tuple per byte-class it would give us an LM-friendly objective. Note that on CIFAR-10 FAUST gets 56% while CwC (#3) gets 78% — FAUST is *not* the strongest variant in this family; it's listed because the abstract specifically mentions "narrowing the gap to BP" which is the target of this investigation.
- **Effort.** Medium.

### 16. Scalable FF (SFF) — Krutsylo 2025

- **Citation.** Krutsylo, 2025. *"Scalable Forward-Forward Algorithm."* arxiv:2501.03176. URL fetched: https://arxiv.org/abs/2501.03176
- **Summary.** Extends FF to modern convolutional architectures (MobileNetV3, ResNet18). Introduces convolution-specific goodness computation. The "scalable" version is hybrid: BP within residual blocks, FF across them. The strict no-BP variant performs similarly to BP on small datasets.
- **Axis modified.** Backbone (CNN); goodness function (channel-wise variants).
- **Reported result.** "Comparable performance to standard backpropagation" on small datasets and transfer learning.
- **Code.** Not surfaced.
- **Portability (2).** Strict-FF variant is conceptually portable but the paper's value is in conv-specific recipes, and "BP within blocks" violates competition rule 4. Lower portability because of the architectural fit (vision conv, not byte-context FC).
- **Effort.** Medium.

### 17. Adaptive Spatial Goodness Encoding (ASGE)

- **Citation.** Gong, Staszewski & Xu, 2025. *"Adaptive Spatial Goodness Encoding: Advancing and Scaling Forward-Forward Learning Without Backpropagation."* arxiv:2509.12394, ICASSP 2026. URL fetched: https://arxiv.org/abs/2509.12394
- **Summary.** Goodness is computed from **spatial feature-map structure** rather than channel-energy sums, decoupling task complexity from channel dimensionality. Targeted at deep CNNs. First FF method to be evaluated on full ImageNet.
- **Axis modified.** Goodness function (spatial); backbone (deep CNN).
- **Reported result.** MNIST 99.65%, FashionMNIST 93.41%, CIFAR-10 90.62%, CIFAR-100 65.42%, **ImageNet 51.58% top-1 / 75.23% top-5** (first FF on ImageNet at scale).
- **Code.** Not surfaced.
- **Portability (2).** "Spatial goodness" is inherently a 2D-vision concept (spatial dimensions of a feature map). For a byte-context FC backbone there is no spatial dimension. Possible analogue is "context-position goodness" over the K-byte input window, but that's a substantial reinterpretation that the paper doesn't license. Strongest *result* in the FF literature, but lowest *transfer* to char-LM FC.
- **Effort.** Large — would require redesigning the FF rule for non-2D data.

### 18. Covariance-Aware Goodness (BiCovG)

- **Citation.** Jiang, Al-Hashimi & Xu, 2026 preprint. *"Covariance-Aware Goodness for Scalable Forward-Forward Learning."* arxiv:2605.04346. URL fetched: https://arxiv.org/abs/2605.04346
- **Summary.** Augments sum-of-squares goodness with **second-order (covariance) statistics** along two axes: cross-channel covariance (modelling inter-feature dependencies) and multi-scale spatial covariance. Adds a "feature alignment layer" to fix representation drift at block boundaries. Aimed at deeper FF training (VGG-16-class).
- **Axis modified.** Goodness function.
- **Reported result.** ImageNet-100 73.01%, Tiny-ImageNet 50.30%, depth up to 16 layers.
- **Code.** Not surfaced.
- **Portability (2).** The BiCovG idea — augment a scalar goodness with off-diagonal terms — is *transferable in principle* to FC FF on bytes (cross-feature covariance is well-defined), but the paper's spatial component doesn't apply, and the paper hasn't demonstrated the FC-only variant.
- **Effort.** Large.

### 19. FF for Spiking Neural Networks (Ghader et al.)

- **Citation.** Ghader, Kheradpisheh, Farahani & Fazlali, 2025. *"Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm."* arxiv:2502.20411. URL fetched: https://arxiv.org/abs/2502.20411
- **Summary.** Applies FF to spiking neural networks (binary-spike units, temporal dynamics). Two forward passes (positive, negative), goodness is accumulated spike activity. Evaluated on static and temporal spiking datasets.
- **Axis modified.** Backbone (SNN).
- **Reported result.** Competitive with BP-trained SNNs on MNIST, Fashion-MNIST, Kuzushiji-MNIST; outperforms on SHD (Spiking Heidelberg Digits).
- **Code.** Not surfaced.
- **Portability (1).** SNNs introduce a whole separate machine-learning stack (surrogate gradients, temporal dynamics, spike encoding); no obvious transfer to a 300 s A100 budget on dense byte-context FC. Listed for taxonomy completeness only.
- **Effort.** Large.

### 20. Direct Feedback Alignment (DFA) — sibling family

- **Citation.** Launay, Poli, Boniface & Krzakala, 2020. *"Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures."* arxiv:2006.12878. Original DFA: Nøkland 2016, arxiv:1609.01596. URL fetched: https://arxiv.org/abs/2006.12878
- **Summary.** **Sibling family.** Instead of propagating error gradients backward through symmetric weights, project the output error onto each hidden layer through **fixed random feedback matrices**. Layer updates remain locally computable (each layer only needs its own activations and the random-projected error from the output). Critically for this investigation, the 2020 paper **explicitly tests DFA on a Transformer trained on a 100M-token NLP corpus**, demonstrating non-trivial training. This is the only forward-only / local-update sibling with verified language-modelling experience.
- **Axis modified.** Sibling: random feedback. Not strictly an "FF rule" — DFA still uses an error signal, just one delivered by a fixed random projection rather than by chained Jacobians.
- **Reported result.** Successfully trains state-of-the-art models including Transformers; "performance close to fine-tuned backpropagation" but with a noticeably larger gap on Transformers than on convnets. The transformer LM result is the headline negative-control: DFA *can* train an LM, but with a measurable accuracy hit.
- **Code.** Multiple implementations; reference: https://github.com/dbehrlich/directFeedbackAlignment (toy); the Launay et al. paper code linked via the arxiv page.
- **Portability (2).** DFA is competition-rule-borderline: it does compute an output error, then projects it across layers. Whether this counts as "backprop across layers" (rule 4) depends on a reading — the projection matrices are *fixed and random*, not learnable, so no gradient flows through them; only the projected error reaches the inner layers. The transformer-LM precedent is the load-bearing reason to list it. If the rules can be read to allow DFA, the path to char-LM is short (final softmax-CE head + random-projected error to each FF layer = standard DFA recipe).
- **Effort.** Large — DFA is genuinely a different algorithm from pass-2's loss surgery; integrating it cleanly with the existing FF stack requires a re-engineered training loop.

---

## Top picks for Phase 3 testing

Prioritised on three criteria: (a) modifies a cheap axis of pass-2's submission.py (negatives, goodness, or readout), (b) at least suggestive language or sequence precedent, (c) code or close reimplementation available. **Recommended cut: 7 variants.**

1. **Mono-Forward (#2)** — Single most natural port to char-LM. Every layer is its own 256-way next-byte classifier with cross-entropy; replaces both goodness and external readout in one shot. Backed by an energy-efficiency follow-up paper using the same NVML methodology as this competition.

2. **SymBa-FF (#1)** — Smallest possible diff from pass-2 (loss-shaping only). The ICP class-conditional auxiliary is specifically interesting for char-LM where the "class" is rich (256 bytes) — class info can be carried in the activation stream.

3. **DeeperForward (#4)** — Two-line modification (LayerNorm + mean goodness). If passes 1/2 were depth-limited by goodness-explosion, this directly unblocks more layers within the 300 s budget.

4. **Channel-wise Competitive FF / CwC (#3)** — Eliminates negative generation entirely by partitioning hidden units into 256 byte-aligned groups. Removes a whole tuning axis from passes 1/2 (which used external-unigram + hard-self mixtures).

5. **Hyperspherical FF / HFF (#8)** — Prototype-per-byte goodness fits char-LM perfectly. Replaces the goodness-softmax-over-256-candidates eval logic with a single matmul. May tie into the byte embedding table for parameter efficiency.

6. **SCFF (#5)** — Self-contrast removes external negative sampling. The only FF variant with explicit RNN demonstration. For char-LM the contrastive pairs (sequence windows with byte swaps) translate directly.

7. **FF for NLP / Gandhi-Gala (#7)** — The pyramidal-θ schedule is one config change. Worth a parallel run as a near-zero-cost confirmation of whether θ-schedule matters at all on bytes.

**Held back from the top cut, kept as Phase 3+ contingencies:**

- **CaFo (#6)** — Algorithmically similar to Mono-Forward (#2); test Mono-Forward first, then CaFo only if MF wins and the schedule-axis becomes the next thing to optimise.
- **Distance-Forward (#10)** — Centroid metric loss is interesting but the cross-layer "collaboration" term needs careful auditing against rule 4. Defer until baseline Mono-Forward result is in.
- **PEPITA (#14)** and **FTP (#13)** — Sibling families. Useful as escape valves if every FF-proper variant fails to clear ~0.40 — at which point the right move is to broaden out of strict FF, and these are the two cheapest non-FF forward-only options.
- **DFA (#20)** — Only consider if competition rule reading allows fixed-random error projection. Highest historical evidence for *some* LM training, but largest engineering distance and largest rule-interpretation risk.

---

## Notable omissions

- **Going Forward-Forward in Distributed Deep Learning** (Aktemur et al. 2024, [arxiv:2404.08573](https://arxiv.org/abs/2404.08573)) — distributed-systems angle; the underlying FF rule is unchanged. No new algorithm.
- **Characterising Training Behavior** (Adamson 2025, [arxiv:2504.11229](https://arxiv.org/abs/2504.11229)) — mechanistic study, not a variant.
- **On Advancements of the FF Algorithm** (Ortiz Torres et al. 2025, [arxiv:2504.21662](https://arxiv.org/abs/2504.21662)) — surveys conv-channel-grouping + LR schedules + block structures; methods overlap with #16, #17, #18.
- **Improved Forward-Forward Contrastive Learning** (Gananath 2024, [arxiv:2405.03432](https://arxiv.org/abs/2405.03432)) — incremental simplification of Ahmed's FFCL (#12).
- **Beyond Backpropagation: Innovative Algorithms** (Spyra 2025, [arxiv:2509.19063](https://arxiv.org/abs/2509.19063)) and **Energy-Efficient Deep Learning Without Backpropagation** (Spyra & Dzwinel 2025, [arxiv:2511.01061](https://arxiv.org/abs/2511.01061)) — empirical follow-ups validating Mono-Forward (#2) under NVML energy measurement; covered under #2.
- **CFF for Vision Transformer** (Aghagolzadeh & Ezoji 2025, [arxiv:2502.00571](https://arxiv.org/abs/2502.00571)) — included as #11; lower portability.
- **Forward Learning with Top-Down Feedback** (Srinivasan et al. 2023, [arxiv:2302.05440](https://arxiv.org/abs/2302.05440)) — *theoretical* unification of FF + PEPITA; not a new algorithm.
- **Training CNNs with FF in Scientific Reports** (2025, https://www.nature.com/articles/s41598-025-26235-2) — Fourier/morphological label-injection for conv FF; vision-specific spatial labels don't transfer to byte-context FC.
- **Predify** (Choksi et al. 2021, [arxiv:2106.02749](https://arxiv.org/abs/2106.02749)) — predictive-coding feedback augmentation, *requires a pretrained feedforward base*. Pretraining is disallowed by the competition rules.
- **Textual Equilibrium Propagation** (2026, [arxiv:2601.21064](https://arxiv.org/abs/2601.21064)) — operates over LLM-as-module compound AI systems; out of scope for byte-level training a model from scratch.
- **FF for SNNs / FFGAF-SNN** (e.g. [arxiv:2507.23643](https://arxiv.org/abs/2507.23643)) — sub-family of #19; spike-encoding mismatch with the competition setup.
- **Meta predictive learning** (Li, Qiu & Huang 2023, [arxiv:2309.04106](https://arxiv.org/abs/2309.04106)) — mean-field predictive-coding language model; toy-scale, no obvious recipe to scale to 300 s A100 budget.
- **CSE-SFP / single-forward-pass sentence embedding** ([arxiv:2505.00389](https://arxiv.org/html/2505.00389)) — single forward pass for sentence embedding *extraction*, not a training algorithm replacement.
- **LPC-SM** ([arxiv:2604.03263](https://arxiv.org/abs/2604.03263)) — local predictive coding + sparse memory; transformer-architecture work, not a backprop replacement.
- **Forward-Only Continual Learning (FoRo)** ([arxiv:2509.01533](https://arxiv.org/abs/2509.01533)) — prompt-tuning on pretrained models; uses pretrained weights which are competition-disallowed.

---

## Honesty note on portability scores

Almost every entry in this survey was evaluated on vision benchmarks. The portability scores are *hopes*, not measurements. They reflect (i) how cheap the variant is to graft into pass-2's `submission.py`, (ii) whether the modification has a non-vision-specific shape, and (iii) whether anything in the paper hints at sequence/language capability. They do not reflect any empirical measurement of byte-level next-token prediction.

The single non-vision FF result in this survey — Gandhi & Gala on IMDb (#7) — used FF as a classifier over a whole movie review, not as a language model. **No paper in the 2022–2025 corpus reports a character-level (or even token-level) language-model perplexity / accuracy under any FF variant.** That gap is exactly what Phase 3 of this investigation aims to fill.
