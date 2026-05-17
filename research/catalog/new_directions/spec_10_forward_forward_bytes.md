# Research Specification 10: Forward-Forward on Byte-Level WikiText

**Status:** Hypothesis evaluation
**Priority:** Medium-High
**Estimated effort:** 1–2 days

---

## Hypothesis

A Forward-Forward (FF) stack trained per-layer on byte-window "goodness" reaches val char-acc ≥ 0.60 on WikiText-103 within the 300-second harness, demonstrating that strictly layer-local learning (no backward sweep, no activation stash) scales beyond toy alphabets to natural-text byte streams. The energy story is the absence of a backward pass and the absence of any cross-layer optimizer state: each layer's update needs only its input, its output, and its own positive/negative goodness scalar.

The hypothesis is not that FF matches the modded-nanogpt baseline; it is that FF defines a *fundamentally different* energy curve — one whose slope (joules vs. accuracy) is shallower than backprop-based methods at the relevant scale. The first experiment measures that slope.

---

## Background

Hinton's Forward-Forward algorithm (2022) replaces backpropagation with two forward passes:
- A **positive pass** on real data, where each layer is trained to maximize its goodness `G(h) = ||h||² − threshold` (or similar).
- A **negative pass** on contrastive "wrong" data, where each layer is trained to minimize goodness.

Each layer is trained independently: its objective depends only on its own input and output, not on the loss at any other layer. There is no backward sweep through the network, no chain rule across depth, and no need to store activations for a later backward pass.

The original FF demonstration was on classification (MNIST, CIFAR). The only sequence-native FF stub in the cybertronai problems repo is `ff-aesop-sequences`, which reached ~53% accuracy on a 30-symbol alphabet. The open question for this task is whether the FF goodness rule provides enough learning signal on a 256-byte vocabulary to lift the model past 0.60 character accuracy, and whether the candidate-scan inference can fit in the harness budget.

---

## What to build

**Architecture:** L stacked FF layers, each a linear projection + ReLU. Width d ≈ 2048, L = 3–4. Byte vocab 256 → one-hot embedding (or learned 64-dim byte embedding, frozen by FF on the first epoch).

**Input window:** at each position t, the input to the FF stack is the concatenation of one-hot byte embeddings for positions [t−W, t−1] (the W-byte left context, W ≈ 32 or 64). The target is byte t.

**Positive data:** real (context, true-next-byte) pairs from WikiText-103.

**Negative data:** two sources, mixed equally:
1. (context, wrong-byte) where the wrong byte is sampled from the unigram distribution.
2. (context, byte) where the context is from a different document (mismatched context). This catches "byte that fits unigram but wrong for this context."

**Per-layer training:** for each layer l, the input is the output `h_{l-1}` from the previous layer (frozen at training of layer l). The objective:
```
L_l = log(1 + exp(threshold − G(h_l^pos))) + log(1 + exp(G(h_l^neg) − threshold))
```
where `G(h) = mean(h²)`. Optimize per-layer with AdamW; each layer trains independently and can be done in parallel across layers if layer-l input is precomputed.

**Inference:** for each output position, score all 256 candidate next-bytes by running the stack on (context, candidate) and computing cumulative goodness across layers. Argmax over candidates gives the prediction.

**Inference batching:** the 256-way candidate scan is the dominant cost. Batch all candidates together (batch dim = num_positions × 256) so one forward pass evaluates all positions and all candidates. This is the load-bearing engineering detail for fitting in the harness budget.

---

## First experiment (go/no-go gate)

**Goal:** measure FF accuracy and energy on byte-level WikiText, and verify candidate-scan inference fits the budget.

**Procedure:**

1. **Inference throughput pre-check (Day 1, 1 hour):** build the FF inference path with random weights at d=2048, L=3. Run on the first 60,000 val characters with a 256-way candidate scan, batched across 8K positions at a time. Measure wall-clock per prediction.
   - If > 5 ms per output byte, the 60,000-char val eval exceeds 300 s. Abort and reduce d or L until throughput fits.

2. **Training (Day 1–2):** train the FF stack on WikiText-103 train within the 300-second harness:
   - Layer 1: train for the first 60 s.
   - Layer 2: train for the next 60 s on layer-1 outputs.
   - Layer 3: train for the next 60 s on layer-2 outputs.
   - Reserve remaining time for evaluation.

3. **Evaluation:** run the candidate-scan inference on the val set. Record char-acc and joules.

4. **Negative-sample ablation:** retrain with only negative-source-1 (unigram wrong byte). Compare accuracy. The gap shows whether mismatched-context negatives carry signal.

5. **Layer count ablation:** rerun with L=1, L=2, L=4. Plot val char-acc vs. L.

**Measurements to record:**

- Inference wall-clock per output byte (ms)
- Val char-acc and training joules for the full FF stack
- Val char-acc with unigram-only negatives (ablation)
- Val char-acc vs. layer count (L=1..4)
- Per-layer goodness gap (mean G on positives − mean G on negatives) at training end — is it increasing through layers?

---

## Go/no-go criteria

**Go (pursue further):** val char-acc ≥ 0.55 within the 300-second harness, AND per-layer goodness gap is positive and *increasing* through layers (i.e., each layer adds discrimination signal).

The second condition is critical: if the gap plateaus at layer 2, adding more layers won't help and FF is shallow-bottlenecked.

**No-go:** val char-acc below 0.45, OR inference throughput cannot meet the 300-s budget even after architecture reduction. A 0.45 result places FF below the Hebbian-associator floor (Spec 6 in files.zip) and means the per-layer goodness signal is not informative at byte vocabulary scale.

**Borderline (0.45–0.55):** the FF stack is learning but undersized. Two remediations to try (one each, day each):
1. Wider layers (d → 4096), same L.
2. Add a small softmax-output classifier on top of layer-L's goodness vector, trained with backprop. This is a "FF + 1 backprop layer" hybrid that breaks strict layer-locality but tests whether the FF features themselves are useful.

Remediation 2 is informative even if it fails: if FF features + 1 backprop layer is still below 0.55, the features themselves are weak.

---

## What a positive result means

A positive result establishes that strictly layer-local learning works at byte-vocab scale on natural text — a substantive empirical claim. The next experiment is parallelization across layers: since layer l's training depends only on layer l-1's frozen outputs, all L layers can be trained simultaneously on different GPU streams after a pipeline fill, reducing total wall-clock proportionally.

The deeper scientific question after go/no-go is: **does the goodness gap correlate with downstream-task usefulness of the layer representations?** A linear-probe accuracy on the layer-L representations would answer this and connect FF to standard representation-quality benchmarks.

A positive FF result also unlocks the H2 hybrid (Spec to be written): FF for cross-depth locality + delta-rule fast weights for within-layer time-locality. Removes both backprop dependencies simultaneously.

---

## What a negative result means

A negative result on byte-level FF is informative for the broader research program: it suggests that the goodness signal is too coarse to discriminate among 256-way next-byte candidates, and that FF requires structured outputs (smaller vocab, factored representations) to work in practice. This is consistent with Hinton's original 53% result on a 30-symbol alphabet.

The remediation path for a future researcher (out of scope for this spec) would be a factored byte representation: predict the byte as a pair (4-bit high nibble, 4-bit low nibble) with 16-way goodness scans for each. This is a research project, not a 300-s submission.

---

## Resources

- Paper: Hinton, 2022 — "The Forward-Forward Algorithm: Some Preliminary Investigations" — https://arxiv.org/abs/2212.13345
- Repository stub: `cybertronai/hinton-problems`, branch `ff-aesop-sequences`
- Internal investigation: `experiments/FF_INVESTIGATION.md`, `experiments/FF_LITERATURE.md`
- Baseline to beat (energy): modded-nanogpt, 51,704 J, val char-acc 0.7374
- Survey floor for any method: must exceed unigram baseline ~0.16
- Harness: 300-second wall-clock, A100-80GB, NVML joule measurement
