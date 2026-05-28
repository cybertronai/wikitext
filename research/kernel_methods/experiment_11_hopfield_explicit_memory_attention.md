# Experiment 11: Modern Hopfield Layer as Explicit-Memory Augmented Attention

## Hypothesis
Inserting a single modern-Hopfield retrieval layer (Ramsauer 2020) — which is mathematically softmax-attention over a *fixed external memory* of stored patterns — into a smaller-than-modded-nanogpt transformer recovers the missing capacity at lower training energy than scaling up the dense transformer. Tests whether kernel-density retrieval over a frozen memory bank can substitute for a few transformer layers.

## Motivation
Ramsauer 2020 established the identity: modern Hopfield network update rule = transformer softmax attention. Practically, a "Hopfield layer" can be used as a retrieval mechanism over a *frozen* set of patterns drawn from training data. This is a **paradigm-B kernel-as-component** (softmax kernel on (query, memory) pairs) and a clean realization of the explicit-memory paradigm without the kNN-LM machinery of exp 08.

Why it might pay off: the memory bank can be much larger than what's stored in transformer parameters (200K patterns × 384 dim = 300 MB vs. ~6 MB of one transformer layer), and computing softmax-attention against it is O(N·M) with M = memory size — comparable to FAVOR+ feature-count cost. If a 4L transformer + Hopfield layer matches a 6L transformer, that's an energy win.

Cross-ref: `survey_kernel_methods_2026_05.md` family-4 (modern Hopfield = kernel density estimator on attention manifold).

## Method
Architecture: 4-layer transformer (modded-nanogpt body, *smaller* than baseline) + one Hopfield retrieval layer between layers 2 and 3:

```
HopfieldLayer:
    K_mem, V_mem ∈ R^(M, d)   # M=4096, frozen, sampled from train
    forward(q):
        attn = softmax(q K_memᵀ / √d) V_mem
        return attn + q  # residual
```

The memory K_mem, V_mem is constructed *once* at the start of training:
1. Sample M = 4096 (context, next-byte) pairs from train
2. Forward-pass through a small temporary network (random init or 1-step trained) to get K_mem; V_mem = embedded next-byte
3. Freeze K_mem, V_mem for the rest of training

Train the rest of the model (4 transformer layers + Hopfield layer + output head) with standard SGD/Muon.

## Memory-Movement Analysis
- Memory bank: M × d × 2 bytes (bf16) = 4096 × 384 × 2 = 3 MB. Fits in L2 cache of A100.
- Per training step: Hopfield layer cost = B × T × M × d = 32 × 1024 × 4096 × 384 = 50 GFLOPs (similar to one attention layer at T=4096; here softmax is over M not T).
- Equivalent to having a 5th transformer layer's worth of compute but with a *frozen* effective key/value matrix.
- Energy: 4-layer body saves ~33% of attention/MLP cost (4/6 layers). The Hopfield layer adds back ~16% (one more attention-ish layer). Net ~83% baseline compute → projected ~43 kJ if accuracy holds.

## Setup
- Dataset: `/data/wiki.train.raw`, `/data/wiki.valid.raw`
- Tokenization: byte (256)
- Model: 4-layer transformer (modded_nanogpt config but L=4 instead of 6) + 1 Hopfield retrieval layer between layers 2-3 + output head
- Memory bank: M = 4096 patterns from train, frozen
- Optimizer: AdamW + Muon (modded_nanogpt setup)
- Hardware: 1 × A100-80GB, 300 s
- Baseline: modded_nanogpt 51.7 kJ / 0.7374; 4-layer modded_nanogpt as ablation
- Metric: val char-acc, NVML energy

## Procedure
1. Copy `submissions/modded_nanogpt/submission.py` → `submissions/hopfield_layer/submission.py`
2. Set `num_layers=4`.
3. Define `HopfieldLayer` module:
   ```python
   class HopfieldLayer(nn.Module):
       def __init__(self, d, M):
           super().__init__()
           self.register_buffer("K_mem", torch.zeros(M, d))
           self.register_buffer("V_mem", torch.zeros(M, d))
       def forward(self, q):  # q: (B, T, d)
           attn_scores = q @ self.K_mem.T / d**0.5  # (B, T, M)
           attn = F.softmax(attn_scores, dim=-1)
           return attn @ self.V_mem + q
   ```
4. Memory initialization (called inside train() before main loop):
   - Sample M random context positions in train.
   - Use the current (random-initialized) embed + first 2 layers to encode → K_mem.
   - Encode the next-byte at each position via same embed → V_mem.
   - Detach and store as buffers.
5. Insert Hopfield layer between layers 2 and 3 in the forward pass.
6. Train normally with cross-entropy + AdamW + Muon.
7. Submit.

## Success Criteria
- **Strong win:** val ≥ 0.70 AND energy ≤ 45 kJ → Hopfield retrieval substitutes for transformer depth
- **Pass:** val ≥ 0.70 AND energy ≤ 51 kJ → matches baseline with different mechanism
- **A/B win:** val(this) > val(4-layer modded_nanogpt without Hopfield) → memory layer added value
- **Refuted:** val < 0.70 → memory not enough to compensate for fewer layers

## Failure Modes & Diagnostics
- **Random-init K_mem useless:** the K_mem patterns are only as useful as the encoder that built them. Try: rebuild K_mem *after* 200 warm-up steps (use the current model's encoder).
- **Memory grows stale:** in principle the body learns to use K_mem; but if K_mem is from random-init encoder, body has to first reverse-engineer that encoding. Adding a learnable linear over K_mem (`K_mem @ W_proj`) may help.
- **Attention over M=4096 underflows in bf16 softmax:** check that softmax temperatures are sane; consider scaling 1/√d → 1/√(2d) for safety.
- **Naive Hopfield with random K is worse than no extra layer:** if so, this provides a clean negative result on "frozen memory" vs "learned memory."

## Estimated Cost
- 1 Modal A100 run, ~8 min wall, expected energy 35-55 kJ
- ~$0.30

## References
- Ramsauer et al. 2020 "Hopfield Networks Is All You Need" (arXiv 2008.02217, ICLR 2021)
- Krotov & Hopfield 2016 "Dense Associative Memory for Pattern Recognition" — dense Hopfield generalization
- Wu et al. 2022 "Memorizing Transformers" (ICLR) — close practical analog
