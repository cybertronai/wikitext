# Research Specification 13: Hyena Hierarchy as Attention Replacement for Byte-Level LM

**Status:** Hypothesis evaluation (highest-priority Manning-branch port)
**Priority:** High
**Estimated effort:** 1–2 days

---

## Hypothesis

Replacing softmax attention with a Hyena operator (implicit long convolution + element-wise multiplicative gating) in an otherwise modded-nanogpt-style stack reaches val char-acc ≥ 0.70 within 300 s on A100-80GB at training energy **≤ 40 kJ** — strictly below both modded-nanogpt (51.7 kJ) and `lwta_k2` (46.1 kJ), the current leaderboard.

The bet is structural: Hyena is sub-quadratic in sequence length (O(N log N) via FFT-based long convolutions) versus attention's O(N²). At byte level with sequence lengths of 1024–4096 inside the 300 s budget, the FFT path should win on FLOPs and therefore on joules, while implicit-conv parameterization (Poli et al. report ~20% training-compute reduction matched-quality on word-level WikiText-103) preserves expressive capacity.

---

## Background

Hyena (Poli, Massaroli, Nguyen, Fu, Dao, Baccus, Bengio, Ermon, Ré, ICML 2023, [arxiv:2302.10866](https://arxiv.org/abs/2302.10866)) defines a recurrence:

```
y = x0 * (h_1 * x1) * (h_2 * x2) * ...
```

where `xi` are linear projections of the input and `hi` are **implicit long convolution filters** parameterized by a small MLP `γ : position → filter_value`. The convolution `h_i * x_i` is computed with FFT in O(N log N). The multiplicative gating gives data-dependent mixing in the spirit of attention without quadratic similarity.

**Why this matters for joules on bytes:** at seq=2048, attention costs ~2.1B element products; an FFT-based long conv costs ~22M (96× ratio in floating ops). Even if A100 tensor-core efficiency closes most of that gap, the published claim is ~20% training-compute reduction on WikiText-103 word level. Byte-level changes the constants but not the asymptotic story.

**Reference implementations**:
- HazyResearch/safari (canonical, ~250 LOC for the operator)
- HazyResearch/flash-fft-conv (CUDA-fused; nice-to-have, not required)
- A direct PyTorch port using `torch.fft` is enough for a Phase-1 submission.

---

## What to build

Modify the modded-nanogpt scaffold (`submissions/modded_nanogpt/submission.py`) by **replacing the CausalSelfAttention block with a Hyena operator**. Keep everything else (byte embedding, RMSNorm, ReLU² MLP, Muon optimizer, learning-rate schedule, vocab=256).

**Hyena operator (single layer):**

```python
class HyenaOperator(nn.Module):
    def __init__(self, d_model, order=2, seq_len=2048, filter_hidden=64):
        super().__init__()
        self.in_proj = nn.Linear(d_model, (order + 1) * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # Implicit filter MLP: maps positional embedding -> filter values
        self.filter_mlp = nn.Sequential(
            nn.Linear(1, filter_hidden), nn.SiLU(),
            nn.Linear(filter_hidden, filter_hidden), nn.SiLU(),
            nn.Linear(filter_hidden, order * d_model),
        )
        self.register_buffer("pos", torch.arange(seq_len).float().unsqueeze(-1) / seq_len)
        self.order = order
        self.d_model = d_model

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        projs = self.in_proj(x).chunk(self.order + 1, dim=-1)  # x0, x1, ..., x_order
        # Get filters h_1, ..., h_order via implicit MLP at all positions
        filters = self.filter_mlp(self.pos[:T])  # (T, order * D)
        filters = filters.view(T, self.order, D).permute(1, 2, 0)  # (order, D, T)
        # Causal FFT-conv of x_i with h_i
        v = projs[0]  # (B, T, D)
        for i in range(self.order):
            v = v * projs[i + 1]
            v = causal_fft_conv(v, filters[i])  # (B, T, D)
        return self.out_proj(v)
```

`causal_fft_conv` zero-pads to 2T, does `rfft → multiply → irfft`, slices first T; standard recipe in safari's implementation.

**Decoding mode (CharModel.predict):** at inference, Hyena can be unrolled to an O(N · state) recurrence (state-space form) or simply re-run as a full forward pass on the buffered context. **For this spec, use the simple re-run path** — the streaming eval calls `predict()` once per char on the current buffer; total eval cost is O(eval_chars × buffer_len × log buffer_len). With buffer_len capped at 2048, this is tractable in the un-budgeted eval phase.

**Sizing target.** Pick a configuration with ≈ same parameter count as modded-nanogpt baseline (so the energy delta is mechanism, not capacity):
- d_model = 512, layers = 8, hyena_order = 2, seq_len = 2048
- Total params ≈ baseline within 10%

**Training.** Use modded-nanogpt's existing recipe verbatim: Muon for 2-D weights, AdamW for embeddings and 1-D scalars, stable-then-decay LR schedule, RMSNorm, ReLU² MLP. The only change is the attention block.

---

## First experiment (go/no-go gate)

**Goal:** confirm Hyena at byte level reaches the 0.70 floor and measure the joule delta.

**Procedure:**

1. Implement `submissions/hyena/submission.py`. Mirror `submissions/modded_nanogpt/submission.py` structure; swap CausalSelfAttention for HyenaOperator.

2. Test locally with a 30 s smoke-test before Modal dispatch (CPU is fine for shape-checking; skip if no time budget).

3. Submit via `python submit.py submissions/hyena/`.

4. Record val char-acc, training joules, training duration, GPU memory peak.

5. If val char-acc < 0.70, try **one** remediation: bump d_model to 640 or layers to 10 (whichever keeps training under 280 s). One retry — no extended hyperparam search at Phase 1.

**Measurements to record:**

- Val char-acc and energy (J)
- Training duration (s) and per-step time
- Parameter count vs. modded-nanogpt baseline
- Activation-memory peak (Hyena should be lower than attention at the same seq length)
- FFT vs. matmul wall-clock breakdown (one-time `torch.profiler` snapshot)

---

## Go/no-go criteria

**Go:** val char-acc ≥ 0.70 AND training joules ≤ 46 kJ (beats lwta_k2). This makes Hyena the new leaderboard candidate.

**Borderline:** val char-acc ≥ 0.70 but joules in (46 kJ, 51.7 kJ]. Hyena beats baseline but not LWTA; report and move on. No kernel-fusion work in this case.

**No-go:** val char-acc < 0.70 even after the one allowed remediation. Hyena is structurally weaker than attention at byte level under this budget; the published word-level result does not transfer. Report and discard the byte-level Hyena direction.

---

## Phase 2 (conditional on Go)

If Phase 1 wins, two follow-ups:

1. **Hyena + LWTA composition.** The two changes are orthogonal (sequence-mixing vs. MLP activation). Stack them; expect additive joule savings.
2. **flash-fft-conv kernel.** Replace `torch.fft.rfft/irfft` with HazyResearch/flash-fft-conv if its dependencies install cleanly in the Modal image. Realized speedup: 2–3× on the FFT step.

Both follow-ups are roughly 1 day each.

---

## What a positive result means

A Hyena win is the first evidence that **sub-quadratic sequence mixers** beat attention on joules at byte level under our budget. It opens the door to: H3 (hybrid 2-attention + Hyena), Mamba (concurrent spec 14), and longer-context experiments where the attention-N² wall hurts more.

The deeper question after Phase 1: **how does Hyena scale with seq_len at fixed budget?** Attention's wall steepens with N; Hyena's stays at N log N. If Hyena wins at 2048 it should win further at 4096 — a separate seq-len sweep is the natural follow-up.

---

## What a negative result means

A negative result means **the implicit-filter parameterization is the bottleneck at byte level**. Two interpretations are possible:

1. *Capacity:* the filter MLP cannot represent the data-dependent mixing structure that softmax attention learns at byte level. Remediation: larger filter MLP, learnable positional features. Out of scope for this spec.
2. *Inductive bias:* bytes have very local dependencies (trigrams near-deterministic) and FFT-based long-conv is a poor fit for the local structure. In that case Mamba (selective SSM, which has stronger local-recency bias) is the better Manning-branch bet.

Either way, the negative result feeds directly into the Mamba spec's interpretation.

---

## Resources

- Paper: Poli, Massaroli, Nguyen, Fu, Dao, Baccus, Bengio, Ermon, Ré — "Hyena Hierarchy: Towards Larger Convolutional Language Models" — [arxiv:2302.10866](https://arxiv.org/abs/2302.10866)
- Reference impl: https://github.com/HazyResearch/safari
- Fused FFT kernel: https://github.com/HazyResearch/flash-fft-conv
- Baseline to modify: `submissions/modded_nanogpt/`
- Current leader: `submissions/lwta_k2/` at 46.1 kJ / 0.7146
- Harness: 300 s, A100-80GB, NVML joules, val char-acc ≥ 0.70 on 60K val chars
