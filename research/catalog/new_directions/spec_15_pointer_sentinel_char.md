# Research Specification 15: Pointer-Sentinel Mixture at the Character Level

**Status:** Hypothesis evaluation (WikiText-native architecture, char-port research)
**Priority:** High (mechanism novelty at low engineering cost)
**Estimated effort:** 2 days

---

## Hypothesis

A small parametric char-LM combined with a **pointer over a recent-character buffer** (sentinel-gated mixture, à la Merity et al. ICLR 2017) reaches val char-acc ≥ 0.70 within 300 s on A100-80GB at training energy **≤ 35 kJ**, beating modded-nanogpt (51.7 kJ) and `lwta_k2` (46.1 kJ) by a wide margin.

The energy bet: WikiText is **extremely repetitive at the char level** — proper nouns, citation formatting, wiki-markup tokens, and common English n-grams recur within thousands of characters. A pointer that copies from a position in the recent buffer costs ~zero additional parameters and zero matmul on a hit. The parametric backbone can therefore be **substantially smaller** than a standalone transformer while still hitting the floor, because the pointer absorbs the long-tail predictable content.

---

## Background

Pointer-Sentinel Mixture LM (Merity, Xiong, Bradbury, Socher, ICLR 2017, [arxiv:1609.07843](https://arxiv.org/abs/1609.07843)) was designed *for WikiText* at the word level. Output probability is a mixture:

```
P(next_token) = g · P_vocab(next_token) + (1 − g) · P_pointer(next_token)
```

where `P_pointer` is computed by attention-style scoring over a buffer of recent tokens — softmax over similarity between a "pointer query" and each buffered token's key, then a sum of probability mass at every buffer position whose token equals the candidate next token. The **sentinel** is an extra learnable vector concatenated to the keys; if the sentinel wins attention, the gate `g` shifts toward the parametric vocab distribution.

**Char-level adaptation (this spec's contribution):**

- Buffer: last `W` characters (e.g., W=1024). Chars are simple one-hot or learned embedding.
- Pointer query: linear projection of the parametric model's final hidden state.
- Pointer keys: a linear projection of buffered char embeddings, plus a relative-position embedding (char-level position matters more than word-level — character bigrams are tight).
- Sentinel: one learned key vector appended.
- For candidate char `c`, `P_pointer(c) = sum_{i: buffer[i] == c} attention_weight[i]`.

**Why char level is favorable:** at the word level, pointer hits are sparse (vocab is ~250K). At the char level, vocab is 256, so pointer hits are dense — most candidate chars appear at multiple positions in any 1024-char buffer. The pointer distribution therefore covers most of the support, while the parametric branch handles rare-character cases (where the pointer would be ≈ uniform).

**Mechanism novelty for the program:** no published char-level pointer-sentinel exists. Char-pointer is mechanically novel and aligns with the project's "capability demo, not leaderboard" framing — even a borderline accuracy result demonstrates the mechanism scales to char streams.

---

## What to build

**Parametric backbone.** Small causal transformer: d_model=256, layers=4, heads=4, seq_len=1024. Approximate parameter count: ~4M. **This is intentionally tiny** — the pointer carries the long tail.

**Pointer module:**

```python
class CharPointerSentinel(nn.Module):
    def __init__(self, d_model, vocab=256, buffer_len=1024):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)          # pointer query from hidden
        self.W_k = nn.Linear(d_model, d_model)          # key from buffered embeddings
        self.sentinel = nn.Parameter(torch.randn(d_model) * 0.02)
        self.W_vocab = nn.Linear(d_model, vocab)         # parametric vocab branch
        self.W_gate = nn.Linear(d_model, 1)              # log-odds for mixture gate
        self.buffer_len = buffer_len

    def forward(self, hidden, buffered_embeds, buffered_chars):
        # hidden: (B, T, D)  — final transformer hidden states
        # buffered_embeds: (B, T, W, D) — embeddings of last W chars at each position
        # buffered_chars: (B, T, W) — int char ids at each buffer position
        q = self.W_q(hidden)                                              # (B, T, D)
        k = self.W_k(buffered_embeds)                                     # (B, T, W, D)
        sent = self.sentinel.expand(*buffered_embeds.shape[:-2], 1, -1)   # (B, T, 1, D)
        keys = torch.cat([k, sent], dim=-2)                               # (B, T, W+1, D)
        logits = (q.unsqueeze(-2) * keys).sum(-1) / math.sqrt(q.shape[-1]) # (B, T, W+1)
        attn = logits.softmax(-1)                                         # (B, T, W+1)
        sentinel_mass = attn[..., -1]                                     # (B, T)
        pointer_attn = attn[..., :-1]                                     # (B, T, W)

        # Scatter pointer mass into vocab slots
        p_pointer = torch.zeros(*hidden.shape[:2], 256, device=hidden.device)
        p_pointer.scatter_add_(-1, buffered_chars, pointer_attn)          # (B, T, 256)

        # Vocab branch
        p_vocab = self.W_vocab(hidden).softmax(-1)                        # (B, T, 256)

        # Mixture gate: sentinel mass is the natural gate value
        g = sentinel_mass.unsqueeze(-1)                                   # (B, T, 1)
        return g * p_vocab + (1.0 - g) * p_pointer
```

**Training.** Cross-entropy on the mixture. The gate is learned implicitly: when the pointer is correct, gradient pushes the sentinel score down; when the parametric branch is correct, gradient pushes sentinel score up. No explicit gate supervision.

**Streaming (CharModel.predict).** Buffer maintenance is the engineering crux. Approach: maintain a rolling deque of length W of recent (char_id, embed) pairs. On `observe(c)`, push. On `predict()`, run the transformer over the *contextual position* (cached KV), then the pointer over the buffered W chars.

**Sizing target.** Backbone ~4M params; pointer module adds ~1.5M (W_q + W_k + sentinel + W_vocab + W_gate). Total ≈ 5–6M, **vs. modded-nanogpt's ~36M**. The bet: pointer compensates for the parameter gap.

---

## First experiment (go/no-go gate)

**Goal:** confirm char-pointer reaches 0.70 with a small backbone, at substantially lower energy than 46 kJ.

**Procedure:**

1. Implement `submissions/pointer_sentinel/submission.py`.

2. Submit. Record metrics including a **pointer-usage diagnostic**: mean `sentinel_mass` across the eval — what fraction of the time does the model trust the pointer vs. the vocab branch?

3. If val char-acc < 0.70:
   - **Remediation A:** widen buffer W = 1024 → 2048 (more chances of a pointer hit).
   - **Remediation B:** increase backbone to d=384, layers=6 (still well below modded-nanogpt size).
   - One remediation only.

**Measurements to record:**

- Val char-acc, training joules, training duration
- Backbone param count
- Mean `sentinel_mass` on val (pointer-usage rate; expect 0.4–0.7)
- Pointer-hit accuracy: when pointer wins, is it correct? Compute on a 1K-char val chunk.
- Vocab-branch accuracy alone: ablate the pointer at inference and report char-acc.

---

## Go/no-go criteria

**Go:** val char-acc ≥ 0.70 AND training joules ≤ 40 kJ. Strong leaderboard candidate.

**Soft-pass (mechanism interesting, leaderboard borderline):** val char-acc ≥ 0.70 AND joules in (40 kJ, 50 kJ]. Reports the energy-versus-LWTA gap; documents the pointer-usage rate as a finding regardless of leaderboard rank. **Even at this level, the pointer-usage rate is a publishable observation.**

**No-go:** val char-acc < 0.70 after one remediation. Two diagnostic paths:
- If pointer-hit accuracy is high (>0.8) but sentinel_mass is low (<0.2), the gate is broken — pointer works mechanically but model doesn't trust it. Add a gate-regularization term and rerun. (Out of scope for this Phase 1 spec; document and move on.)
- If pointer-hit accuracy is low, the keys/queries aren't aligning. The mechanism doesn't work at char level. Discard.

---

## Phase 2 (conditional on Go)

1. **Pointer over a long buffer (W=4096).** WikiText repetition spans paragraphs; longer buffer should help. Memory-bounded by the W² attention inside the pointer module.
2. **Hierarchical pointer:** pointer over chars within recent buffer + pointer over phrase-anchors detected by a small RNN over the buffer. Two-level mixture.
3. **Pointer + LWTA backbone composition.** Already-small backbone gets LWTA in its MLP.

---

## What a positive result means

A char-level pointer-sentinel that beats modded-nanogpt on joules at substantially fewer parameters is the first evidence that **explicit copy mechanisms** are competitive on byte char-LM. It opens the door to copy-mechanism heavy architectures (RETRO-style retrieval, kNN-LM at char level) under the same harness.

The deeper finding (regardless of leaderboard rank): how much of WikiText char-prediction is **copying from recent history** vs. **generating from a learned distribution**? The mean `sentinel_mass` is a direct numerical answer to that question. No prior char-LM has reported it.

---

## What a negative result means

A negative result means **char-level pointer attention does not align** under the small-backbone-large-buffer trade we tried. Two interpretations:

1. The parametric backbone is too small to produce useful query vectors. (Refuted if pointer-hit accuracy is high but gating is bad — the backbone works enough to query well; the mixture isn't learning the right gate.)
2. The pointer's "match a candidate char by attention-weighted sum" formulation is too coarse — at char level, *which* recent occurrence to copy from is highly context-dependent in ways that simple key-query similarity misses. RETRO-style chunked attention may be the cleaner test of the copy hypothesis.

---

## Resources

- Paper: Merity, Xiong, Bradbury, Socher — "Pointer Sentinel Mixture Models" — [arxiv:1609.07843](https://arxiv.org/abs/1609.07843)
- Original AWD-LSTM successor: https://github.com/salesforce/awd-lstm-lm (word-level; not directly usable but useful for the pointer-keys design)
- Baseline to modify: `submissions/modded_nanogpt/` (the backbone shrinks; everything else stays)
- Current leader: `submissions/lwta_k2/` at 46.1 kJ / 0.7146
- Harness: 300 s, A100-80GB, NVML joules, val char-acc ≥ 0.70 on 60K val chars
