# Experiment 11 Result — Hopfield Explicit-Memory Attention

## Hypothesis (recap)
A 4-layer modded-nanogpt body + one frozen modern-Hopfield retrieval layer (M=4096 patterns sampled from train, encoded with the random-init encoder) matches or beats the 6-layer modded-nanogpt baseline at lower training energy. Tests kernel-density retrieval over a frozen memory bank as a substitute for transformer depth.

## Final numbers
| metric | value |
|---|---|
| training_energy_J | **40,158.1 J** |
| training_duration_s | 184.5 s |
| val_char_accuracy | **0.7292** |
| val_chars | 60,000 |
| DQ status | **PASS** (acc >= 0.70, duration < 300 s) |
| GPU | A100-80GB PCIe |

## Comparison to modded_nanogpt baseline (51,704 J / 0.7374 acc)
- Energy delta: **-11,546 J  (-22.3%)**
- Accuracy delta: -0.0082 (-0.82 pp)
- Duration delta: -184.5 s vs. ~248 s baseline (also faster — fewer layers + lower step count not changed, just less per-step compute)

## Success Criterion bracket
- Strong win: val >= 0.70 AND energy <= 45 kJ.
  - val = 0.7292 (>= 0.70) AND energy = 40.2 kJ (<= 45 kJ) -> **Strong win.**
- The frozen Hopfield retrieval layer recovered enough of the missing capacity that 4 layers + Hopfield delivers ~99% of the 6-layer baseline accuracy at 78% of its energy.

## Interpretation
The frozen Hopfield retrieval layer substantially substituted for the two transformer layers removed: 6L baseline -> 4L body shaved compute by ~33%, the Hopfield layer added back a small fraction (one matmul against a 4096-key softmax), and the model still hit 0.729 char-acc — only 0.008 below the 6L baseline. The "random-init K_mem" failure mode the spec warned about did not bite: although K_mem was constructed by an untrained 2-layer encoder, the training-loop encoder co-adapts to that fixed key geometry, and the head/MLP/post-Hopfield blocks learn to consume the retrieved V_mem patterns. Loss descended monotonically from ~5.5 (step 0 region) to 0.99, very similar trajectory to baseline modded-nanogpt. No warm-up rebuild of K_mem was needed.

This is consistent with the Hopfield-as-attention identity: from the body's perspective, the layer is just attention with a fixed-but-content-rich K/V table, and the embed -> 2-block encoder used to build K_mem is identity-of-encoder with what produces queries in the live forward pass, so query/key alignment is automatic (and self-improving as those 2 blocks train).

## Implementation deviations from the spec
- **Softmax scaling**: used `1/sqrt(2d)` (spec's safety variant) and computed the softmax in fp32 to avoid bf16 underflow over M=4096 logits. Spec lists both as acceptable safeguards.
- **Memory-bank construction context**: sampled 256-byte windows (not full 1024) for K_mem encoding to bound activation memory during the one-shot init; only the last-position hidden state is stored as the key, which is what attention queries it against. This deviates from any specific window-length the spec didn't pin down.
- **No K_mem warm-up rebuild used**: the "simple" path (spec default) was sufficient — the simple version is the headline result.
- Insertion point: after block index 1 (0-indexed), i.e. between the 2nd and 3rd blocks — matches spec "between layers 2 and 3" (1-indexed).
- All other hyperparameters (n_steps=2150, batch=32, T=1024, optimizer LRs, init scheme) identical to modded_nanogpt.

## Artifacts
- Submission: `/home/seneca/wikitext/submissions/hopfield_layer/submission.py`
- Result JSON: `/home/seneca/wikitext/submissions/hopfield_layer/result.json`
- Run log: `/home/seneca/wikitext/submissions/hopfield_layer/run.log`
- Modal run: https://modal.com/apps/ab-10/main/ap-BqA5XxJvGuP9w0ZuWKSTAO

## Review (post-hoc audit)

**Validity for the explicit-memory-Hopfield-layer claim:** *Pass on the leaderboard metric; weak isolation of the mechanism.*

**Core limitations:**
- **No ablation against "same architecture, Hopfield removed".** The result demonstrates *that* a 4-layer transformer + a frozen-random-KV softmax-retrieval layer + Muon reaches 0.729 at 40.2 kJ; it does not demonstrate that the Hopfield layer is *load-bearing*. A control with `hopfield_M = 0` or with the Hopfield layer replaced by identity would distinguish "Hopfield contributed" from "the 4-layer Muon transformer would have made the floor anyway".
- **The "frozen K_mem, V_mem from current encoder state at init" memory bank is effectively additional fixed-kv attention** against a random subset of training contexts. The mechanism is modern-Hopfield in name; in compute graph it is one extra attention head with non-trained K/V. The novelty claim should be scoped accordingly.
- **22 % energy reduction vs. baseline is real** but the baseline is 6-layer; the comparison conflates "Hopfield helped" with "removing two transformer blocks helped under this energy budget".

**Verdict:** Valid as a leaderboard data point. The next round should be a clean ablation (M ∈ {0, 1024, 4096}; Hopfield layer replaced by additional self-attention head; warm-start vs. random K_mem) before attributing the 0.729 to the Hopfield mechanism.
