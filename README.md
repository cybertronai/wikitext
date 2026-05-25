# wikitext

Train a character-level language model from scratch on WikiText-103.
Submissions are scored by lowest energy to reach 0.7 character prediction accuracy. 
Scorer is agnostic to backprop.
Submissions of novel learning algorithms are encouraged!

## Quickstart

**Prerequisites:**
1. [Modal](https://modal.com) account.
2. Python 3.11

```bash
# From wip-wikitext/

python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

modal token new

python submit.py submissions/modded_nanogpt
```

## Record History

The `Energy (J)` column reports **`total_energy_J`** (GPU NVML net of idle baseline + CodeCarbon CPU estimate, floored at `duration_s × 50 W`) for rows dated **2026-05-20 and later**. Earlier rows report the prior NVML-only `training_energy_J`. The semantic change is the new total-system-energy rule per @yaroslavvb2's Telegram note; see `MAINTAINING.md` and the `EnergyMeter` source for details. Upstream-leaderboard rows from before the change have been re-run under the new harness — those re-runs appear below as the canonical entries for those submissions; the original rows are preserved for history.

| Date | Energy (J) | Val char-acc | GPU | Config | Submission | Contributor |
|------|-----------:|-------------:|-----|--------|------------|-------------|
| 2026-05-12 |     51,704 | 0.7374    | A100 80GB PCIe | modded_nanogpt | [dir](submissions/modded_nanogpt) | @KellerJordan |
| 2026-05-18 |     46,222 | 0.7238    | A100 80GB PCIe | lwta_k4        | [dir](submissions/lwta_k4)        | @ab-10 |
| 2026-05-18 |     46,132 | 0.7146    | A100 80GB PCIe | lwta_k2        | [dir](submissions/lwta_k2)        | @ab-10 |
| 2026-05-18 |     55,459 |       DQ | A100 80GB SXM4 | hyena            | [dir](research/catalog/new_directions/hyena)            | @ab-10 |
| 2026-05-18 |     20,348 |       DQ | A100 80GB SXM4 | pointer_sentinel | [dir](research/catalog/new_directions/pointer_sentinel) | @ab-10 |
| 2026-05-18 |      3,612 |       DQ | A100 80GB PCIe | chunker_d1       | [dir](research/catalog/new_directions/chunker_d1)       | @ab-10 |
| 2026-05-18 |        735 |       DQ | A100 80GB PCIe | ppm_c            | [dir](research/catalog/new_directions/ppm_c)            | @ab-10 |
| 2026-05-17 |         70 |       DQ | A100 80GB SXM4 | P2-A_random_projection | [dir](research/forward-forward-deep/runs/phase2/P2-A_random_projection) | @ab-10 |
| 2026-05-19 |     60,864 |       DQ | A100 80GB PCIe | mamba_byte           | [dir](submissions/mamba_byte)           | @claude-mamba |
| 2026-05-20 |      1,752 |       DQ | A100 80GB SXM4 | gpu_ngram_w31_k10    | [dir](submissions/gpu_ngram_w31_k10)    | @follow-up-paq-prediction |
| 2026-05-20 |     13,936 |       DQ | A100 80GB SXM4 | chunker_phase1_v2    | [dir](submissions/chunker_phase1_v2)    | @explore-chunker-2026-05-19 |
| 2026-05-20 |     24,417 |       DQ | A100 80GB SXM4 | bpe_internal_nn_v2   | [dir](submissions/bpe_internal_nn_v2)   | @subagent-xorfix-2026-05-19 |
| 2026-05-20 |     53,683 | 0.7246    | A100 80GB PCIe | lwta_k4              | [dir](submissions/lwta_k4)              | @ab-10 (re-run on new harness; total_J = 44,329 gpu + 9,354 cpu) |
| 2026-05-20 |     54,614 | 0.7145    | A100 80GB PCIe | lwta_k2              | [dir](submissions/lwta_k2)              | @ab-10 (re-run on new harness; total_J = 44,583 gpu + 10,031 cpu) |
| 2026-05-21 |      2,474 | 0.7031    | A100 80GB PCIe | subset_70_mkn        | [dir](submissions/subset_70_mkn)        | @exp-batch-iter4 |
| 2026-05-21 |      3,092 | 0.7050    | A100 80GB PCIe | gpu_ngram_w31_k11    | [dir](submissions/gpu_ngram_w31_k11)    | @follow-up-paq-prediction |
| 2026-05-21 |      4,607 | 0.7047    | A100 80GB PCIe | paq_mixer_v3         | [dir](submissions/paq_mixer_v3)         | @worker-paq-mixer |
| 2026-05-21 |      8,602 | 0.7184    | A100 80GB PCIe | gpu_ngram_o14_xorfix | [dir](submissions/gpu_ngram_o14_xorfix) | @subagent-xorfix-2026-05-19 |
| 2026-05-21 |      9,591 | 0.7063    | A100 80GB PCIe | chunker_phase1_v1    | [dir](submissions/chunker_phase1_v1)    | @explore-chunker-2026-05-19 |
| 2026-05-21 |     14,578 | 0.7184    | A100 80GB PCIe | deep_backoff_kn      | [dir](submissions/deep_backoff_kn)      | @nakajimagabriel |
| 2026-05-21 |     19,922 | 0.7328    | A100 80GB SXM4 | lwta_k4_alpha_065    | [dir](submissions/lwta_k4_alpha_065)    | @subagent-L2clean-2026-05-19 |
| 2026-05-21 |     20,743 | 0.7390    | A100 80GB SXM4 | alpha_06             | [dir](submissions/alpha_06)             | @subagent-xorfix-2026-05-19 |
| 2026-05-21 |     62,006 | 0.7337    | A100 80GB SXM4 | modded_nanogpt       | [dir](submissions/modded_nanogpt)       | @ab-10 |


## Rules

Train a character-level language model from scratch on [WikiText-103-raw-v1](https://huggingface.co/datasets/Salesforce/wikitext).
Submissions that meet the constraints below are ranked by **training energy (joules)**, lower wins.
Greedy-argmax char-accuracy is computed on the first 60,000 chars of each split; val is gated by rule 5, test is reported alongside but not gated.

**Submissions must:**

1. Train from scratch. (No pre-trained weights — WikiText overlaps WebText, so pre-trained init poisons the comparison.)
2. Use the standard WikiText-103 train/valid/test split. (You can change batch size, sequence length, attention structure, etc.; just don't change the underlying streams of characters.)
3. Expose a streaming next-character distribution via the `CharModel` API. (The runner calls `predict()` for position `i` strictly before `observe()` commits the ground-truth at position `i` — within-document future-peeking is structurally impossible.)
    a. Implementing `CharModel` ABC from `wikitext.py` is the most straightforward way to do this.
4. Finish training in **< 300 s wall-clock** on the pinned Modal A100-80GB PCIe, measured from the first call into `train()` to its return. (Eval is not charged against this budget.)
5. Attain **val char-acc ≥ 0.70** on the first 60,000 chars of the val split.

### Internal representations

The char-level scoring contract (rules 2 + 3) constrains the **eval-facing interface**, not the model's internal representation. Submissions should pick whichever internal unit (bytes, BPE, WordPiece, words, ...) best optimises their architecture's energy-to-accuracy ratio, and translate to per-char probabilities at the `CharModel` boundary.

- **Allowed.** Internal BPE / WordPiece / word vocabulary; subword-aware decoders; char-level marginalisation over partial-token continuations; hybrid architectures that mix per-char and per-token computation.
- **Encouraged** where it helps. Sub-quadratic mixers (Hyena, Mamba, etc.) benefit from shorter sequences; BPE typically gives 4–5× shorter context vs bytes, which can lift the per-step budget enough for those architectures to actually converge inside 300 s.
- **Tokeniser merge tables (GPT-2 BPE, SentencePiece, etc.) are not "pretrained weights".** They are deterministic algorithms over byte/codepoint streams, allowed under rule 1.
- **Pretrained tokeniser *model weights* are not allowed** — e.g., shipping a SentencePiece model that was trained on external data — same rationale as rule 1.

For an internal-BPE submission, `predict()` returns `P(next_char | observed_chars)` by marginalising over BPE tokens consistent with the current byte buffer: for an active partial buffer `p` and candidate next char `c`, `P(c) = Σ_t P(t | committed_context) · 1[bytes(t) starts with p+c]`, normalised over the candidate set whose tokens start with `p`.


[^1]: More energy efficient
[^2]: As of writing this
