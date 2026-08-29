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

## 8k matmul

[`benchmark_8k_matmul.py`](benchmark_8k_matmul.py) measures one full dense
`8192 x 8192` INT8-by-INT8 GEMM with an INT32 result. It uses the conventional
throughput convention of two operations per multiply-accumulate:

```text
8192^3 = 549,755,813,888 multiply-accumulates
2 x 8192^3 = 1,099,511,627,776 conventional operations
```

The measured pipeline starts with preallocated GPU buffers, fills both 64 MiB
input matrices with the nonzero constant `1`, then computes the full 256 MiB
output with `torch._int_mm`. The left input is row-major and the right input is
column-major (PyTorch's fast SM80 INT8 layout). Buffer allocation, CUDA/context
startup, warmup, and Modal boot are excluded; assigning all 134,217,728 input
entries and writing every output entry are included. A full-output check
verifies `C == 8192` before measurement.

### Modal result

The committed [raw result](8k_matmul_results.json) was measured on a Modal
A10G worker (`NVIDIA A10`, 150 W limit) with PyTorch 2.5.1+cu124 and CUDA 12.4.
Each row is the pooled mean of three aggregate trials. Initialization ran
164,601 times, while multiplication-only and the direct end-to-end pipeline
each ran 7,305 times; every trial lasted about 15 seconds and synchronized only
at its boundaries. `±` is one sample standard deviation across the three
aggregate trial means, not a confidence interval or full uncertainty estimate.

| Phase | Total runs | Time/run (ms) | Raw GPU-board J/run | Idle-adjusted GPU J/run | Share of isolated raw sum |
|---|---:|---:|---:|---:|---:|
| Initialize A + B | 164,601 | 0.2735 | 0.040892 ± 0.000016 | 0.024700 ± 0.000016 | 4.7% |
| Multiply A x B | 7,305 | 6.0850 | 0.836974 ± 0.002877 | 0.476704 ± 0.002952 | 95.3% |
| Direct initialize + multiply | 7,305 | 6.2806 | **0.889112 ± 0.004106** | **0.517257 ± 0.004137** | — |

The standalone stages sum to `0.877867 J` raw per pipeline, within 1.3% of the
direct `0.889112 J` measurement. Subtracting multiplication from the direct
pipeline gives a second initialization estimate of `0.052138 J`; together the
two methods put input initialization at roughly `0.041-0.052 J` of raw
GPU-board energy, or about 5% of the pipeline. GEMM-only throughput was 180.69
TOPS.

"Raw" is the full NVML board-energy delta. "Idle-adjusted" subtracts the
mean 59.21 W before/after idle baseline. The two calibrations spanned
53.83-64.58 W; propagating that spread puts the direct pipeline's adjusted
estimate at `0.4835-0.5510 J/run`, substantially wider than its workload-trial
standard deviation. These are GPU-board Joules, not the leaderboard's
CodeCarbon-assisted `total_energy_J`: host CPU, facility overhead, and Modal
startup are not included. Raw energy is the less model-dependent total;
idle-adjusted energy estimates the workload's incremental GPU cost. Energy
numbers are hardware-specific and should not be compared directly with A100
leaderboard runs.

Run on the task-pinned Modal A100-80GB, or reproduce the committed A10G run:

```bash
# In the venv from Quickstart; defaults to task.INSTANCE_TYPE (A100-80GB).
python benchmark_8k_matmul.py --yes

# Exact GPU class used for the committed result.
WIKITEXT_MATMUL_GPU=A10G python benchmark_8k_matmul.py --yes
```

## Current Leaderboard

> Submissions ranked by training energy (joules, lower wins) under the **2026-05-28 `CharModel.predict()` contract** (`predict()` returns a single committed `str`). Only submissions that have been ported to the new contract **and** re-run on the new harness appear here. The historical leaderboard is preserved below as a frozen record; entries there are not runnable under the new contract until ported. Tracking list of pending ports + re-runs lives at [`submissions/OUTDATED.md`](submissions/OUTDATED.md).
>
> Energy column reports **`total_energy_J`** (GPU NVML net of idle baseline + CodeCarbon CPU estimate, floored at `duration_s × 50 W`).

| Date | Energy (J) | Val char-acc | GPU | Config | Submission | Contributor |
|------|-----------:|-------------:|-----|--------|------------|-------------|
| 2026-05-28 |      2,866 | 0.7031    | A100 80GB SXM4 | subset_70_mkn     | [dir](submissions/subset_70_mkn)     | @gabrielnan (total_J = 1,321 gpu + 1,545 cpu) |
| 2026-05-28 |      5,116 | 0.7047    | A100 80GB SXM4 | paq_mixer_v3      | [dir](submissions/paq_mixer_v3)      | @gabrielnan (total_J = 2,026 gpu + 3,090 cpu) |
| 2026-05-28 |      5,214 | 0.7050    | A100 80GB SXM4 | gpu_ngram_w31_k11 | [dir](submissions/gpu_ngram_w31_k11) | @gabrielnan (total_J = 2,040 gpu + 3,174 cpu) |
| 2026-06-04 |     37,388 | 0.7409    | A100 80GB SXM4 | spectral_newton   | [dir](submissions/spectral_newton)   | @miyuhoriuchi + @atbcalvo (total_J = 26,557 gpu + 10,831 cpu) |

## Historical Leaderboard (pre-`bugfix/sampling`)

> ⚠️ **Frozen for history; not runnable under the current contract.** All rows below were scored against the prior `CharModel.predict() -> dict[str, float]` API, which the runner consumed via `argmax`. The new contract requires `predict() -> str` (committed character); submission code targeting the old contract crashes the runner with a type mismatch. Each entry needs a mechanical signature update (`return out` → `return max(out, key=lambda c: out[c]) if out else ""`, semantics-identical) and a re-run before its row can move to the Current Leaderboard above. See [`submissions/OUTDATED.md`](submissions/OUTDATED.md) for status.
>
> The `Energy (J)` column reports **`total_energy_J`** (GPU NVML net of idle baseline + CodeCarbon CPU estimate, floored at `duration_s × 50 W`) for rows dated **2026-05-20 and later**. Earlier rows report the prior NVML-only `training_energy_J`. The semantic change is the new total-system-energy rule per @yaroslavvb2's Telegram note; see `MAINTAINING.md` and the `EnergyMeter` source for details.

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
| 2026-05-19 |     60,864 |       DQ | A100 80GB PCIe | mamba_byte           | [dir](submissions/mamba_byte)           | @gabrielnan |
| 2026-05-20 |      1,752 |       DQ | A100 80GB SXM4 | gpu_ngram_w31_k10    | [dir](submissions/gpu_ngram_w31_k10)    | @gabrielnan |
| 2026-05-20 |     13,936 |       DQ | A100 80GB SXM4 | chunker_phase1_v2    | [dir](submissions/chunker_phase1_v2)    | @gabrielnan |
| 2026-05-20 |     24,417 |       DQ | A100 80GB SXM4 | bpe_internal_nn_v2   | [dir](submissions/bpe_internal_nn_v2)   | @gabrielnan |
| 2026-05-20 |     53,683 | 0.7246    | A100 80GB PCIe | lwta_k4              | [dir](submissions/lwta_k4)              | @ab-10 (re-run on new harness; total_J = 44,329 gpu + 9,354 cpu) |
| 2026-05-20 |     54,614 | 0.7145    | A100 80GB PCIe | lwta_k2              | [dir](submissions/lwta_k2)              | @ab-10 (re-run on new harness; total_J = 44,583 gpu + 10,031 cpu) |
| 2026-05-21 |      2,474 | 0.7031    | A100 80GB PCIe | subset_70_mkn        | [dir](submissions/subset_70_mkn)        | @gabrielnan |
| 2026-05-21 |      3,092 | 0.7050    | A100 80GB PCIe | gpu_ngram_w31_k11    | [dir](submissions/gpu_ngram_w31_k11)    | @gabrielnan |
| 2026-05-21 |      4,607 | 0.7047    | A100 80GB PCIe | paq_mixer_v3         | [dir](submissions/paq_mixer_v3)         | @gabrielnan |
| 2026-05-21 |      8,602 | 0.7184    | A100 80GB PCIe | gpu_ngram_o14_xorfix | [dir](submissions/gpu_ngram_o14_xorfix) | @gabrielnan |
| 2026-05-21 |      9,591 | 0.7063    | A100 80GB PCIe | chunker_phase1_v1    | [dir](submissions/chunker_phase1_v1)    | @gabrielnan |
| 2026-05-21 |     14,578 | 0.7184    | A100 80GB PCIe | deep_backoff_kn      | [dir](submissions/deep_backoff_kn)      | @gabrielnan |
| 2026-05-21 |     19,922 | 0.7328    | A100 80GB SXM4 | lwta_k4_alpha_065    | [dir](submissions/lwta_k4_alpha_065)    | @gabrielnan |
| 2026-05-21 |     20,743 | 0.7390    | A100 80GB SXM4 | alpha_06             | [dir](submissions/alpha_06)             | @gabrielnan |
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
