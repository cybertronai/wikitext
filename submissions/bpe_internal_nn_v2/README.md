# bpe_internal_nn_v2 — Internal BPE transformer with multiprocess encode

**Paradigm:** Internal BPE tokenizer (tiktoken GPT-2 merges) + small
transformer trained on tokens, with marginalization at predict() to
return per-byte probabilities.

## Fixes over v1

v1 DQ'd at 300s after step ~1200/1500. The break-down:
- tiktoken encode_ordinary: **74s** (single-threaded, 540M bytes → 118M tokens)
- NN training: ~0.18s/step × 1500 = 270s budgeted
- Total: ~344s > 300s cap → DQ.

Two changes:
1. **Threaded encode** via `concurrent.futures.ThreadPoolExecutor` with
   N=8 workers. tiktoken's `encode_ordinary` is Rust and releases the
   Python GIL → true parallelism. Multiprocessing was tried first but
   the dynamically-imported `user_submission` module can't be pickled
   to subprocesses. Threads sidestep the pickling issue. Split at
   whitespace boundaries so GPT-2 BPE merges line up identically across
   chunks. Expected: 74s → ~10-15s.
2. **n_steps = 1000** (vs 1500). v1 loss at step 1000 was 4.40; at
   1200 was 4.25. Cap at 1000 trades 2pp acc for 50s headroom.
3. **max_len = 384** (vs 512). Minor compute reduction per step.

## Expected

- Energy: 15-25 kJ
- Accuracy: 0.71-0.74 (BPE may unlock better acc than 256-vocab char-level
  since longer effective context)
- L2-clean: yes (encode is CPU but bounded ~15s + GPU NN training dominates)

## Risk

- Multiprocess encode could behave differently across chunks (BPE merge
  boundaries). Mitigation: split at whitespace, which is a stable
  pre-tokenizer boundary in GPT-2's regex.
- L2 risk if multiprocess encode is too dominant. Mitigation: train
  duration is 200s+ of pure GPU activity, encode is ~5% of total.

## Smoke test

PASS on `fixtures/tiny/` (485 bytes → small NN, single-process encode).
