# mamba_byte

Mamba-style selective state-space model on byte-level WikiText-103.

Paradigm: `WTX-I023` (linear-time SSM as transformer-attention substitute).
Claude-tag `CLA-005`-adjacent.

## Paradigm

Mamba (Gu & Dao 2024, arxiv 2312.00752) is a selective state-space
model whose per-step cost is independent of sequence length: each
block carries an `(d_inner, d_state)` hidden state and steps it
recurrently. Combined with a byte vocabulary (no tokeniser, no
out-of-vocabulary), the variant is "MambaByte" (Wang 2024, arxiv
2401.13660).

Hypothesis: at byte granularity a selective SSM with linear-time
complexity can chew through a longer context window (here 2048 vs the
transformer's 1024) at the same wall-clock, and the streaming-time
state is tiny so inference is fast — both training and the 60K-char
eval run cheaply relative to the modded_nanogpt baseline.

## Implementation choice

**Pure-PyTorch selective SSM** — no `mamba-ssm` / `causal-conv1d` CUDA
kernels. Rationale:

* The Modal `ghcr.io/ab-10/wikitext-bench:latest` image bundles
  `torch 2.5.1+cu124` but NOT `mamba-ssm`. Installing at `train()` time
  would burn ~30–60 s of the 300 s wall-clock cap and is brittle (the
  sdist builds against a specific torch ABI).
* The pure-PyTorch fallback is built from `torch.cumsum` / `exp`
  primitives that fuse acceptably in bf16 on A100. We give up the
  fully-fused `selective_scan_cuda` kernel speedup but keep the
  asymptotic O(n · d_state) memory and O(n · d_inner · d_state) time.

### Numerical stability: chunked scan

The naive "log-cumsum trick" — `h_t = exp(cs_t) · cumsum_k (b_k ·
exp(-cs_{k-1}))` — overflows fp32 once `cs` accumulates to ~ -90 or
lower (the corresponding `exp(-cs)` exceeds `1e39`). At our config
(`dt_max ≈ 0.1`, `A_max = -8`, per-step `log_decay` up to ≈ −0.8)
this overflow appears by L ≈ 50.

Fix: **chunk the scan**. We split the sequence into chunks of
`SCAN_CHUNK = 32` and use the parallel-scan trick *within* a chunk
while carrying the running hidden state across chunks recurrently:

```
for start, end in chunks(L, 32):
    cs    = cumsum(log_decay[start:end])             # ≤ 32 · 0.8 ≈ 26
    inner = b[start:end] * exp(log_decay - cs)        # bounded
    inner_cs = cumsum(inner)
    h        = exp(cs) * (inner_cs + h_carry)         # h_carry from prev chunk
    h_carry  = h[-1]
```

`exp(cs)` and `exp(log_decay - cs)` both stay in `[exp(-26), 1] ≈
[5e-12, 1]` — comfortably in fp32 range. Gradient flows through both
the within-chunk parallel scan and the cross-chunk carry.

### Streaming inference

Each `MambaBlock` exposes a `step(x_t, conv_state, ssm_state)`
recurrent path that updates a tiny O(1) state:

* `conv_state`: `(d_inner, d_conv)` — last 4 inputs to the causal
  depthwise conv.
* `ssm_state`: `(d_inner, d_state)` — the running SSM hidden state.

The `MambaByteCharModel` wrapper caches these per layer and takes one
recurrent step per observed byte. Per-byte cost is `O(n_layer ·
d_inner · d_state)`, **independent** of how many bytes have been
observed — no KV-cache trim, no context window. That's the structural
win over attention for the long streaming eval.

We verified parallel-scan vs step-by-step recurrent inference agree
to within 0.13% relative difference on a length-80 sequence spanning
multiple chunks (a consequence of fp32 cumsum-vs-recurrence
floating-point error; well below noise).

## Architecture

Per Mamba block:

```
x  -> in_proj -> [x', z]                    (Linear, expand=2)
x' -> conv1d(d_conv=4, depthwise, causal)
   -> silu
   -> selective_ssm(dt, B, C, A, D)         (B, C, dt all data-dependent)
z  -> silu                                  (gate)
out = out_proj(ssm_out * gate)              (Linear)
```

Stacked with pre-LayerNorm residuals; final `LayerNorm + lm_head`
(weight-tied to embedding).

| Hyperparameter | Value  |
|----------------|--------|
| `vocab_size`   | 256    |
| `d_model`      | 192    |
| `n_layer`      | 4      |
| `d_state`      | 16     |
| `d_conv`       | 4      |
| `expand`       | 2      |
| `d_inner`      | 384    |
| `dt_rank`      | 12 (auto: ceil(d_model/16)) |
| `ctx_len`      | 1024 (was 2048; halved to fit A100-40GB) |
| `batch_size`   | 16 (was 64; quartered to fit A100-40GB)  |
| `n_steps`      | 4000 (was 1500; bumped to recover tokens) |
| Optimizer      | AdamW (lr=3e-4, betas=(0.9, 0.95), wd=0.1) |
| LR schedule    | 5% warmup, cosine to 0 |
| Grad clip      | max-norm 1.0 |
| Params         | ~1.06M total |

Embedding init is rescaled to N(0, 0.02) so the tied lm_head produces
logits with `ln(256) ≈ 5.55` initial cross-entropy (default
`nn.Embedding` is N(0, 1) which yields a useless 185 starting loss).

`dt` bias is initialised so `softplus(bias)` is uniformly in
`[1e-3, 1e-1]` (per Mamba paper §3.6, "broad init of dt"), and `A` is
init'd as `-(1..N)` per inner channel (standard S4 init).

## OOM fix (post-mortem)

First attempt at `ctx_len=2048, batch_size=64` OOM'd on A100-40GB
inside `selective_scan`. Root cause: the chunked scan retains ~5-7
fp32 `(B, L, d_inner, d_state)` tensors per layer for backward —
even though the scan is chunked for *numerical* stability, autograd
still holds the full-sequence intermediates because the chunks are
concatenated into the layer output. At the original config the
forward-activation footprint was

  `5 tensors × 64 batch × 2048 L × 384 d_inner × 16 N × 4 B × 4 layer`
  `≈ 64 GB`

— more than 1.5× the GPU budget before even counting parameters,
gradients, optimizer state, conv activations, or the autocast bf16
shadow tensors.

Fix: halve `ctx_len` (2048 → 1024) and quarter `batch_size` (64 →
16), an 8× cut in per-step compute and a ~16× cut in scan-activation
memory. Bump `n_steps` (1500 → 4000) to recover some of the lost
token throughput within the 300 s wall-clock cap. New activation
footprint is ≈ 8 GB, comfortably inside 40 GB with headroom for the
rest of the training state.

We did NOT reach for the deeper fixes (gradient checkpointing,
recurrent-only forward, dropping `d_state` to 8) because the
arithmetic showed Option 1 alone has ~4× safety margin.

## Expected Modal numbers

Baseline (`modded_nanogpt`): 47,285 J / 0.7362 val char-acc / 294.9 s.
Current leader (`nano_plus_ngram`): 11,801 J / 0.7063.

For mamba_byte we target:

* **Energy: ~10–25 kJ.** The 4000-step train at d_model=192,
  ctx=1024, bs=16 has roughly `4000 * 16 * 1024 * 1.06M ≈ 7e13` FLOPs,
  ~10 % of the modded baseline's compute. The pure-PyTorch chunked
  scan is substantially slower than the fused CUDA kernel (we
  estimate 2-4× overhead from materialising the `(B, T, D, N)`
  intermediate). Net: roughly a third of the baseline's GPU time →
  10-20 kJ.
* **Wall-clock: ~80-220 s** (well under the 300 s cap).
* **Val char-acc: 0.69-0.73.** Highly uncertain — and lower than the
  pre-OOM-fix estimate because we trained on ~3× fewer tokens
  (64M vs 197M originally targeted). MambaByte's byte-level numbers
  in the paper are competitive with same-FLOPs transformers on
  enwik8 (BPC ~1.6) but the val target here is greedy-argmax
  char-acc, which weights short-range/format patterns heavily. Real
  risk we land at or below the 0.70 floor.

## Risks

* **Untested SSM on this task.** MambaByte literature is enwik8 BPC
  not char-acc; the metric weights short-range format/repetition very
  highly, which is roughly attention's home turf. We could land below
  the 0.70 floor and be DQ'd.
* **Pure-PyTorch scan slower than fused CUDA kernel.** We're trading
  away the main "Mamba is 5x faster than attention on A100" headline
  by not using `selective_scan_cuda`. The chunked PyTorch scan
  materialises the full `(B, T, D, N)` activation tensor (= for our
  config: `64 * 2048 * 384 * 16 * 4 B ≈ 3.2 GB` of fp32 activations
  per layer, smaller in bf16 autocast) which is memory-bandwidth
  bound. On A100 we expect 2-4× wall-clock vs fused kernel.
* **Numerical stability still fragile.** Even with chunking, very
  long contexts at extreme `dt` values can saturate `exp(cs)` to 0
  inside a chunk (gradient vanishing). Should be benign for our
  config but the scan is not as bulletproof as the CUDA kernel which
  uses a different in-kernel reformulation.
* **Chunk-boundary gradient.** The carry `h_carry = h[:, -1]` is fed
  into the next chunk's `exp(cs) * (inner_cs + h_carry)`, so
  gradients DO flow across chunk boundaries. We did not verify this
  matches the fully-recurrent gradient to high precision; if it
  diverges meaningfully (it shouldn't — same math, just rearranged)
  training could underperform an ideal selective scan.
* **Tied lm_head.** Weight-tying to the embedding halves the head
  param count but slightly couples representation and output spaces.
  On byte-level this is usually neutral or mildly positive.

## Smoke test

Ran on the 485-byte `fixtures/tiny` corpus:

```
[mamba] SMOKE mode (train=485 bytes)  ctx=64
[mamba] 0.03M params  cfg=TrainConfig(d=32 L=2 d_state=8 d_conv=4 expand=2 ctx=64 bs=2 steps=2)
SMOKE PASS
```

Smoke-mode triggers when `len(train_bytes) < 10_000` OR when
`SMOKE_TEST_ONLY=1` is set; it shrinks to `d=32, L=2, d_state=8, n_steps=2,
ctx_len=64, batch_size=2` so the test runs in ~1 s on CPU.

Independent sanity checks (run during development on CPU):

* Initial loss = 5.56 ≈ `ln(256) = 5.545` (init rescaled).
* Parallel-scan vs step-by-step recurrent forward agree to ~0.13%
  relative error on length-80 sequences spanning multiple chunks.
* Tiny train on a length-200-repetition of `"the quick brown fox jumps
  over the lazy dog. "` drives loss from 5.59 → 3.97 in 80 steps and
  achieves 86% char-acc on the same string — confirms the scan +
  optimizer + streaming inference are wired correctly.

## Author

`@claude-mamba` — experimental B4 / paradigm WTX-I023 / claude-tag
CLA-005-adjacent.
