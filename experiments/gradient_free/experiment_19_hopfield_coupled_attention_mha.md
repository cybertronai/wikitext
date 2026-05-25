# Experiment 19: Hopfield-Coupled Attention (MHA) — Cross-Layer Score EMA in a 4-Layer Trunk

## Hypothesis
Replacing vanilla self-attention with **Modern Hopfield Attention** (Masumura & Taki, NeurIPS 2025, arXiv 2511.20698) on the 4-layer Muon trunk recovers most of the −0.008 char-acc gap that `hopfield_layer` shows vs 6-layer `modded_nanogpt`, at **strictly lower energy** than the 6-layer baseline, with **zero added trainable parameters**. Concretely: at α' = 0.5 we expect val char-acc ≥ 0.735 at energy ≤ 42 kJ. The α' = 0 cell of the sweep is the *de facto* 4-layer-Muon baseline whose absence is the central attribution gap in the current Hopfield portfolio.

## Motivation
Three convergent reasons make this the right next Hopfield experiment, not another K_mem variant.

**1. The current portfolio cannot attribute the `hopfield_layer` PASS to Hopfield.** `hopfield_layer` (4-layer Muon trunk + frozen-random-K retrieval, M=4096) hit 0.7293 at 40.2 kJ vs 6-layer baseline 0.7374 / 51.7 kJ. Without a 4-layer-Muon-without-Hopfield control, the result is consistent with either "Hopfield contributes ~0 pp" or "Hopfield contributes ~2 pp." Exps 01–04, 10 all vary the Hopfield bank but **share** the same trunk-without-baseline confound. An honest portfolio needs an attention-side Hopfield experiment whose α' = 0 cell IS the missing baseline.

**2. The hopfield_layer Hopfield is barely Hopfield.** K_mem is built by encoding train windows through the **random-init** first two transformer blocks (`submission.py:452`, `_init_hopfield_memory:367`). Keys are a fixed random projection of bytes; only queries see SGD. Even if exp 01's M-sweep finds a capacity scaling, what's being tested is "frozen-random-projection retrieval," not modern-Hopfield associative memory. MHA, by contrast, derives the Hopfield update from continuous-time Hopfield dynamics *without* the adiabatic approximation — i.e., MHA is what you get when you take the Hopfield ODE seriously instead of collapsing it to a single softmax. The cross-layer EMA term `h_n = α' h_{n-1} + (1−α') Q_n K_n^T` IS the non-adiabatic correction. Setting α' = 0 recovers vanilla attention. **This is the cleanest "Hopfield ≠ attention" parameter in the literature.**

**3. Independent empirical signal on our exact task.** Masumura & Taki report MHA improves GPT-2 small on WikiText-103 from 22.87 → 20.70 PPL (table 1) and ablates α' = 0 to 69.89 vs α' = 0.5 to 72.13 on CIFAR-100 ViT-Tiny (table 6, +2.24 pp). Their wins are on the *same corpus* we evaluate on (different tokenization — they use BPE, we use bytes — but the rank-collapse / token-uniformity mechanism they cite is tokenization-invariant). Importantly, **MHA adds no trainable parameters**, so the energy cost is essentially baseline + a per-layer EMA add — directly attributable.

This experiment is also the cleanest extension of the existing Hopfield-direction theory: every prior Hopfield design in this repo (exps 01, 02, 03, 04, 10, hopfield_layer) varies the **external memory bank**; not one varies the **attention-internal Hopfield coupling**. The portfolio has been exploring one axis of a 2-axis space.

## Method
Replace `CausalSelfAttention` in modded-nanogpt with `HopfieldCoupledSelfAttention`. Single architectural change: between layers, pass a per-head per-position **attention-score EMA** `h` and use it as the pre-softmax logits.

Pseudocode (one layer):
```
# vanilla:  scores = q @ k.T / √d ;  attn = softmax(scores_causal) ;  y = attn @ v
# MHA:
scores = q @ k.T * scale            # (B, H, T, T) — pre-causal raw logits
h = α' * h_prev + (1 - α') * scores  # EMA across layers (h_prev = scores at layer 0)
attn = softmax(causal_mask(h))      # softmax of the EMA, not raw scores
y = attn @ v
return y, h                          # pass h to next layer
```

Residual stream and MLP block structure are **unchanged**:
```
x = x + attn(norm1(x), h_prev)  # attn returns (h_attn_out, h_new)
x = x + mlp(norm2(x))
```

α' ∈ [0, 1] is the only new hyperparameter; it is a fixed **scalar** (not learned in this experiment — keeps attribution clean). We sweep it.

**Critical implementation note**: this requires materializing the score matrix, so `F.scaled_dot_product_attention` (which fuses softmax) cannot be used. We write explicit `softmax(QK^T)V` with the math backend. At B=32, H=6 (d=384/d_h=64), T=1024, score tensor = 32·6·1024·1024·2 B = 384 MB per layer bf16. Two layers' worth held simultaneously (current + previous h) = 768 MB peak, well within A100 80GB. Backward pass needs to save activations — torch's checkpoint API can hold this in 1.5 GB.

**Streaming inference**: at predict step t, layer ℓ holds a per-position EMA cache `h_t^ℓ ∈ R^(1,H,1,T)`. We pass this layer-to-layer at each token. Per-token cost is O(T·H·d_head), same as one row of normal attention — no extra asymptotic cost.

## Memory-Movement Analysis
- **Per training step**: with SDPA we'd have ~3-4× speedup vs math attention; explicit math attention at T=1024 is ~2.5× slower in our regime. Attention is ~35–40 % of per-step time in modded-nanogpt at L=4; total slowdown estimate ≈ 1.5×. 4-layer SDPA modded-nanogpt baseline would take ~170 s (66 % of the 6-layer 246 s baseline); 4-layer MHA target ≈ 255 s, comfortably under the 300 s cap.
- **Per-layer h tensor**: (B, H, T, T) bf16 = 384 MB. Total active = 768 MB (current + prev). Negligible against 80 GB.
- **Extra FLOPs**: one (B, H, T, T) elementwise EMA per layer = 200 M ops per step. <0.1 % of total. Energy delta vs 4L vanilla: O(1 %).
- **Parameter delta vs baseline**: **zero**. (α' is a Python float.)

## Setup
- Dataset, optimizer, n_steps, batch, T: **identical** to `submissions/hopfield_layer/submission.py` (n_steps=2150, bs=32, T=1024, Muon+AdamW).
- Trunk: **4-layer** modded-nanogpt (matches the trunk in `hopfield_layer`; this is the depth at which the comparison is informative).
- New scalar: α' ∈ {0.0, 0.3, 0.5, 0.7}.
- Attention internals: explicit-math `softmax(causal_mask(EMA(QK^T)))V`; KV cache same as `hopfield_layer`.
- No frozen memory bank, no LWTA, no other mechanism — **isolated**.

## Procedure
The submission is implemented at `submissions/mha_alpha05/submission.py`: 4-layer Muon trunk + `HopfieldCoupledAttention` with `F.scaled_dot_product_attention` (memory-efficient backend) consuming an additive `attn_mask` — see "Kernel" section below for the algebraic rewrite that makes this fused. α' is hardcoded per submission directory (one directory per α' value, matching the existing repo convention).

Sibling submissions are created once via:
```bash
for alpha in 00 03 07; do
  cp -r submissions/mha_alpha05 submissions/mha_alpha${alpha}
  sed -i "s/else 0.5\$/else 0.${alpha:1}/" submissions/mha_alpha${alpha}/submission.py
done
```

Then the sweep:
```bash
python3 submit.py submissions/mha_alpha00 --yes  # α'=0 control — runs first
python3 submit.py submissions/mha_alpha05 --yes  # α'=0.5 headline
python3 submit.py submissions/mha_alpha03 --yes
python3 submit.py submissions/mha_alpha07 --yes
```

Each directory writes its own `result.json`, no overlap. A 6-layer α' = 0.5 follow-up (if the primary PASSes) requires editing `TrainConfig.num_layers` to 6.

A correctness + perf smoke test for the kernel itself lives at `submissions/mha_alpha05/test_kernel.py`. Run it on any CUDA box before launching Modal jobs:
```bash
python submissions/mha_alpha05/test_kernel.py
```
Checks: (i) α' = 0 path matches vanilla SDPA bitwise-ish, (ii) α' = 0.5 path matches an explicit math-attention reference, (iii) backward gradients flow non-NaN and non-zero through the cross-layer h chain, (iv) forward throughput is ≤ 1.5× SDPA (the bound used to budget wall-clock).

## Kernel
The MHA update needs to materialize the (B, H, T, T) score-EMA tensor at each layer so the next layer can EMA against it (at our shapes B=32, H=6, T=1024 bf16: 384 MB per layer). Naive math attention (compute scores → softmax → @V as three ops) is HBM-bound and ~1.5–2× slower than SDPA at these shapes.

`HopfieldCoupledAttention` exploits SDPA's additive `attn_mask` argument with a small algebraic rewrite:

```
h_n = α' h_{n-1} + (1 − α') Q K^T · scale
    ≡ Q K^T · scale_sdpa + attn_mask     where scale_sdpa = (1 − α') · scale,
                                              attn_mask  = α' · h_{n-1}
```

so `softmax(h_n) V = softmax(Q K^T · scale_sdpa + attn_mask) V`, which is exactly what `F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale_sdpa)` computes — routed to PyTorch's memory-efficient attention backend on A100 (xformers-derived, fused, supports backward natively, no torch.compile or custom autograd.Function needed). The (T,T) causal mask broadcasts into the additive mask add, no extra materialization.

The next layer's `h_n` is still materialized via one explicit `torch.matmul(q, k.transpose(-2, -1)) * scale` outside SDPA — unavoidable, because cross-layer state requires HBM persistence.

Streaming inference (T_query = 1) uses an explicit math path; the score tensor is < 6 MB and SDPA's per-call overhead would dominate.

**Tried first**: `torch.nn.attention.flex_attention` with a captured-tensor `score_mod`. Worked for forward (α'=0 matched SDPA bitwise on Modal A100). Failed at α'>0 because PyTorch 2.5.1's `FlexAttentionAutogradOp` asserts `not any_buffer_requires_grad` on captured tensors — fixed in 2.6+, not in our pinned Modal image (`ghcr.io/ab-10/wikitext-bench:latest` = torch 2.5.1+cu124). FlexAttention also requires a host C compiler at runtime for Triton's JIT, which the leaderboard image doesn't ship. The SDPA-bias rewrite sidesteps both issues and the pre-flight smoke test (`test_kernel.py`) measured a 2.60× fwd+bwd slowdown of the attention stack — translating to ~70 s of extra wall-clock over the full 2150-step training, comfortably inside the 300 s cap.

## Success Criteria
- **Attribution-PASS (primary goal)**: α' = 0 cell completes the leaderboard. Whatever it scores, the gap between α' = 0 and `hopfield_layer` (0.7293) is the unambiguous contribution of the frozen-K_mem retrieval layer. The gap between α' = 0 and α' = 0.5 is the unambiguous contribution of the Hopfield-coupled attention mechanism.
- **Strong PASS**: α' = 0.5 achieves val ≥ 0.735 at energy ≤ 42 kJ — recovers 6-layer accuracy at <82% of its energy with a strictly Hopfield-attributable mechanism.
- **Weak PASS**: α' = 0.5 > α' = 0 by ≥ 0.005 acc at matched energy — demonstrates non-zero Hopfield utility distinct from depth or external memory.
- **Refutation**: α' ∈ {0.3, 0.5, 0.7} all within ±0.003 of α' = 0 → at char-level depth-4, cross-layer Hopfield coupling buys nothing; the published GPT-2 / ViT wins do not transfer to small char-LM. **This is itself a publishable negative result** — it would say that rank collapse / token uniformity (the proposed MHA mechanism) is not a bottleneck for 4-layer char-LM.

## Failure Modes & Diagnostics
- **Time-cap DQ**: with the SDPA-bias kernel the smoke test measured 2.60× fwd+bwd slowdown of the attention stack at training shapes (per-step 32 ms extra, projected 69 s extra over 2150 steps). 4L baseline target is ~165 s, so MHA projects to ~234 s with ~66 s headroom vs the 300 s cap. If a Modal run still exceeds 290 s, drop `n_steps` from 2150 to 1800 across all four α' cells (keeps comparisons matched).
- **Softmax saturation at high α'**: large α' makes h accumulate magnitude across layers (it's a running average of unbounded scores). If max(|h|) > 30 at the last layer, softmax goes one-hot. Diagnostic: log `h.abs().max(), h.std()` at layer L−1, step 1000. Mitigation: insert an RMSNorm on `h` before passing to SDPA's `attn_mask`. Add only if diagnostic flags.
- **α' = 0 sanity check fails**: if the α' = 0 cell doesn't match a vanilla SDPA accuracy run within 0.003 char-acc, the kernel or scale logic has a bug. The pre-flight `test_kernel.py` test 1 catches the layer-level version of this; it confirmed exact equality (diff = 0) on Modal A100 because at α'=0 the code takes the `is_causal=True` SDPA fast path directly.
- **MHA wins on training loss but not val acc**: published MHA results are on perplexity; char-acc is argmax-only and less sensitive to distribution-sharpening. Mitigation: also log val NLL.
- **Streaming inference numerical drift**: the streaming path (T=1) uses explicit math while training uses SDPA-with-bias — output may drift slightly. Diagnostic: compare full-batch eval acc vs streaming eval acc on the first 1000 valid chars; if diff > 0.005, fp32-promote the streaming softmax (already done — it casts h.float() before softmax).
- **Memory-efficient backend missing on host**: PyTorch's SDPA dispatcher needs the xformers-derived efficient-attention kernel to handle the additive mask. The Modal image's torch 2.5.1+cu124 has it. If a future image switches to a CUDA build without efficient-attention, SDPA falls back to the MATH backend (correct but ~3× slower than memory-efficient), which would bust the budget. Diagnostic: print `torch.backends.cuda.sdp_kernel_choice()` at first forward.
- **A100 SKU heterogeneity across sweep cells**: Modal's `gpu="A100-80GB"` label covers both PCIe and SXM4 variants. The leaderboard `gpu_name` column records which variant landed. Observed during this sweep: α'=0, 0.3, 0.7 ran on PCIe (idle 55 W, stress 232 W, stress_energy ≈ 8.7 kJ), α'=0.5 ran on SXM4 (idle 64 W, stress 350 W, stress_energy ≈ 13.2 kJ). Raw `training_energy_J` between SKUs is not apples-to-apples — SXM4 is ~50% more power-hungry per unit work. Mitigation: report **normalized energy = training_J / stress_J** as the cross-SKU comparable metric. PCIe baselines (`hopfield_layer` = 40.2 kJ / 8.7 kJ-stress ≈ 4.6, `modded_nanogpt` = 51.7 kJ / 8.7 kJ-stress ≈ 5.9) are the right reference for that ratio.

## Why This Beats the Existing Tier-3 Ablation
The proposed Tier-3 "M ∈ {0, 1024, 4096} + Hopfield-vs-self-attention-head" matrix tests *external-memory* Hopfield. Exp 19 tests *internal* Hopfield. Both are valid, but **Exp 19 is strictly more informative per run**:
- The α' = 0 cell **is** the missing 4-layer-vanilla baseline (resolves the original attribution critique in one run).
- The α' > 0 cells test a published, theoretically-grounded Hopfield mechanism with no confound from random-projection keys.
- No external memory bank → no "is the bank random / data-derived / trained?" confound (the structural problem with hopfield_layer).
- 4 runs total vs 3, but each is more diagnostic; the M-sweep can run in parallel if desired.

## Estimated Cost
4 Modal A100-80GB runs × ~4–5 min wall each (4-layer modded-nanogpt scaffolding is ~170s base; MHA overhead pushes to ~250s) ≈ **$1.60**. If the optional 6-layer α' = 0.5 follow-up runs, add ~$0.50. Total budget: **≤ $2.10**.

## References
- Masumura & Taki 2025 "On the Role of Hidden States of Modern Hopfield Network in Transformer" (NeurIPS 2025, arXiv 2511.20698) — primary
- Ramsauer et al. 2020 "Hopfield Networks Is All You Need" (arXiv 2008.02217) — adiabatic-Hopfield-as-attention identity (which MHA generalizes)
- Krotov & Hopfield 2016 "Dense Associative Memory for Pattern Recognition" — exponential storage in continuous-state Hopfield
- Hoover et al. 2024 "Energy Transformer" (NeurIPS 2023) — alternative non-adiabatic Hopfield-in-transformer line; could be a follow-up
- Martins et al. 2023/2024 "Sparse Modern Hopfield Networks" / "Sparse and Structured Hopfield Networks" — orthogonal axis (sparsemax instead of softmax) — possible exp 20
- `submissions/hopfield_layer/submission.py`, `submissions/modded_nanogpt/submission.py` — implementation references
- `research/kernel_methods/result_11.md` — the result this experiment attributes
- `experiments/gradient_free/experiment_01_hopfield_memory_sweep.md` — orthogonal external-memory axis (complementary, not redundant)
