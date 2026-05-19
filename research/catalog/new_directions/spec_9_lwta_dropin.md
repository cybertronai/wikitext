# Research Specification 9: Local Winner-Take-All as a Drop-In Activation in modded-nanogpt

**Status:** Hypothesis evaluation (lowest-effort, fastest signal)
**Priority:** High (cost-adjusted)
**Estimated effort:** 2–6 hours

---

## Hypothesis

Replacing the ReLU/GELU activation in the modded-nanogpt MLP blocks with Local Winner-Take-All (LWTA, group size k=2 or 4) preserves val char-acc at or above 0.70 while reducing total training energy by ≥ 10%. The reduction comes from two compounding effects: (a) only 1/k of the MLP hidden units propagate forward, so post-MLP linear ops see a sparse activation; and (b) only 1/k of MLP weights receive a gradient per token, so the optimizer state traffic (Adam moments) drops by the same factor.

This is the lowest-engineering-cost direction in the entire research program. The hypothesis can be tested in a single morning.

---

## Background

Local Winner-Take-All (Srivastava et al., 2013 — "Compete to Compute") partitions an MLP's hidden units into groups of k. For each group, the forward pass keeps only the maximum-pre-activation unit; the rest are zeroed. The backward pass routes gradient only to the winner.

LWTA was originally proposed as a regularization mechanism (it forces specialization across the group), but the compute story is what motivates this spec. In a standard MLP block, the matmul `h_2 = W_2 · activation(W_1 · x)` costs full FLOPs regardless of activation. With LWTA, `activation(W_1 · x)` is structurally sparse with exactly 1/k nonzeros per row. The second matmul `W_2 · h_2` reduces to `sum over winners of W_2[:, winner] * h_2[winner]`, which is 1/k of the dense cost.

On A100 tensor cores, this sparsity is only useful if the sparse-matmul kernel is competitive with the dense one — which it is for structured (block-sparse, fixed-pattern) sparsity at the granularity LWTA produces. PyTorch's `torch.sparse` is not the right path; a custom CUDA kernel or Triton implementation is required to realize the FLOP savings. The first experiment uses dense kernels with a mask (no FLOP win but correct semantics) to establish whether accuracy holds; only if accuracy holds does the kernel investment make sense.

---

## What to build

**Modification to modded-nanogpt:** in each transformer block's MLP, replace `gelu(x)` (or whatever activation modded-nanogpt uses) with:

```python
def lwta_k(x, k):
    # x shape: (batch, seq, d_mlp); d_mlp divisible by k
    g = x.reshape(*x.shape[:-1], -1, k)  # group
    winner = g.argmax(dim=-1, keepdim=True)
    mask = torch.zeros_like(g).scatter_(-1, winner, 1.0)
    return (g * mask).reshape(*x.shape)
```

Two variants:
- **k=2:** half of units win, 50% structural sparsity.
- **k=4:** one in four wins, 75% sparsity.

Everything else in modded-nanogpt is unchanged.

For the kernel-side win (only if Phase 1 succeeds), implement a Triton kernel that fuses the LWTA winner selection with the subsequent `W_2 · h_2` matmul, skipping zero columns. Target speedup: ~k× on the MLP step; total wall-clock reduction is bounded by the MLP fraction of total compute (~50% in standard transformers).

---

## First experiment (go/no-go gate, Phase 1)

**Goal:** determine whether LWTA-k preserves val char-acc at the modded-nanogpt baseline configuration. This is the gate before any kernel work.

**Procedure:**

1. Clone the modded-nanogpt submission. Swap GELU for `lwta_k` with k=2 in **all** MLP blocks.

2. Run a full submission within the 300-second harness. Record val char-acc, training joules, training duration.

3. Repeat with k=4.

4. If both fail to reach 0.70, try a partial swap: LWTA only in the deeper half of MLP blocks (more redundancy where to absorb the sparsity), GELU in the shallow half. This is the standard remediation in the LWTA literature.

5. If a partial-swap configuration reaches 0.70, that is the Phase-1 winner; record its joules.

**Measurements to record:**

- Val char-acc for: (full-LWTA k=2), (full-LWTA k=4), (deep-only LWTA k=2), (deep-only LWTA k=4)
- Training joules and duration for each
- Mean fraction of MLP units active per token (sanity check: should be 1/k)
- Per-layer activation entropy: are winners diverse across the batch, or does the same unit always win?

---

## Go/no-go criteria

**Go (proceed to kernel work):** at least one LWTA configuration reaches val char-acc ≥ 0.70 within the 300-second harness, AND the per-unit win frequency is reasonably uniform (no single unit accounts for > 10% of wins across the batch, indicating no degenerate collapse).

**No-go:** no configuration reaches 0.70, AND the best configuration is below 0.65.

The most likely failure mode is **winner collapse**: a small subset of units win consistently, effectively reducing the model's capacity. Check per-unit win frequency in the measurements above. If collapse is the cause, two remediations exist:
1. Add an entropy regularizer on the winner distribution.
2. Initialize MLP weights with a small bias toward uniform winning (column-norm equalization).

Try remediation 1 first (one-line change); if it does not lift accuracy past 0.70, declare no-go.

**Borderline (best 0.68–0.70):** the activation change is borderline-compatible; do not pursue the kernel work, but note the residual gap and check whether the standard `Muon`-style optimizer in modded-nanogpt is interacting badly with the sparse activation. Try a brief AdamW fallback before declaring no-go.

---

## Phase 2 (conditional on Go)

Only if Phase 1 succeeds:

1. Implement the fused Triton kernel for `LWTA + W_2` step.
2. Rerun the winning configuration within the harness; measure realized wall-clock speedup and joules.
3. Compare against the dense-mask version to isolate the kernel contribution from any accuracy noise.

The kernel win is bounded by the MLP fraction of total transformer compute, which is roughly 2/3 in standard transformers and lower in modded-nanogpt (which uses Muon-trained heavy attention). The realistic upper bound on energy reduction is ~30%.

---

## What a positive result means

A positive Phase-1 result is a near-free leaderboard improvement on the modded-nanogpt baseline. The kernel work in Phase 2 turns the structural-FLOP win into a real-joule win.

The deeper question after go/no-go is: **does LWTA's selection sparsity compose with attention sparsity (e.g., Top-K attention) for an additive energy gain?** The two mechanisms are independent (one in MLP, one in attention), so they should compound.

---

## What a negative result means

A negative result means modded-nanogpt's MLP capacity is already saturated by the dense activation, and removing (k−1)/k of forward signal cannot be recovered by training. Note which failure mode applied: capacity (accuracy degraded smoothly with k) or collapse (a few units dominating). The two have different research implications.

A negative result also informs Spec 6 (Hebbian floor) and the LWTA variant of Spec 12 (chunker + LWTA hybrid): if LWTA degrades modded-nanogpt by more than 0.05 char-acc, those hybrids should drop LWTA from their stacks.

---

## Resources

- Paper: Srivastava, Masci, Kazerounian, Gomez, Schmidhuber, NeurIPS 2013 — "Compete to Compute" — https://papers.nips.cc/paper/5059-compete-to-compute
- Repository stub: `cybertronai/schmidhuber-problems`, branch `compete-to-compute`
- Baseline to modify: `submissions/modded_nanogpt/` (current best at 51,704 J / 0.7374)
- Triton documentation: https://triton-lang.org
- Harness: 300-second wall-clock, A100-80GB, NVML joule measurement
