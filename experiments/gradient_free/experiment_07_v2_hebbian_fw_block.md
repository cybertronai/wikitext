# Experiment 07 v2: Hebbian Fast-Weight Block — wall-clock fix + normalization fix

## Status

v1 submitted as `submissions/hebbian_fw_block/` on 2026-05-25 17:28Z.
**DQ on `train_time_exceeded`**: 36.5 kJ at 300.001 s; only reached step 500/2150
before kill. Loss curve was healthy (5.55 → 1.23), no divergence. The kill mode
was the Python serial `for t in range(Tc):` scan inside `_scan_chunk`, not the
training rule.

## What v1 got wrong (re-ranked by post-run evidence)

1. **Critical (now confirmed): per-step Python loop dominates wall-clock.** v1
   `_scan_chunk` runs `T_chunk * 4` einsum ops per chunk (read, retrieve,
   residual, write) with Python dispatch overhead per step. Measured: ~0.59 s
   per outer step at T=1024, B=32. Baseline modded_nanogpt does the same
   per-block work in ~0.10 s. The fast-weight block alone consumed ~6× a
   normal block's compute, which is why only 500/2150 steps ran.

2. **Moderate (de-escalated from critical): L2-normalize + dropped sum-of-keys
   denominator is non-canonical.** The literature-faithful reference (Schlag,
   Irie, Schmidhuber 2021 arXiv 2102.11174 §4.2 / Eq. 29) uses
   **sum-normalization on a kernel feature map φ** (ELU+1 or DPFP), not L2 on
   raw projections. v1's `F.normalize(q, dim=-1)` + no denominator is a
   different rule. Post-hoc evidence: v1 loss curve was monotonically
   decreasing, so this rule is at least **stable** — not divergent as a worst
   case would predict. Promoted to moderate; the wall-clock bug is what
   actually killed the run.

3. **Minor (post-run): β-gate has no training signal under W.detach().** With
   W treated as a non-differentiable hidden state, the β projection trains
   only via the read path `o_t = W q_t` (which sees a detached W). β
   essentially does not learn; v1's `sigmoid(beta(x))` collapses to a
   roughly-constant value over training. v1 implements this correctly per
   spec; the spec is what's wrong about what β can buy.

## v2 design

Two-axis fix: parallelize the scan **and** restore a literature-faithful
normalization. Both are necessary; either alone DQs.

### Axis A — parallelized scan (mandatory)

Replace the Python `for t in range(Tc):` loop with the **WY-Householder
parameterization** from DeltaNet (Yang, Wang, Shen, Panda, Kim 2024 arXiv
2406.06484, Algorithm 2). At chunk size C=64, the chunk-internal scan becomes
two C×C triangular solves and one (C, C, d) matmul, fully parallel within the
chunk. Cross-chunk recurrence carries (B, d, d) W as before.

Concretely (pseudocode adapted from Yang 2024 Eq. 13–14):

```python
def _scan_chunk_wy(q_c, k_c, v_c, beta_c, W):
    # q_c, k_c, v_c: (B, C, d), beta_c: (B, C), W: (B, d, d)
    # Cross-chunk read: gradients carry through q_c.
    out_cross = torch.einsum('bde,bce->bcd', W, q_c)           # (B, C, d)

    # In-chunk causal scan compiled to triangular solves.
    # Form lower-triangular L: L[i,j] = beta_c[:, i] * (k_c[i] . k_c[j])
    L = torch.einsum('bcd,bed->bce', k_c, k_c)                 # (B, C, C)
    L = L * beta_c[:, :, None]
    L = torch.tril(L, diagonal=-1)                             # strict lower
    I = torch.eye(C, device=L.device).expand_as(L)

    # u = (I + L)^{-1} (v_c - W @ k_c)
    v_target = v_c - torch.einsum('bde,bce->bcd', W, k_c)      # (B, C, d)
    u = torch.linalg.solve_triangular(
        I + L, v_target * beta_c[:, :, None], upper=False, unitriangular=True
    )

    # Cross-chunk read addition: (q_c . k_c[<t]) u[<t]
    Att = torch.einsum('bcd,bed->bce', q_c, k_c)               # (B, C, C)
    Att = torch.tril(Att, diagonal=-1)
    out_in = torch.einsum('bce,bed->bcd', Att, u)              # (B, C, d)

    # Update cross-chunk W: W <- W + sum_t u_t k_t^T
    W_new = W + torch.einsum('bcd,bce->bde', u, k_c)
    return out_cross + out_in, W_new
```

Verified by reference (T=128, brute-force O(T²) Python scan) before any
Modal run. The reference test belongs in `submissions/hebbian_fw_block_v2/
test_scan.py`.

**Throughput target**: ≤ 0.18 s/step at T=1024, B=32 (within 2× of a normal
attention block). At 2150 × 0.18 = 387 s, still over budget — see Axis C.

### Axis B — normalization (mandatory; choose one)

Pick exactly one of the two literature-faithful options and document the
choice. Mixing them (v1's behavior) is what made v1 hard to interpret.

**Option B1 — Schlag sum-normalization (paper-faithful):**

```python
def phi(x):  # ELU+1 feature map (Katharopoulos)
    return F.elu(x) + 1.0

q = phi(self.q(h))                                    # (B, T, d), strictly positive
k = phi(self.k(h))
q = q / (q.sum(dim=-1, keepdim=True) + 1e-6)          # row-stochastic
k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)
```

This matches Schlag 2021 Eq. 29 exactly. v1's L2-on-raw is replaced.

**Option B2 — L2-norm with z-denominator (DeltaNet practice):**

```python
q = F.normalize(self.q(h), dim=-1, eps=1e-6)
k = F.normalize(self.k(h), dim=-1, eps=1e-6)
# Carry a per-stream z = sum_t β_t k_t alongside W; divide read by <q_t, z>.
```

Gated DeltaNet (Yang 2024 arXiv 2412.06464) uses this with a learnable forget
gate; choose B2 if going the DeltaNet route to inherit their published-known
hyperparameters.

**Default for v2: B1 (Schlag sum-norm + ELU+1).** It is the named rule the
docstring claims to implement.

### Axis C — budget realism

v1 had 2150 steps planned. v2 reduces to one of:

- **T = 512 (half v1's sequence length).** Halves per-step scan cost while
  keeping the per-byte target identical. n_steps = 2150 unchanged.
- **n_steps = 1500.** Match `ff_pretrain_then_sgd` Stage-2 count which lands
  ~0.71 at 48 kJ. With v2 axis-A speedup expected at ≤ 0.18 s/step:
  1500 × 0.18 = 270 s, leaves 30 s headroom.

**Pick T=512 for v2 (preserves longer-context training signal); fall back to
n_steps=1500 with T=1024 if T=512 lands < 0.68 acc.**

## Hypothesis

With axis-A parallelization + axis-B1 sum-norm + axis-C T=512: val char-acc
≥ 0.72 at energy ≤ 45 kJ, finishing in ≤ 290 s wall.

If this fails by ≥ 0.05 acc below `hopfield_layer` (0.7293), the conclusion
is **structural**: gradient-free delta-rule writes do not carry the same
inductive bias as Hopfield's frozen-prototype retrieval, despite Schlag 2021
showing they are formally equivalent at the linear-attention output level.

## Success criteria

- **Pass**: val ≥ 0.70 at energy ≤ 50 kJ in < 300 s. First Hebbian
  fast-weight non-DQ submission.
- **Strong pass**: val ≥ 0.73 (matches hopfield_layer) at any energy ≤ 50 kJ.
- **Refutation**: val < 0.65 in spite of n_steps and T being matched to
  baseline. Locks in "delta rule + no-grad ≠ Hopfield" as a finding.
- **DQ on time again**: forces re-spec to n_steps=1500 with T=512, or
  abandon as energy-frontier-incompatible.

## Failure modes & diagnostics

- **WY solve numerical instability** at bf16: cast triangular-solve to fp32
  per Yang 2024 §4.2; verified necessary on H100/A100 in their codebase.
- **β = sigmoid saturates near 0.5**: log `beta.mean()` and `beta.std()` per
  100 steps. If std < 0.05 by step 500, β is not learning — expected under
  the no-grad-on-W contract, document as confirming the moderate-#3 finding.
- **Cross-chunk W grows unbounded**: log `W.abs().mean()` per chunk. The
  delta-rule residual write is what prevents this in theory; sum-norm on k
  caps the contribution per token. If `W.abs().mean()` > 10 at chunk 8,
  there is a remaining numerical bug.
- **Reference-test mismatch > 1e-4** between WY-parallel and brute-force
  scan at T=128: hard-abort before Modal spend.

## References

- Schlag, Irie, Schmidhuber 2021 "Linear Transformers Are Secretly Fast
  Weight Programmers" arXiv 2102.11174, especially §4.2 / Eq. 29 (the
  sum-normalization rule v1 misattributed).
- Yang, Wang, Shen, Panda, Kim 2024 "Parallelizing Linear Transformers
  with the Delta Rule over Sequence Length" arXiv 2406.06484, Algorithm 2
  (WY-Householder parallel scan).
- Yang et al. 2024 Gated DeltaNet arXiv 2412.06464 (the L2-normalize +
  forget-gate variant for Option B2).
- Katharopoulos et al. 2020 ICML "Transformers are RNNs" — origin of the
  ELU+1 feature map used in Option B1.

## Cross-references

- `submissions/hebbian_fw_block/` (v1, DQ)
- `experiments/gradient_free/experiment_07_hebbian_fast_weight_block.md` (v1 spec)
- `submissions/modded_nanogpt/submission.py` (baseline reference for scan
  throughput target)
