# Research Specification 14: Mamba (Selective SSM) Byte-Level Language Model

**Status:** Hypothesis evaluation (safe-novelty Manning-branch entry)
**Priority:** High (off-the-shelf, lowest implementation risk)
**Estimated effort:** 1 day

---

## Hypothesis

A pure Mamba stack (no attention) trained from scratch on bytes reaches val char-acc ≥ 0.70 within 300 s on A100-80GB at training energy **≤ 42 kJ**, beating both modded-nanogpt (51.7 kJ) and `lwta_k2` (46.1 kJ).

Mamba's selective state-space mechanism (Gu & Dao, 2023, [arxiv:2312.00752](https://arxiv.org/abs/2312.00752)) gives linear-time sequence mixing with content-dependent ∆, B, C gating, which provides a recency/selectivity inductive bias well suited to **highly autoregressive byte streams** where local-conditional next-byte entropy is low. Published claim: Mamba-3B matches transformers of 2× size; **5× higher generation throughput than transformers**. This is a structural FLOP-and-memory win that should translate to joule savings at training time on our budget.

---

## Background

Mamba is a structured state-space model (SSM) of the form

```
h_t = A_bar(x_t) · h_{t-1} + B_bar(x_t) · x_t
y_t = C(x_t) · h_t
```

where `A_bar`, `B_bar`, `C` are functions of the input `x_t` (the **selectivity**). The hidden state `h_t` has fixed dimension `d_state` (typically 16) regardless of sequence length — constant memory per step, unlike attention's growing KV cache. The selective scan is implemented via a custom CUDA kernel (`selective_scan_cuda`) that processes the sequence in O(N · d · d_state) with linear memory.

**Why this fits the benchmark:**
- Linear-time training: no N² attention wall at any seq length.
- Constant-state inference: streaming `predict()` is one Mamba step (O(d · d_state) FLOPs per char), trivially fast inside the 60K-char eval.
- Strong recency bias: byte trigrams are near-deterministic; Mamba's content-dependent ∆ should latch onto these efficiently.

**Reference implementation:** `pip install mamba-ssm` (state-spaces/mamba) gives `MambaBlock`. The custom CUDA kernel comes pre-built for A100; no kernel work required.

---

## What to build

**Submission: `submissions/mamba/submission.py`.** Use the modded-nanogpt scaffold for byte embedding, RMSNorm pre-norm, output projection, optimizer, LR schedule, and CharModel wrapper. Replace the transformer block body with a Mamba block.

**Per-block structure (verbatim Mamba):**

```python
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    def forward(self, x):
        return x + self.mamba(self.norm(x))
```

**Stack:** byte embed → `N` MambaBlocks → final RMSNorm → linear head (tied to embed if room). No MLP block (Mamba already has internal expand=2 expansion that subsumes MLP).

**Sizing target.** Match modded-nanogpt parameter count within 10%:
- d_model = 512, layers = 16, d_state = 16, d_conv = 4, expand = 2

**Training.**
- Optimizer: AdamW (Muon is over-engineered for Mamba's 1-D state mixing weights; AdamW is the published recipe).
- LR schedule: cosine with 10% warmup, peak ≈ 5e-4 (Mamba paper uses this).
- Batch / seq_len: pick to saturate A100-80GB memory. Probably batch=24, seq_len=2048.
- Mixed-precision: bf16 (Mamba's selective_scan_cuda supports bf16).

**Streaming inference (CharModel.predict).** Mamba can run in *recurrent mode* by stepping the SSM one token at a time with state carry-over. The `mamba-ssm` package exposes a `step(x, state)` interface for this. Implement CharModel.predict via recurrent stepping (O(d · d_state) per char, ≪ O(seq · d) full re-run).

---

## First experiment (go/no-go gate)

**Goal:** measure pure-Mamba's joule cost to reach 0.70 char-acc.

**Procedure:**

1. Implement `submissions/mamba/submission.py` per the structure above. Add `mamba-ssm` and its CUDA dep (`causal-conv1d`) to the submission's runtime; submit.py installs from PyPI inside the Modal image.

2. Smoke-test locally if `mamba-ssm` installs on the dev VM; otherwise go straight to Modal.

3. Submit. Record metrics.

4. If val char-acc < 0.70, try **one** remediation: layers 16 → 20 with seq_len cut 2048 → 1536 to stay under 300 s.

**Modal image note.** `mamba-ssm` requires `causal-conv1d` which needs a CUDA build. The submit.py prebuilt image (`ghcr.io/...`) may not include these. The agent may need to add a custom Modal Image step:

```python
image = (modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04")
    .pip_install("torch==2.3.0", "mamba-ssm==2.2.2", "causal-conv1d==1.4.0", ...))
```

If image rebuild blows past 5 min, fall back to a **pure-PyTorch Mamba implementation** (slower but no custom kernel) — `johnma2006/mamba-minimal` is a 100-LOC clean reference. This is a known engineering risk.

---

## Go/no-go criteria

**Go:** val char-acc ≥ 0.70 AND training joules ≤ 46 kJ. Mamba becomes the new leaderboard top.

**Borderline:** val char-acc ≥ 0.70 but joules in (46 kJ, 52 kJ]. Mamba beats modded-nanogpt but not lwta_k2. Report; do not pursue kernel optimization.

**No-go:** val char-acc < 0.70 after one remediation. Mamba's recency bias does not translate to byte-level char-acc under 300 s; or the package install ate the budget. Distinguish these two cases in the report.

---

## Phase 2 (conditional on Go)

1. **Mamba + LWTA in the inner expand-MLP.** Mamba's `expand=2` block has an internal SwiGLU-style activation. Swap that for LWTA-k=2 → compound the two energy savings.
2. **Hybrid Mamba + 2-attention layers (H3 / Striped Mamba).** H3 found two attention layers suffice on top of SSM for associative recall. Test whether two attention layers cost less joules than they save in accuracy.
3. **State scaling:** d_state from 16 → 64. Larger state hurts FLOPs but is a known accuracy lever.

---

## What a positive result means

A pure-Mamba win on byte char-LM is the first evidence on this benchmark that **SSM beats both attention and LWTA** on the joule axis. The interpretive lever for the program is: **content-dependent recurrence is a real mechanism, not just a paper trend**. It also positions H3-style hybrids (Mamba + sparse attention) as the natural follow-up.

---

## What a negative result means

Two cases to disambiguate:

1. **Engineering failure (package install ate budget, kernel build broke).** Not a research result; rerun with the pure-PyTorch fallback.
2. **Capability failure (val char-acc < 0.70 with working implementation).** SSM's recurrence-only structure cannot match attention's exact-recall at byte level under 300 s. This is interesting — suggests **attention's exact KV lookup is doing real work** that recurrence cannot substitute. The H3 hybrid spec (2-attention + Mamba) becomes the targeted follow-up.

---

## Resources

- Paper: Gu, Dao — "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" — [arxiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Official impl: https://github.com/state-spaces/mamba
- Pure-PyTorch reference (fallback): https://github.com/johnma2006/mamba-minimal
- Baseline to modify: `submissions/modded_nanogpt/`
- Current leader: `submissions/lwta_k2/` at 46.1 kJ / 0.7146
- Harness: 300 s, A100-80GB, NVML joules, val char-acc ≥ 0.70 on 60K val chars
