# Spec 07 — Context-Tree Weighting (CTW) — Results

**Status:** DQ — below 0.70 char-acc floor.

| Metric | Value |
|---|---|
| Submission dir | [`submissions/ctw_d24/`](../../submissions/ctw_d24) |
| GPU | A100-80GB SXM4 (Modal) |
| Wall-clock (train) | 147.4 s |
| Energy (train) | **715.8 J** |
| Val char-acc | **0.4752** |
| DQ reason | val_accuracy_below_floor (0.4752 < 0.70) |
| Modal run | https://modal.com/apps/ab-10/main/ap-Q0lozW3ZOAR0XAttLaT0Ks |

Compared to:
- modded_nanogpt baseline: 51,704 J / 0.7374
- PPM-C order-6 (closest cousin): 735 J / 0.6525
- Spec's predicted energy: 1–5 kJ
- Spec's predicted acc: 0.65–0.73

## Implementation

Bit-level CTW (Willems-Shtarkov-Tjalkens 1995) with depth D = 24 bits
(3 bytes of bit context). Inline C extension built on Modal at start-up
(gcc apt-installed), hash-array trie of ≤40M nodes, 20 B/node.
Numerically-stable formulation stores **β = P_e / (P_e + P_w_children) ∈
[0, 1]** per node instead of the cumulative log_pw / log_pe, so the per-
bit update only involves bounded quantities:

```
R(d, b) = β_d · kt_d(b) + (1 − β_d) · R(d+1, b)
β_d_new = β_d · kt_d(b) / R(d, b)
```

Byte prediction tree-expands the 8 bits in nested fashion (255 bit
predictions/char, ~10.7K char/s on the Modal worker).

Training streams the first 80 MB of the train split — the host ingested
all 80 MB at 0.54 MB/s, leaving 150 s of unused budget.

## What worked, what surprised

- **β-parametrisation was load-bearing.** First attempt stored log_pw /
  log_pe directly; precision destroyed at ~10⁸ accumulated bits and
  prediction collapsed to the all-zeros byte (0% acc). Switching to the
  bounded β ratio (Willems' standard online form) fixed it.
- **Initial β = 0.5 (not 1.0).** A virgin node has P_e = 1 and
  P_w_children = 1·1 = 1, so β = 1/(1+1) = 0.5. Setting β = 1 at
  allocation makes the CTW collapse to per-node KT and ignore deeper
  context — the same all-zeros symptom.
- **D = 24 bits = 3 byte-equivalent context.** This is shallower than
  PPM-C's order-6 byte trie (48 bits of bit-context). Empirically that's
  the dominant reason 0.4752 < 0.6525 (PPM-C). The Bayesian-optimal
  mixing of CTW does NOT recover the gap from halving the context
  depth.

## Did it beat the spec's energy estimate?

Yes, on energy: 716 J vs the spec's 1–5 kJ band (lower end). Also
beat PPM-C's 735 J marginally.

No, on accuracy: 0.4752 is **below** the spec's predicted 0.65–0.73
band. The spec extrapolated CTW's published enwik8 bpc (1.85–2.0,
implying char-acc ~0.65–0.73), but those numbers correspond to deeper
D (cmix uses D ≥ 48) and to bytewise (not bit-aligned) variants that
mix per-byte tables.

## Worth a second-round investment?

**Maybe, but not at this depth.** The accuracy floor is tied to context
depth, not to algorithmic refinement; doubling D would push memory to
~5 GB (still fits 80 GB) and roughly double train time (147 → ~300 s,
right at the budget). A speculative second-round design would be:
**bytewise CTW** (Sadakane-Okazaki-Imai 2000), which keeps the
Bayesian mixing but conditions on byte-aligned tables and mixes 256
candidates per node — this is what cmix actually uses. Energy ~ PPM-C
+ small mixer overhead; expected acc 0.68–0.72 on this corpus.

The headline win of the current result is the **716 J floor**: lowest-
energy submission attempted in this portfolio, and a clean
demonstration that the harness will accept a CPU-only, NVML-quiet,
sub-1 kJ pipeline. The leaderboard slot is gated by the 0.70 floor,
not by energy, so the natural next step is the bytewise-CTW redesign
rather than tuning D.

## Review (post-hoc audit)

**Validity for discarding CTW on char-LM:** *Valid only for D=24 bit-level CTW.*

**Core limitations:**
- **D=24 is shallower than the published-state-of-the-art CTW variants** (cmix uses D ≥ 48 with byte-aligned bytewise tables). The writeup correctly identifies depth as the dominant bottleneck and does not claim to have discarded CTW in general.
- **Train was throughput-bound at host I/O** (0.54 MB/s ingest, 150 s of unused budget). The 716 J energy is real but the *time* axis was idle; a faster ingest path or larger pre-loaded corpus would meaningfully change the data-seen-per-budget figure even at D=24.
- **Bytewise CTW (Sadakane-Okazaki-Imai) was named in the writeup as the right next step** and not run. This is the only credible CTW variant that can plausibly clear 0.70 within budget; not testing it leaves the "is CTW competitive on this benchmark" question open.

**Verdict:** Discards D=24 bit-CTW only. The cheap (sub-1 kJ) infra is a positive standalone finding — any CPU-only / NVML-quiet method now has a working harness path. Bytewise CTW remains the live follow-up.
