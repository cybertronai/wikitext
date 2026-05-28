# Experiment N4 — PAQ8-scale context mixing (paq8_gpu)

## Hypothesis
`paq_mixer_v3` is at 3.58 kJ / 0.7048 with **7 + bias features** over **11
standard byte n-gram orders** and an independent-table (Witten-Bell)
backbone. PAQ8/CMIX reach ~0.12 bpc on enwik8 via **two compounding
moves**: (a) replace the WB-independent backbone with a **KN-interpolated
base** (the proven path that took `gpu_ngram_o14_xorfix` from 0.7048 →
0.7184); (b) add **orthogonal context families** beyond standard byte
n-grams — sparse skip contexts, case-folded contexts, and a match-model.
The mixer becomes a tiny logistic over per-model log-probs + features.

## Concrete design
**Base distribution (always present)**: KN-interpolated chained backoff
at K=12 on GPU (reuse `gpu_ngram_w31_k11`'s GPU build path, extended one
order). ~1.5 s GPU, ~150 J.

**PAQ8 model families** added on top of the KN base:
1. **Standard byte n-gram orders 1..11** (already used by paq_mixer_v3).
2. **Skip-1 sparse contexts** at length-5: `b[-6,-5,-4,-3,-2]` → `b[0]`
   (gap of 1). Built on GPU via the same `_build_top_order_gpu` pipeline
   but with the rightmost byte being position 0 not position -1. Cheap
   because it's a one-shot order-6 build.
3. **Case-folded order-7**: lowercase the byte stream, then build
   order-7 byte n-grams over that. Captures word-stem regularities
   independent of casing.
4. **Word-context order**: hash on (last 8 bytes including whitespace
   boundary). Standard order-7 reuse — skip; instead reuse standard
   order-8 since it's already cheap.
5. **Match model (online)**: at predict time, search the last 8 KB of
   history for the longest exact suffix match of length ≥4; predict the
   byte that followed it. Implemented via tiny rolling-hash table.

**Mixer**: 2-layer MLP, input = [features × K_models, per-model
log-prob-at-each-byte (gathered at training-targets)], hidden=32, output
= K_models softmax weights. ~1.5K params. Trained on 200K positions from
a 2 MB held-out tail with Adam 1500 steps. Loss = mean -log(sum_k w_k *
p_k[target]).

**Predict**: KN base + each PAQ8 family contributes a 256-vec; mixer
weights determine convex combination. Argmax.

## Engineering plan
- Tables built top-down on GPU; total build < 90 s based on paq_mixer_v3
  timing.
- Per-position predict cost is now (K=11 + 3 PAQ8 families + 1 match
  model) ≈ 15 lookups vs paq_mixer's 11; expect ~1800 char/s eval (eval
  is not energy-scored).
- Train phase target: **3-5 kJ / ≤ 200 s**.

## Pareto target
Sub-5-kJ / ≥ 0.72 displaces nothing in the leaderboard but cements the
mid-PAQ corner. Stretch target ≥ 0.74 / < 10 kJ would displace
`alpha_06`. Likely outcome is **3-5 kJ / 0.715-0.725** — a substantive
Pareto extension of `paq_mixer_v3`.
