# paq8_gpu

PAQ8-scale context mixing (experiment N4 from research/gradfree_analysis.md).

## What this is

KN-interpolated chained-backoff base at order-11 + three orthogonal
PAQ8-family context models, all mixed by a tiny logistic mixer.

Families:
1. **KN base** (orders 1..11, chained backoff): the standard byte n-gram.
2. **Skip-1 sparse context**: predict `b[0]` from `b[-6..-2]` with a
   gap at position `-1`. Captures word-stem regularities through
   whitespace.
3. **Case-folded order-7**: build order-7 n-grams over the lowercased
   byte stream. Captures stems independent of capitalisation.
4. **Match model**: at predict time, scan the last 2 KB of stream for
   the longest exact suffix match (>= 4 bytes); predict the byte that
   followed it with high mass.

## Mixer

2-layer MLP, 14 features → hidden 32 → 4 family weights via softmax.
Trained on 80K held-out positions (last 2 MB of train) with Adam,
1500 steps, log-sum-exp loss against the true next byte. ~2K params.

## Expected

- Build: ~150-200 s on A100.
- Energy: ~4-6 kJ.
- Val char-acc: 0.715-0.725 (likely below alpha_06's 0.7405).

## Author

`@worker-paq8-gpu`
