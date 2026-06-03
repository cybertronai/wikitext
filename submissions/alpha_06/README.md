# alpha_06 — Clean hybrid α=0.60 sweep

**Paradigm:** Hybrid NN + W31 GPU KN n-gram at α=0.60 (NN weight).

## Mechanism

Identical to `alpha_065` (NN d=256 L=4 + W31 GPU KN n-gram) except
ALPHA = 0.60 (was 0.65). More weight to the n-gram, less to NN.

## Hypothesis

α sweep history (clean hybrids):
- α=0.5 — 0.7063 (E3 / nano_plus_ngram)
- α=0.65 — 0.7387 / 0.7407 (alpha_065 current best clean acc)
- α=0.7 — 0.7324 / 0.7332 (clean_hybrid_w31, alpha_07_deep)
- α=0.8 — 0.7225 (clean_hybrid_a08)

Concave-up curve through α=0.5..0.7. Testing α=0.6 to bracket whether
sweet spot is at α=0.65 or shifted lower.

## Expected

- Energy: 14-16 kJ (same compute as alpha_065)
- Accuracy: 0.73-0.74 (likely close to alpha_065's 0.7407, possibly higher)
- L2-clean: yes (alpha_065 lineage is fully GPU-active)

## Smoke test

PASS on `fixtures/tiny/`.
