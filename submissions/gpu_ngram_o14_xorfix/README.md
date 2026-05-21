# gpu_ngram_o14_xorfix — GPU order-14 KN with XOR sign-bit-fix sort

**Paradigm:** GPU port of W3/W31 chained KN backoff with a sign-bit-safe GPU sort.

## Mechanism

Identical to `gpu_ngram_o14` but replaces the slow CPU re-sort (~150s
on Modal) with an in-place GPU sign-bit-XOR trick:

```python
sort_lo = lo ^ (1 << 63)   # flip sign bit on a sort-key copy
sort_hi = hi ^ (1 << 63)
order = torch.argsort(sort_lo, stable=True)  # signed sort → unsigned lex
...
```

This produces unsigned lex order directly from `torch.sort`'s signed
comparator, no CPU pass needed. The original `(hi, lo)` byte payloads
ride along through the same permutation; the `_gpu_table_to_w3_layout`
function reads them un-XORed when decoding to bytes.

## Hypothesis

- `gpu_ngram_o14` (CPU re-sort): 5,143 J / 0.7184 (acc clean)
- `gpu_ngram_w3` (W31, buggy sort, order-12): 1,847 J / 0.7114
- Target: 1.5-2.5 kJ / 0.7184 — best of both worlds

Eliminating the 150s CPU re-sort phase should drop energy ~3× while
keeping accuracy at the W3-CPU level (0.7184).

## Expected

- Energy: 1.5-2.5 kJ
- Accuracy: 0.7184 (matching W3 CPU + O14 GPU)
- L2-clean: yes (GPU active throughout build)

## Smoke test

PASS on `fixtures/tiny/` (485 bytes → max_order clamped).
