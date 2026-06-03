# gpu_ngram_w31_k11

W31 (gpu_ngram_w3) at MAX_ORDER=11 instead of 12.

## Why

Test the PAQ subagent's honest follow-up prediction. paq_mixer_v3 landed
at 1,744 J / 0.7047 / PCIe, but the J win decomposed as:
  - (a) dropping MAX_ORDER 12→11 (skipped expensive top-order materialise),
  - (b) lucky PCIe SKU.

Their prediction: pure W31 at order-11 would land similar J with HIGHER
acc (~0.71 vs paq_v3's 0.7047), because chained-KN is more J-efficient
than PAQ per-order mixing (paq paid +29% J for +0.07pp acc at iso-K).

## Change from W31 (gpu_ngram_w3)

ONE line: `MAX_ORDER = 12` -> `MAX_ORDER = 11`. Nothing else.

## Expected

- J: ~1,700-1,850 (paq_v3 zone, since paq_v3's J win came largely from the K=12->K=11 step)
- Acc: ~0.71 (between W31 K=12 0.7114 and paq_v3 K=11 0.7047)
- GPU: random PCIe vs SXM4 (Modal can't pin)

## Author

`@follow-up-paq-prediction`
