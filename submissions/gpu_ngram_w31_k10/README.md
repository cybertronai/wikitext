# gpu_ngram_w31_k10

Plain chained-KN W31 with `MAX_ORDER = 10`. Floor probe — tests whether dropping one more order keeps J under W31_K11's 1,245 J while staying above acc 0.70.

**Hypothesis:** lands ~900 J. Acc risk: 0.68-0.70 (might DQ).

**Why:** W31_K11 confirmed J = 1,245 / acc 0.7050 with margin 0.50pp. K=10 tests if floor margin survives further depth reduction.
