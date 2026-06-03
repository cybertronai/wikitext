# subset_70_mkn

**Paradigm:** Winners-stack: 70% data subset + MKN smoothing. Iter-2 exp 4/10.

**Mechanism:** Bit-for-bit `pitman_yor_k11` (MKN) but trained on first 70% of WikiText-103.

**Hypothesis:** subset_70 lifted J from 1,245 → 781 with -0.0033pp acc. MKN lifted acc from 0.7050 → 0.7066 (+0.0016) at lower J. Stacking should give:
- J: ~70% × 1,146 ≈ 800 J
- Acc: ~0.7017 + 0.0016 (MKN lift) ≈ 0.7033

**Expected J:** 750-900 J.
**Expected acc:** 0.7025-0.7050. **Crucially, more margin to floor than subset_70 alone.**

**Information value:** if both effects compose, this is the J leader at safer floor margin. If MKN's lift doesn't compose with subset, we learn the subset crunches MKN's count-of-counts statistics.
