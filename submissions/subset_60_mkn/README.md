# subset_60_mkn

**Paradigm:** Winners-stack variant — push the data subset further (70% → 60%) on top of MKN smoothing.

**Mechanism:** Bit-for-bit `subset_70_mkn` (Modified Kneser-Ney, order 11) but trained on the
first **60%** of WikiText-103 instead of 70%. Single knob changed: `SUBSET_FRAC` default 0.7 → 0.6.

**Hypothesis:** subset_70_mkn lands at **2,866 J / 0.7031 acc** — only +0.31pp above the 0.70 floor.
The 1.0 → 0.7 cut cost just -0.0033 acc, so a further 0.7 → 0.6 cut (~14% fewer train bytes) should
cost a similar small amount of accuracy while shaving energy. Because energy splits ~1,321 J GPU +
1,545 J CPU, less data cuts both the GPU count/sort and the CPU-side table build.

- **Expected J:** ~2,300–2,600 J (less data → cheaper sort/dedup + cheaper CPU table build).
- **Expected acc:** 0.7010–0.7025 — **margin to floor is the risk.** If it dips below 0.70 it DQs.

**Information value:** If acc holds ≥0.70, this becomes the new J leader and tells us the subset
curve is still favorable below 70%. If it DQs, we learn 70% is near the accuracy cliff for MKN-11
and the next lever must be CPU-side build cost, not data volume.

**Status:** Not yet run on the pinned Modal A100. Run with `python submit.py submissions/subset_60_mkn`.
