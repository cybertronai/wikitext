# Experiment 21: RAT-SPN via PyJuice on K=16 Byte Window (EM Training)

## Hypothesis
A Region-Adaptive Tensorized Sum-Product Network (RAT-SPN; Peharz 2020) with K=16-byte input window, depth=6, 4 sum layers, 3 replicas, categorical leaves over V=256, trained by 3 epochs of online EM via PyJuice (Liu, Peharz, Van den Broeck 2024) on 1 M windows, clears val char-acc 0.40–0.55 within the 300 s budget. Cheng et al. (Interspeech 2014) reached PTB ppl 93 on word-level LM with K=4 and a 40× training corpus; modern PyJuice infrastructure (1–2 OOM faster than 2014 SPNs) lets us go to K=16 and meaningful corpus sizes in 300 s. The expected ceiling is below 0.70 — SPNs are bandwidth-bound on the GPU and have limited expressivity at the practical structure sizes — but the result is the **first PyJuice byte-LM number anywhere**, useful as a capability demo.

## Motivation
SPNs / probabilistic circuits are the canonical "tractable inference + EM training" family — fully gradient-free, exactly normalized, with linear-time conditional inference. No SPN submission exists in this repo; the nearest neighbor is `ctw_d24` (Context-Tree Weighting, related tractable PC, DQ at 0.475). PyJuice's 2024 systems contribution makes large-scale PC training GPU-practical for the first time. Even if the result lands at 0.45, it informs the joules-per-acc frontier for the bandwidth-bound corner of the gradient-free landscape and provides a strong data point against the claim that "probabilistic circuits are the right path for LM."

## Method
**Structure**: RAT-SPN region graph with byte window of size K=16. Region splits: at each depth, partition the byte set into two random subsets; replicas R=3 give multiple structures averaged at root. Depth D=6 → 2^6 = 64 leaf regions per replica. Sum layers: 4 layers of sum nodes with width 32 each → ~2K sum nodes per replica, ~6K total. Total parameters ~5 M.

**Leaves**: 64 categorical leaves per replica × 3 replicas = 192 leaves, each over V=256 bytes. Parameters: 192 · 256 · 3 (replicas) = wait, leaves are per-replica → 192 · 256 = 49K leaf parameters.

**Training**: online EM via PyJuice's `pyjuice.optim.PCOptimizer` with `lr=0.1`, batch_size=256, 3 epochs over 1 M windows extracted from WikiText-103.

**AR inference for char prediction**: at eval time, window is the last K=16 observed bytes plus 1 unknown next byte (the one we're predicting). To get P(c_{t+1} = v | c_{t-15}..c_t):
- Set the first 15 leaves to observed values c_{t-15}..c_{t-1} (single-position evidence).
- Marginalize the 16th leaf (the predicted position).
- For each candidate next-byte v, compute P(window with last leaf = v); normalize over v.

Cost: one batched forward pass with the (V, 1) sweep over the last leaf — PyJuice supports this natively as marginal evaluation.

## Memory-Movement Analysis
- **Per-window forward FLOPs**: ~5 M ops (5 M parameters, each visited once) ≈ 5 M FLOPs. PyJuice reports ~5 FLOPs/byte arithmetic intensity (Liu 2024 §5) — **strongly bandwidth-bound** (compared to TN methods at 100+ FLOPs/byte).
- **Training**: 1 M windows · 5 M FLOPs · 3 epochs = 1.5 · 10¹³ FLOPs. At PyJuice's typical A100 throughput (~50 GFLOPs effective due to bandwidth bound, per Liu 2024 §6) → 300 s. **Right at the budget.** Mitigation: drop to 2 epochs or 500K windows.
- **Memory**: 5 M params · 4 B = 20 MB. Activations per batch: B · network_size · 4 = 256 · 6K · 4 = 6 MB. **Tiny.**
- **Eval**: 60K chars · 256 candidates · forward_pass · 5 M ops = 7.7 · 10¹³ FLOPs ≈ 1500 s at PyJuice's 50 GFLOPs effective. **DOES NOT FIT** in eval budget. Mitigation:
  - PyJuice's marginal evaluation returns the entire V-dim distribution in one forward pass (the last-leaf marginal). Cost: one forward = 5 M FLOPs / char. 60K chars · 5 M = 3 · 10¹¹ FLOPs ≈ 6 s. **Fits.**
- **Bandwidth bound**: every node value is a random-access lookup. PyJuice's CSR-style layout amortizes this but still loses to dense GEMM by 10×. Realistically achieves 5–15% A100 peak.

## Setup
- PyJuice version ≥ 0.0.5 (current as of May 2026). `pip install pyjuice`.
- RAT-SPN: K=16 input vars, depth=6, num_sum_layers=4, replicas=3, sum_width=32, categorical leaves over V=256.
- Training: 3 epochs over 1 M windows from WikiText-103 train, batch B=256, EM with lr=0.1.
- Window construction: sliding-window over the train text with stride 1; 1 M windows ≈ 16 MB of underlying text.
- All fp32 (PyJuice's default).
- Compare against: `ctw_d24` (0.475 / 0.7 kJ); experiment_26 (HCLT, pending).

## Procedure
1. `mkdir submissions/ratspn_k16_pyjuice`.
2. Add `pip install pyjuice` to submission's runtime requirements. Verify Modal install latency.
3. Implement `build_ratspn(K, depth, num_sum_layers, replicas, num_categories, sum_width)` using PyJuice's region-graph constructor.
4. Implement `train_pyjuice(spn, train_text, n_windows, batch_size, epochs)`: stride-1 windowing, calls to `pyjuice.optim.PCOptimizer.step()`.
5. `CharModel`: maintain a ring buffer of last K=16 bytes; `predict()` calls `spn.forward(buffer, marginalize_last=True)` to get a (256,) distribution; `observe(c)` rolls the buffer.
6. `python submit.py submissions/ratspn_k16_pyjuice --yes`.

## Success Criteria
- **Capability**: val char-acc ≥ 0.45 — strictly above CTW's 0.475 is not required (CTW had different alphabet handling); ≥ 0.45 is the honest target.
- **Strong**: val char-acc ≥ 0.55, energy ≤ 40 kJ.
- **Surprise**: val char-acc ≥ 0.70 — clears floor. First PyJuice byte-LM result.
- **Refutation**: val char-acc ≤ 0.40 — RAT-SPN structure with K=16 cannot carry English n-gram statistics; either depth or window size is too small. SPN family disqualified for byte LM at this compute budget.

## Failure Modes & Diagnostics
- **PyJuice install fails on Modal**: pin to a known-good version (0.0.5); fall back to manual SPN implementation if necessary.
- **Bandwidth bound dominates wall-clock**: budget cushion limited. If 3 epochs exceed 300 s, drop to 2 or to 500K windows.
- **EM converges in 1 epoch**: log train log-likelihood per epoch. If flat after iter 1, structure is too small; increase `sum_width` to 64.
- **Per-char inference > 1 ms**: 60K chars × 1 ms = 60 s eval, still fits but tight; if PyJuice's marginal eval is >5 ms/char, batch eval is needed (PyJuice supports batched forward).
- **Categorical leaves with V=256 are non-standard for PyJuice**: most PyJuice examples use V=2 (binary) or V=10 (digits). Verify the categorical-leaf factory accepts V=256 by running a 100-window pre-flight before the main training.
- **Region graph generation is non-deterministic** (random splits): seed the random partitioning and log the resulting structure for reproducibility.

## Estimated Cost
1 Modal A100-80GB run × ~5 min wall ≈ $0.10. A K=32 window variant would scale to ~10 M parameters and likely exceed the wall-clock — keep for v2. A depth=8 variant: +$0.10.

## References
- Liu, Peharz, Van den Broeck 2024, "Scaling Tractable Probabilistic Circuits: A Systems Perspective", ICML / arXiv:2406.00766 — PyJuice paper.
- Peharz, Lang, Vergari, Stelzner, Molina, Trapp, Van den Broeck, Kersting, Ghahramani 2020, "Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits", ICML / arXiv:2004.06231 — RAT-SPN architecture.
- Cheng, Kok, Pham, Chieu, Chai 2014, "Language Modeling with Sum-Product Networks", Interspeech — original SPN-LM (word-level, K=4).
- Poon & Domingos 2011, "Sum-Product Networks: A New Deep Architecture", UAI / arXiv:1202.3732.
- `research/non_nn_methods/spec_06_sum_product_network_lm.md` — research-tier spec this operationalizes.
- `submissions/ctw_d24/result.json` — closest tractable PC baseline (0.475 acc).
