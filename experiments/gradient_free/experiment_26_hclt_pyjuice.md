# Experiment 22: Hidden Chow-Liu Tree (HCLT) Probabilistic Circuit via PyJuice

## Hypothesis
A Hidden Chow-Liu Tree (HCLT; Liu, Van den Broeck 2021) probabilistic circuit, with structure learned from training data via mutual-information-driven tree extraction and parameters learned by EM via PyJuice, clears val char-acc 0.50 within 300 s — strictly above RAT-SPN (experiment_25) because HCLT structure is *data-informed* rather than random. Liu 2024 reports HCLT outperforming RAT-SPN by 5–10% NLL on tabular and image PCs. Translated to byte LM with K=16 window, this should give comparable lift. **Expected val char-acc 0.50–0.60**, ceiling around 0.65, plausibly below the 0.70 floor but a strong PC baseline.

## Motivation
RAT-SPN (experiment_25) uses random region splits — the structure is uninformed by data. HCLT learns the tree structure from mutual information between variable pairs, then converts the Chow-Liu tree into a PC by introducing latent variables at each internal tree node. This is **the strongest off-the-shelf PC structure in PyJuice 2024**. For byte-level text, the natural CLT structure should capture short-range byte dependencies (consonant-vowel pairs, common bigrams) directly in the tree topology before any parameter learning happens — analogous to how MERA's disentangler layer (experiment_24) captures pairwise structure before the tree. HCLT is the PC-family counterpart.

## Method
**Structure learning** (one-shot, no backprop):
1. From a 100K-window sample, compute pairwise mutual information MI(c_i, c_j) for all i, j ∈ [0, K-1].
2. Build the maximum-MI spanning tree (Chow & Liu 1968). For K=16, the MST has 15 edges.
3. Convert to PC: introduce latent variable Z_e per tree edge with hidden_size=H=32; each leaf is a categorical(V=256); each edge factor is a P(Z_e | Z_parent_e) · P(leaf_child | Z_e) factor.
4. PC is then a tree-structured PC with H · K parameters at categorical layers + H² · (K-1) at inter-edge transition factors.

**Parameter learning** (EM via PyJuice):
- 3 epochs of mini-batch EM, batch 256, lr=0.1, over 1 M windows.

**AR inference**: same as experiment_25 — last byte unmarginalized, evaluate marginal P(c_K | c_<K) by a single PyJuice forward pass with marginalization flag.

## Memory-Movement Analysis
- **Parameter count**: H² · 15 + H · 16 · V = 32² · 15 + 32 · 16 · 256 = 15K + 131K = 146K parameters. Smaller than RAT-SPN, faster per-step.
- **Per-window forward FLOPs**: ~150K FLOPs/window (one pass through ~150K-param network at PyJuice's typical efficiency).
- **Training**: 1 M windows · 150K FLOPs · 3 epochs = 4.5 · 10¹¹ FLOPs. At PyJuice's 50 GFLOPs effective: 10 s. **Plenty of headroom for more epochs or bigger H.**
- **MI computation**: 16 · 16 / 2 = 120 pairs; each pair MI requires histogram over (V × V) = 65K bins, accumulated over 100K windows · K = 1.6 M observations per pair. Fast: ~5 s on CPU; ~1 s on GPU.
- **Memory**: tiny. ~0.6 MB params.
- **Eval**: 60K chars · 150K FLOPs / char = 9 GFLOPs ≈ 0.2 s. Free.
- **Arithmetic intensity**: still PyJuice's ~5 FLOPs/byte (bandwidth-bound) but the network is so small that the bandwidth bound doesn't matter — total training is 10 s regardless.

## Setup
- K = 16-byte input window, V = 256, hidden_size H = 32 per edge.
- MI estimation: 100K windows sampled from train; histograms in fp32.
- Training: 3 EM epochs over 1 M windows, batch B=256, PyJuice's PCOptimizer with lr=0.1.
- All fp32 (PyJuice default).
- Compare against: experiment_25 (RAT-SPN); `ctw_d24` (0.475).

## Procedure
1. `mkdir submissions/hclt_k16_pyjuice`. `pip install pyjuice`.
2. Implement `compute_pairwise_mi(text, K, sample_size)` returning a (K, K) MI matrix.
3. Build CLT: `chow_liu_tree(mi_matrix)` returns a list of (parent, child) edges (use `scipy.sparse.csgraph.minimum_spanning_tree` on -MI; convert to undirected tree rooted at 0).
4. Implement `build_hclt(tree, H, V)` returning a PyJuice PC object: walk the tree, instantiate hidden variables at each edge, wire categorical leaves at each node.
5. EM training via `pyjuice.optim.PCOptimizer`.
6. `CharModel`: ring buffer of K=16 bytes; `predict()` calls forward with last-position marginalization.
7. `python submit.py submissions/hclt_k16_pyjuice --yes`.

## Success Criteria
- **Primary**: val char-acc ≥ experiment_25 (RAT-SPN). Validates data-informed > random structure.
- **Strong**: val char-acc ≥ 0.55, energy ≤ 20 kJ.
- **Surprise**: val char-acc ≥ 0.70 — clears floor. First HCLT byte-LM result.
- **Refutation**: val char-acc ≤ experiment_25 — MI-driven tree structure does not help over random for byte sequences (perhaps because byte adjacencies are already the strongest MI, and a CLT trivially links them, which is what RAT-SPN's random splits also approximately get).

## Failure Modes & Diagnostics
- **MI computation underflows / overflows for rare bytes**: add Laplace smoothing α=1 to bigram counts before MI calc.
- **CLT tree is degenerate** (chain rather than balanced tree): expected — most MI is between adjacent positions, so the MST is approximately a chain c_0–c_1–c_2–…–c_15. This makes HCLT structurally similar to an HMM with H=32 states. Diagnostic: print tree structure; if all edges are (i, i+1), report this as the result.
- **PyJuice HCLT factory may not exist**: PyJuice has HCLT support per the 2024 paper but may require manual construction if the high-level API differs. Fall back to PyJuice's `RegionGraph` API and manually wire a CLT-shaped region graph.
- **Hidden_size H=32 too small for V=256 leaves**: try H=64 (params 4×, training time ~40 s, still fits).
- **EM plateaus at 1 epoch**: typical for small PCs; not an error.

## Estimated Cost
1 Modal A100-80GB run × ~3 min wall ≈ $0.06. Variant H=64: +$0.06. Variant K=32: +$0.10.

## References
- Liu, Van den Broeck 2021, "Tractable Regularization of Probabilistic Circuits", NeurIPS — HCLT introduction.
- Chow & Liu 1968, "Approximating discrete probability distributions with dependence trees", IEEE Trans. Info Theory — original CLT.
- Liu, Peharz, Van den Broeck 2024, "Scaling Tractable Probabilistic Circuits: A Systems Perspective", ICML / arXiv:2406.00766 — PyJuice & HCLT in modern PC pipeline.
- Choi, Vergari, Van den Broeck 2024, "Building Expressive and Tractable Probabilistic Generative Models: A Review", arXiv:2402.00759.
- Companion: experiment_25 (RAT-SPN); `research/non_nn_methods/spec_06_sum_product_network_lm.md`.
