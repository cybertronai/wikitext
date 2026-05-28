# Experiment 05: Rerun Nyström-KRR Hybrid with Train-Then-Discard-Head Protocol (Diagnoses Exp 07's Failure)

## Hypothesis
`nystrom_krr_hybrid` (prior exp 07, DQ'd at 0.300 acc / 32 kJ) failed not because deep kernel learning is dead at this scale, but because the spec's required protocol — "train transformer + linear head end-to-end first, *then* discard the head and replace with Nyström-KRR readout" — was skipped, going straight to kernel-readout-from-random-init (which collapses into paradigm-A and the 0.37 ceiling we now know is structural). Correctly executing the two-stage protocol should reach ≥0.70 val acc.

## Motivation
This is a **claim-verification experiment** for the previous portfolio. The deep-kernel-learning hypothesis (Wilson 2016, Bradshaw 2017) is: SGD-trained representation + closed-form kernel readout > either alone. Our existing empirical data refutes "kernel-readout on random features" (the paradigm-A ceiling). It does NOT yet refute deep-kernel-learning, because the one submission that tested it skipped the SGD pretraining stage.

If this fails too, deep kernel learning is empirically refuted at our scale. If it passes, it's a new paradigm-B win — frozen SGD-trained encoder + gradient-free closed-form readout.

Builds directly on `experiments/kernel_methods/experiment_07_falkon_krr_learned_embedding.md` — explicitly re-running that spec's headlined protocol that the executed submission skipped.

## Method
**Stage 1 (SGD, ~200 s of budget):** train the full modded_nanogpt as on the leaderboard — 4-layer transformer + linear head + AdamW/Muon — for ~1500 steps. Cross-entropy loss as usual.

**Stage 2 (closed-form, ~30 s of budget):** discard the linear head. Forward-pass a fixed-size random subset of the train set through the frozen encoder to collect (h_i, y_i) pairs where h_i ∈ R^d is the encoder's hidden state at the next-byte-prediction position and y_i ∈ {0,…,255}.

Fit a Nyström-approximated kernel ridge regression: choose m=512 landmark points (random subset of (h_i)), build the m×m kernel matrix K_mm with the dot-product kernel k(x,y) = (1 + x·y/d)², compute the closed-form readout
W = (K_mm + λI)⁻¹ K_mn Y_n   (size m × 256)
where K_mn is m × n_samples and Y_n is n_samples × 256 (one-hot targets).

**Inference:** for each position, compute the same encoder forward + k(h, landmarks_m) · W → 256-vector of scores → softmax (with a learned-or-fixed temperature) for the streaming `predict()` API.

## Memory-Movement Analysis
- Stage 1 is just normal SGD training (well-understood compute-bound).
- Stage 2: building K_mn with n_samples=50000, m=512: 25M kernel evals, each is one dot product in d=384 → 9.6 GFLOPs total, milliseconds on A100.
- The (K_mm + λI)⁻¹ K_mn is a (512, 512) solve + (512, 512) × (512, 50000) matmul + (512, 50000) × (50000, 256) matmul = ~13 GFLOPs total. Trivial.
- Stage-2 energy is negligible (<200 J).
- At inference: per-byte cost is encoder forward + dot product against 512 landmarks + one matmul against (512, 256) = ~6× cheaper than the full softmax head. **Actually faster than the baseline at inference time, not slower.**

## Setup
- Stage-1 model: 4-layer modded_nanogpt body + linear head, trained as in `modded_nanogpt/submission.py`.
- Stage-2 readout: Nyström KRR with m=512 landmarks, kernel = dot-product polynomial deg 2, λ swept over {1e-3, 1e-2, 1e-1, 1}.
- Total wall: stage 1 capped at 200 s, stage 2 + λ-sweep at <60 s.
- Baseline: `modded_nanogpt` (6L, 51.7 kJ / 0.7374) and `hopfield_layer` (40.2 kJ / 0.7293). Also the prior `nystrom_krr_hybrid` failure (0.300 acc).

## Procedure
1. `cp -r submissions/modded_nanogpt submissions/nystrom_correct`
2. Reduce `num_layers=4` and `n_steps=1500` to leave wall-clock budget for stage 2.
3. After training loop completes, before returning the CharModel:
   a. Forward-pass 50000 random train positions through the encoder; collect (h_i, y_i).
   b. Pick 512 of those (h_i) as landmarks Z.
   c. Compute K_mm = (1 + Z @ Z.T / d) ** 2, K_mn = (1 + Z @ H.T / d) ** 2.
   d. Solve W = torch.linalg.solve(K_mm + λI, K_mn @ Y_onehot) — 4× λ values, hold out 5000 positions as a tiny λ-tune set, pick best λ.
4. Replace the `predict()` method on the CharModel: instead of `softmax(proj @ h)`, compute `softmax(((1 + Z @ h / d) ** 2) @ W)`.
5. Submit.

## Success Criteria
- **Strong**: val ≥ 0.73, energy ≤ 45 kJ → deep-kernel-learning is alive and competitive with the SGD head it replaces.
- **Pass**: val ≥ 0.70, any energy < 51 kJ → refutes the prior nystrom_krr_hybrid failure as a protocol bug.
- **Refutation (interesting)**: val < 0.70 with stage-1-trained encoder → deep-kernel-learning genuinely doesn't help; closed-form readout cannot match a 50-step-fine-tuned linear head. Adds a new entry to the dead list.

## Failure Modes & Diagnostics
- Stage-1 wall overruns: hard-cap at 200 s via internal time check; if exceeded, do a shortened stage-2 anyway.
- K_mm is ill-conditioned: add λI ≥ 1e-3 always; verify cond(K_mm) < 1e8.
- Encoder hidden state is bf16 — cast to fp32 for kernel arithmetic. K_mm in fp32 (1 MB), fine.
- Numerical mismatch between train and eval streams: verify by feeding 100 train positions back through the new `predict()` and confirming top-1 accuracy on those is ≥0.8 (sanity check that stage 2 fit).

## Estimated Cost
1 Modal run, ~10 min, ~$0.40.

## References
- `experiments/kernel_methods/experiment_07_falkon_krr_learned_embedding.md` — the original spec with the protocol this rerun honours
- Wilson et al. 2016 "Deep Kernel Learning" (AISTATS)
- Bradshaw et al. 2017 "Adversarial Examples, Uncertainty, and Transfer Testing Robustness in Gaussian Process Hybrid Deep Networks" — DKL applied to a deep net.
- Williams & Seeger 2001 "Using the Nyström method to speed up kernel machines"
