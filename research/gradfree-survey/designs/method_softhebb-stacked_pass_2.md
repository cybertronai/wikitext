# Shallow Wide Hebbian Patch Bank + Ridge (pass 2)

Direction chosen: **B** — drop the deep stack entirely. One Hebbian layer at very large width, with the closed-form ridge doing the actual prediction work. Pass-1 failure was specifically diagnosed as channels-not-differentiating and Gram rank-deficient through a 4-layer stack; the principled response is to (a) remove all but one Hebbian layer so failure modes don't compound, (b) make that layer wide enough that even a low-utilization fraction of channels still gives the ridge enough rank to fit, and (c) compare against an explicit random-projection control inside the same harness so we can isolate the contribution of the Hebbian rule itself.

## 1. Hypothesis

The depth in pass 1 was actively hurting (each layer's WTA collapse compounded). A single Hebbian patch-bank layer at width 8192, treated as a feature extractor for a closed-form ridge readout, should at minimum match a random-projection baseline of the same width, and at best modestly exceed it. We will learn: (i) whether the SoftHebb soft-WTA rule contributes *any* discriminative information over a random Gaussian projection on byte text, and (ii) whether the per-channel entropy pathology of pass 1 was a depth artifact or an intrinsic property of the rule on bytes. A positive delta of Hebbian over random — even a tiny one — is the interesting finding regardless of absolute accuracy.

## 2. Model

- Input: one-hot byte tensor in R^{256}, then a fixed flat patch view of the last K=16 bytes => raw feature vector x in R^{256*16=4096} per timestep. (One-hot avoids byte-value-as-scalar artifacts.)
- One Hebbian layer: W in R^{H x 4096}, H = 8192. Output u = W @ x. Soft-WTA activation y = softmax(u / tau) over the H units.
- Readout: linear W_out in R^{256 x H} + bias, fit by ridge to next-byte one-hot. No nonlinearity, no second layer, no concatenation across positions.
- Side-by-side control: identical pipeline with W frozen to a Gaussian random matrix (std = 1/sqrt(4096)) — same H, same tau, same ridge. Run both within the 300 s budget; report both.

## 3. Training procedure

1. Encode train_text to uint8 on GPU; take the first 200M chars (more than pass 1 since we have only one Hebbian layer to train).
2. Hebbian training of W (SoftHebb rule, fp32 weights):
   - Stream non-overlapping windows of length 4096 bytes, batch B=64 (262144 chars/step, ~763 steps over 200M).
   - For each step: build x via one-hot + length-16 sliding patches => tensor of shape (B*T, 4096) with ~16M rows per step after stride 1; subsample stride 4 to keep ~4M rows/step.
   - u = x @ W^T; y = softmax(u / tau, dim=channel).
   - SoftHebb update: dW = eta * (y^T @ (x - (y @ W))) / N_rows. Apply once per step. Anneal eta linearly to 0 across the sweep.
   - Every 50 steps, renormalize rows of W to unit L2 norm (cheap insurance against drift; pass 1 noted this as a fallback, here we enable it from the start).
   - Track per-channel mean activation; if at step 100 the channel-usage entropy is < 0.3 * log(H), raise tau by 1.5x once and continue (one adjustment, no full retry — we want signal, not a tuned win).
3. Feature collection: stream 30M chars through W, build phi(t) in R^H, subsample stride 8 -> N ~= 3.75M rows.
4. Closed-form ridge: chunked Gram Phi^T Phi (H=8192 => Gram is 8192x8192 = 256 MB fp32, fits easily). Cholesky solve for all 256 outputs simultaneously. Lambda = 1e-2 * trace(Gram)/H.
5. Repeat steps 3-4 for the random-W control (same code path, W replaced by frozen Gaussian).
6. CharModel: ring buffer of last 16 bytes; observe(c) shifts; predict() computes one matmul (8192x4096) + softmax + ridge head.

## 4. Hyperparameters

- H = 8192; patch length K = 16; input dim = 4096.
- tau = 0.5 (lower than pass 1's 1.0; with H=8192 we need sharper competition for the WTA to break symmetry).
- eta = 0.01, linearly annealed to 0 over the sweep.
- Hebbian sweep: 200M chars, batch 64 x 4096, stride-4 subsample of training patches.
- Ridge: N ~= 3.75M, H = 8192, lambda = 1e-2 * trace/H.
- Row-renormalize W every 50 steps.
- One tau bump allowed (1.5x) if entropy collapse detected at step 100.
- SEED honored for Gaussian control init and for subsampling.

## 5. Expected wall time (A100-80GB)

- Hebbian sweep: 763 steps * (matmul 4M x 4096 x 8192 in bf16 ~ 1.3e14 ops/step / 200 TFLOP/s ~ 0.65 s) ~ 500 s of compute -> too much. Mitigation: cut subsample to stride 8 (drops to ~250 s) and batch 32 (drops to ~125 s). Locked plan: batch 32, stride 8, 200M chars => ~120 s Hebbian.
- Feature collection: ~15 s.
- Gram (3.75M x 8192 chunked) + Cholesky: ~20 s.
- Random control (no training, just collection + ridge): ~35 s.
- Overhead: ~20 s.
- Total: ~210 s, inside the 300 s budget.

## 6. Success criterion

Honest: I do not expect either configuration to reach 0.70. Realistic outcomes:
- val_char_acc(Hebbian) in [0.20, 0.40], val_char_acc(random) in [0.18, 0.38], delta positive => weak but real signal that SoftHebb learns *something* on bytes.
- val_char_acc(Hebbian) ~= val_char_acc(random) => the family is unsuited and pass 1's collapse was the rule, not a tuning failure.
- Stretch: 0.55 for Hebbian (would still beat pass 1's 0.119 by 4.6x and constitute a meaningful finding).
- Energy: < 6 kJ for the combined run (no optimizer state, single sweep, one matmul per step).

The interesting datum is the **delta**, not the absolute number.

## 7. Failure modes anticipated

- The entire local-Hebbian-on-bytes family may be unsuited to LM: bytes lack the smooth local statistics images have, so Hebbian competition has nothing geometric to latch onto. This pass is designed to confirm or deny that with one clean measurement.
- WTA collapse at H=8192: more channels means more room for dead units. Mitigation: row renorm + one tau bump.
- Ridge rank deficiency at H=8192: Gram may still be ill-conditioned; lambda scaling and Cholesky-with-jitter fallback (add 1e-6 * trace/H to diag if Cholesky fails) handle this.
- Random baseline beats Hebbian: a clean negative result; we report it honestly.
- 16-byte receptive field is short vs. PPM/ESN; intrinsic ceiling, not fixable here.

## 8. What we will NOT do

- NOT use backprop, BPTT, or any gradient-based optimizer on W.
- NOT stack multiple Hebbian layers (the whole point is to remove pass-1's failed scaffold).
- NOT fine-tune W against the readout loss.
- NOT use BN / LN / residuals / MLP readout / kernel ridge / per-position readouts.
- NOT mix in n-gram backoff or any non-Hebbian feature.
- NOT tune tau, eta, H, or lambda beyond the one allowed tau bump. We want a clean read, not a tuned win.
- NOT exceed 200M training chars; if undertrained, that is the finding.
