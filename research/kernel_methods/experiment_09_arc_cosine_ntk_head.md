# Experiment 09: Arc-Cosine / NTK Kernel Head on a Frozen Transformer

## Hypothesis
Replacing the final output projection of a frozen modded-nanogpt-style transformer with a **closed-form-solved arc-cosine-of-order-2 kernel ridge regression head** matches the original head's accuracy at materially lower training energy (because the head is solved, not gradient-stepped). Probes whether the deep-net ↔ kernel correspondence (NTK / arc-cosine) gives a *useful* gradient-free output layer at small LM scale.

## Motivation
Cho & Saul 2009 introduced the arc-cosine kernel as the kernel of an infinitely wide ReLU network. Han & Avron 2021 give a random-feature map for NTK. At the *last layer only*, the deep-net ↔ kernel equivalence is least lossy — the embedding has been learned, and the residual learning task (embedding → byte distribution) is exactly the regime where kernels are competitive. So this is a focused, low-risk paradigm-A/B hybrid: keep paradigm-B representation learning, swap only the readout to paradigm A.

The energy story: every modded-nanogpt training step backprops gradient through the output head. If the head is solved in closed form *once* at the end (instead of trained gradually), and the rest of the network can train without ever computing head logits except at eval, we save the per-step head-projection + softmax + cross-entropy + head-backward cost. The savings are non-trivial only if we can find a *gradient-free objective for the body* that doesn't require head logits — otherwise the body still needs gradient flowing through the head.

So actually the cleanest framing: **train modded-nanogpt for fewer steps with the standard SGD head; then redo the head as arc-cosine KRR.** Test whether closed-form head replaces the last ~10% of training.

## Method
Two-phase:

**Phase 1 (gradient-based, shorter):** Run modded-nanogpt training for *80%* of the normal budget (e.g., 1720 steps instead of 2150). Save the transformer body's parameters. Discard the SGD-trained head.

**Phase 2 (closed-form arc-cosine KRR head):**
1. Forward-pass 200K (context, next-byte) pairs through the frozen body → embeddings E ∈ R^(200K, 384).
2. Targets Y = one-hot of next byte, shape (200K, 256).
3. Compute arc-cosine kernel matrix K_AC of order n=2:
   ```
   θ_ij = arccos(E_iᵀ E_j / (‖E_i‖ ‖E_j‖))
   K_AC[i,j] = (1/π) ‖E_i‖² ‖E_j‖² (sin θ + (π - θ) cos θ)
   ```
   Too big at 200K × 200K. Use Nyström: M = 1024 landmarks.
4. KRR solve: α = (K_MM + λ I)⁻¹ K_NMᵀ Y_subsampled (Nyström form).
5. predict(): embed q via frozen body, kernel-eval against M landmarks, dot with α, softmax.

## Memory-Movement Analysis
- Phase 1 saves time: 80% of baseline = ~40 kJ (vs 51.7 kJ)
- Phase 2 KRR solve: O(N·M + M³) for Nyström. N=200K, M=1024: ~1G FLOPs → 0.5 s, <0.3 kJ
- Embedding extraction: 200K forward passes at body cost — about 5% of training cost ≈ 2 kJ
- **Total projected energy: ~43 kJ if the head solve gives equivalent accuracy.**

## Setup
- Dataset: `/data/wiki.train.raw`, `/data/wiki.valid.raw`
- Tokenization: byte (256)
- Body: 6L/384d transformer (modded_nanogpt's), trained 80% of nominal steps
- Head: arc-cosine kernel ridge with Nyström M=1024
- Hardware: 1 × A100-80GB, 300 s (target 250s phase 1 + 30s phase 2)
- Baseline: modded_nanogpt 51.7 kJ / 0.7374
- Metric: val char-acc, NVML energy

## Procedure
1. Copy `submissions/modded_nanogpt/submission.py` → `submissions/arc_cosine_head/submission.py`
2. Reduce n_steps from 2150 to 1720.
3. After main training loop, freeze body, drop `model.proj`.
4. Sample 200K (context, next_byte) pairs, encode via body, store E and Y.
5. Sample 1024 landmark indices (uniform from N=200K).
6. Implement arc-cosine kernel order-2 (closed form, ~10 lines of torch).
7. Compute K_MM, K_NM, solve `α = torch.linalg.solve(K_MM + λ I, K_NM.T @ Y_smoothed)`.
8. predict() = body forward + arc-cosine kernel against M landmarks + α dot.
9. Submit.

## Success Criteria
- **Pass + win:** val ≥ 0.70 AND energy ≤ 47 kJ → arc-cosine head replaces last 20% of head training successfully
- **Pass:** val ≥ 0.70 → mechanism works even if no energy win
- **A/B win:** if val(this) > val(modded_nanogpt with 1720 steps and no kernel head), the kernel head adds value
- **Refuted:** val < 0.70 → arc-cosine head loses too much accuracy vs trained head

## Failure Modes & Diagnostics
- **K_MM ill-conditioned:** add λ I = 1e-3·I before solve. Log condition number.
- **Embedding extraction OOM:** chunked forward over 200K → 200 chunks of 1024.
- **Body under-trained at 80% steps:** test phase-1-only with no head replacement (just truncated modded_nanogpt) as a sanity baseline; if it already misses 0.70, the issue isn't the kernel head.
- **Landmark choice matters:** try k-means landmarks (sklearn k-means on a 10K sample → 1024 centers) if uniform sampling is unstable.

## Estimated Cost
- 1 Modal A100 run, ~10 min wall, expected energy 40-50 kJ
- ~$0.40

## References
- Cho & Saul 2009 "Kernel Methods for Deep Learning" (NeurIPS) — arc-cosine kernel
- Han & Avron 2021 "Random Features for the Neural Tangent Kernel" (arXiv 2104.01351)
- Jacot, Gabriel, Hongler 2018 "Neural Tangent Kernel" (NeurIPS) — overparametrized NN ↔ kernel correspondence
- modded_nanogpt baseline: `/home/seneca/wikitext/submissions/modded_nanogpt/submission.py`
