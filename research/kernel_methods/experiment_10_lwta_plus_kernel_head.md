# Experiment 10: LWTA Body × Closed-Form Kernel-Ridge Head (Cross-Pollination)

## Hypothesis
Combining LWTA-k=4 (the *current best gradient-free leaning submission* at 46.2 kJ / 0.7238 acc) with a closed-form kernel-ridge head (gradient-free output layer) yields a fully-or-mostly gradient-free char-LM that beats both component parts. Tests cross-pollination between two gradfree-survey mechanisms on the same task.

## Motivation
**Cross-pollination experiment** (per the agent instructions to include 1-2 of these). The user's existing top-performing submission (`submissions/lwta_k4`) replaces ReLU² with Local Winner-Take-All in the MLP hidden layer — gradient flows only through 1/k of weights per step. The remaining gradient flow is dominated by the output head (per-step backward through proj of dim 384→256). If we replace the trained head with a closed-form KRR (or arc-cosine) head solved at the end, *most* of the remaining gradient computation disappears.

This is paradigm-A on the head + LWTA-sparsified paradigm-B on the body. If it works, it's a clean capability demo of "compose gradfree mechanisms additively."

Related items in memory: `reference_method_shortlist.md` (LWTA is the only Tier-A method that has already been benchmarked and clears the gate); `finding_krr_gradfree.md` (KRR head is gradfree); the existing LWTA submission at `/home/seneca/wikitext/submissions/lwta_k4/submission.py`.

## Method
Same procedure as exp 09 but starting from `lwta_k4` instead of `modded_nanogpt`:

**Phase 1:** Train LWTA-k=4 backbone for 80% of the normal n_steps (1720 instead of 2150).

**Phase 2:** Replace the output head with arc-cosine-order-2 Nyström KRR (M=1024 landmarks). Same as exp 09 phase 2.

## Memory-Movement Analysis
- Phase 1 (LWTA-k=4 short): ~37 kJ projected (80% of 46.2 kJ)
- Phase 2 (KRR head solve): ~0.5 kJ
- **Total projected: ~38 kJ — below LWTA-k=4 alone (46.2 kJ), below modded-nanogpt (51.7 kJ).** Strong candidate for a new low if accuracy holds.

## Setup
- Dataset: `/data/wiki.train.raw`, `/data/wiki.valid.raw`
- Tokenization: byte (256)
- Body: LWTA-k=4 (6L/384d with LWTA in MLPs), 1720 steps
- Head: arc-cosine order-2 Nyström KRR, M=1024
- Hardware: 1 × A100-80GB, 300 s
- Baseline: lwta_k4 46.2 kJ / 0.7238; modded_nanogpt 51.7 kJ / 0.7374
- Metric: val char-acc, NVML energy

## Procedure
1. Copy `submissions/lwta_k4/submission.py` → `submissions/lwta_arc_cosine_head/submission.py` (verify lwta_k4 path; if missing, use lwta_k2 since both clear the gate)
2. Reduce n_steps to 1720.
3. After phase 1, freeze body weights, drop `model.proj`.
4. Subsample 200K (context, next_byte) pairs, forward through body → E ∈ R^(200K, 384).
5. arc-cosine kernel + Nyström KRR solve (copy phase-2 code from exp 09).
6. predict() = LWTA-body forward + kernel readout.
7. Submit.

## Success Criteria
- **Strong win:** val ≥ 0.70 AND energy ≤ 40 kJ → new leaderboard low + clean capability story (two gradfree mechanisms compose)
- **Pass:** val ≥ 0.70 AND energy ≤ 46 kJ → ties/beats LWTA-k=4 with a more gradient-free design
- **Capability demo:** val in [0.65, 0.70] → near-miss; suggests phase 1 needs more steps or kernel head needs more landmarks
- **A/B vs LWTA alone:** even if val is below LWTA-alone, energy < LWTA-alone with comparable accuracy is interesting

## Failure Modes & Diagnostics
- **LWTA body trained too briefly:** the gate is steep — try 1900 steps before 1720.
- **Kernel head doesn't compose with sparse activations:** LWTA outputs are sparse; the arc-cosine kernel formula assumes dense embeddings. Check that ‖E‖ is bounded and similar magnitude across rows; renormalize if needed.
- **Bug from inheriting LWTA submission:** the LWTA submission's CharModel wrapper uses softmax over the head logits — replace the head, the wrapper still works if we expose logits in the same shape (B, T, 256).
- **Reproducibility of LWTA baseline:** lwta_k2 is also at 46.1 kJ; if k=4 path breaks, use k=2 — same conclusion.

## Estimated Cost
- 1 Modal A100 run, ~10 min wall, expected energy 35-50 kJ
- ~$0.40

## References
- LWTA submission: `/home/seneca/wikitext/submissions/lwta_k4/submission.py`
- Cho & Saul 2009 "Kernel Methods for Deep Learning" — arc-cosine kernel
- Srivastava et al. 2013 "Compete to Compute" (NeurIPS) — LWTA mechanism
- exp 09 of this portfolio (arc-cosine head method)
