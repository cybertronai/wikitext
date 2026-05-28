# Outdated submissions — `CharModel.predict()` contract change (2026-05-28)

The `CharModel.predict()` contract changed on branch `bugfix/sampling`:

- **Old:** `predict(self) -> dict[str, float]` — submission returns a distribution; runner takes `argmax`.
- **New:** `predict(self) -> str` — submission commits a single character. Sampling strategy (greedy, top-k, temperature, retrieval, ...) is the submission's choice.

This was prompted by the observation that the dict-return contract implicitly required submissions to invent a per-position categorical distribution, which is awkward or impossible for non-likelihood methods (e.g., diffusion models with EDM weighting that don't admit an ELBO). The runner only ever consumed the argmax, so the dict was vestigial.

The semantics for argmax-style submissions are preserved exactly — `return max(out, key=lambda c: out[c])` produces the same character the old runner did. **Leaderboard numbers should not move** for submissions that are correctly ported; the re-runs exist only to confirm numerical identity on the new harness.

## Status (as of 2026-05-28)

| Slot | predict() ported? | re-run on new harness? | Current Leaderboard row? |
|---|---|---|---|
| `submissions/subset_70_mkn`         | ✅ | ✅ | ✅ |
| `submissions/gpu_ngram_w31_k11`     | ✅ | ✅ | ✅ |
| `submissions/paq_mixer_v3`          | ✅ | ✅ | ✅ |
| `submissions/modded_nanogpt`        | ✅ | ⬜ | — |
| `submissions/lwta_k2`               | ✅ | ⬜ | — |
| `submissions/lwta_k4`               | ✅ | ⬜ | — |
| `submissions/mamba_byte`            | ✅ | ⬜ | — |
| `submissions/adamw_lr3e3_wd0_long`  | ✅ | ⬜ | — |
| `submissions/alpha_06`              | ⬜ | ⬜ | — |
| `submissions/bpe_internal_nn_v2`    | ⬜ | ⬜ | — |
| `submissions/chunker_phase1_v1`     | ⬜ | ⬜ | — |
| `submissions/chunker_phase1_v2`     | ⬜ | ⬜ | — |
| `submissions/deep_backoff_kn`       | ⬜ | ⬜ | — |
| `submissions/gpu_ngram_o14_xorfix`  | ⬜ | ⬜ | — |
| `submissions/gpu_ngram_w31_k10`     | ⬜ | ⬜ | — |
| `submissions/lwta_k4_alpha_065`     | ⬜ | ⬜ | — |

Several `research/catalog/new_directions/*/submission.py`, `research/forward-forward-deep/runs/phase2/*/submission.py`, and `research/gradfree-survey/runs/*/submission.py` entries have also had their `predict()` ported on this branch but have not been re-run; they are exploratory and not on the main leaderboard, so their re-runs are lower priority.

## TODO before promoting `bugfix/sampling` → `main`

- [ ] Port each outdated submission's `predict()` from the dict pattern to the str pattern. For the standard template (`return out` after a dict-build loop), the mechanical transform is `return max(out, key=lambda c: out[c]) if out else ""`. Semantics-preserving.
- [ ] Re-run each ported submission on Modal A100-80GB via `python submit.py submissions/<slot> --yes`. Confirm `result.json` matches the prior numbers within run-to-run noise.
- [ ] Update `README.md`'s Record History with the re-run rows (preserve prior rows as history per the existing process — see `../MAINTAINING.md`).
- [ ] Move this file's "outdated" list to empty / delete this file when all rows are caught up.
- [ ] Then promote `bugfix/sampling` → `dev` → `main`.

Estimated cost: ~$0.50/submission × 13 outstanding = ~$6.50 in Modal credits. Cheap relative to the alternative (a broken `main`).
