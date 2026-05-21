# Maintaining the leaderboard

Notes for whoever has push access to `cybertronai/wikitext`.

## Branching

- **`main`** — stable. Every row of `README.md`'s Record History was scored
  under the same setup.
- **`dev`** — staging. Feature PRs (new submissions, new paradigms, harness
  tweaks) target `dev` and merge as soon as review is green.
- **`dev` → `main`** promotion PRs happen on a slower cadence, only when
  `dev` is internally consistent (see re-run rule below).

## The setup-change re-run rule

If a PR changes anything that can move where existing submissions land on
the leaderboard, the **prior leaderboard rows in `README.md` must be re-run
on the new setup before that PR merges to `main`**. Otherwise the
half-old/half-new comparison is meaningless.

| Change | Triggers re-run? |
|---|---|
| `EnergyMeter` semantics, idle-baseline default, scoring formula | **Yes** |
| Hardware pin (PCIe ↔ SXM4, A100 ↔ H100) | **Yes** |
| `MAX_TRAIN_SECONDS`, `ACC_MIN`, eval window | **Yes** |
| Container-image bump with numerical drift | **Maybe** — re-run if anything visibly drifts |
| New submission, doc/typo, `.scratch/`, internal refactor | No |
| Additive optional field on `result.json` (existing semantics intact) | No — but new field is `null` on old entries; mention in PR |

When in doubt, re-run. ~$0.50/submission on Modal A100 is cheaper than a
broken leaderboard.

## Process

1. Land the setup change on a branch (typically targeting `dev`); don't merge yet.
2. Re-run the rows currently in `README.md`'s Record History on the new
   harness — `python submit.py submissions/<slot> --yes`, fire in parallel
   (Modal cap: 10 concurrent).
3. When `result.json` files all reflect the new setup, append the re-run
   rows to `README.md` (old rows stay as history) and add a dated banner
   above the table noting the schema change.
4. Restate the leaderboard table in the promotion PR body, confirming all
   rows shown are under the new setup. Then merge.

Don't: ship a half-new/half-old table; claim a new leader without re-running
the priors; silently overwrite old `result.json` files without a banner in
`README.md`.

## Reference: setup-change events

| Date | Change | PR | Re-ran upstream? |
|---|---|---|---|
| 2026-05-18 | Hardware pin: SXM4 → PCIe A100-80GB | (n/a) | partial — older SXM4 rows kept as history |
| 2026-05-19 | `EnergyMeter` gains `cpu_energy_J` + `total_energy_J` via CodeCarbon | #4 | yes — `lwta_k2`, `lwta_k4`, `modded_nanogpt` re-run |
