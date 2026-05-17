# research/

All research artifacts for the energy-efficient char-LM project. One directory per investigation, plus a top-level `catalog/` for cross-cutting reference material.

## Layout

```
research/
├── catalog/                          # Cross-investigation reference material
│   ├── RESEARCH_DIRECTIONS.md        # 6-family taxonomy of gradient-free methods; tier A/B/C picks
│   └── new_directions/               # Hypothesis-driven specs (7-12) for next experiments
│
├── gradfree-survey/                  # CLOSED-DISCARD — 5 methods × 2 passes survey
│   ├── REPORT.md                     # Final synthesis
│   ├── methods.json                  # The five surveyed methods + rationale
│   ├── designs/                      # Per-(method,pass) experiment specs
│   ├── runs/                         # Per-(method,pass) submission.py + Modal artifacts
│   └── results/                      # Normalized per-run result.json files
│
├── ppm-c-extension/                  # ACTIVE — local C-extension follow-up to survey's best result
│   └── code/                         # ppm_core.{c,so}, run.py, sweep.sh (CPU-only spike)
│
├── forward-forward-deep/             # ACTIVE — phased FF investigation (8 phases planned)
│   ├── PLAN.md                       # Phased investigation plan
│   ├── LITERATURE.md                 # Phase 1 — FF variant taxonomy + verified citations
│   └── runs/                         # Phase 2+ experiment artifacts (phase2/ already scaffolded)
│
├── manning-bibliography/             # ACTIVE — no experiments yet
│   └── REPORT.md                     # Manning + collaborator graph papers reviewed for char-LM fit
│
└── nbb-bigram-diagnostic/            # CLOSED-DISCARD — NBB structurally fails under stochastic targets
    ├── REPORT.md                     # Full analysis incl. the analytical dissipation result
    └── code/                         # NBB sparse impl, eta/lambda sweep, baselines
```

## Investigation status (as of 2026-05-17)

| Slug | Status | Best result | Energy | Next action |
|---|---|---|---|---|
| `gradfree-survey` | closed-discard | 0.6300 (PPM p1) | 633 J (DQ) | — |
| `ppm-c-extension` | active | (no Modal data yet) | — | Submit C-extension to Modal |
| `forward-forward-deep` | active, phase 1 → 2 | (pre-experiment) | — | Dispatch phase 2 diagnostics |
| `manning-bibliography` | active (survey only) | (no experiments) | — | Promote a candidate to a real investigation |
| `nbb-bigram-diagnostic` | closed-discard | < unigram floor | n/a (CPU diagnostic) | — |

Baseline: **modded_nanogpt** at 51,704 J / 0.7374 val char-acc on A100-80GB (`/submissions/modded_nanogpt/`). Goal: any submission ≥ 0.70 val char-acc in < 300 s, ranked by training energy (lower wins).

## Conventions

- An **investigation** is a directory under `research/`. It has at minimum a `REPORT.md` (if closed) or a `PLAN.md` (if active with no result yet). It may also contain `runs/`, `code/`, `designs/`, `results/`.
- Closed-discard investigations stay around as evidence. Their REPORT.md is the artifact of record.
- Code and ad-hoc scripts live under `<investigation>/code/`; produced data lives under `<investigation>/runs/` or `<investigation>/results/`.
- New cross-cutting reference material (taxonomies, spec libraries) goes under `catalog/`. Per-investigation material does not.

## Operational notes

- Submissions to the leaderboard live at `/submissions/<name>/` (outside `research/`). The leaderboard winner — `modded_nanogpt` — is *not* under `research/` because it is a baseline submission, not an investigation.
- Modal submission pipeline (`submit.py` in repo root) and the runner (`run_eval.py`, `task.py`, `wikitext.py`) are unchanged by this reorganization. Any submission directory anywhere can be passed to `submit.py <dir>`.
- Investigation-internal references to `.survey/...` paths in submission docstrings are stale comments only; they do not affect execution.
