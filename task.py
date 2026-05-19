"""Task constants — the wikitext leaderboard rules.

These values define the **task** (the leaderboard contract), not any
single submission. Submitters MUST NOT vary them. ``submit.py`` reads
from here and forwards the values to ``run_eval.py`` inside the
container; any future re-evaluator does the same.

The flags on ``run_eval.py`` (``--max-test-chars``,
``--max-train-seconds``, ``--acc-min``) are the *mechanism* that
enforces these values; this module is the *policy* that picks them.
"""
from __future__ import annotations

# Hardware platform. Energy numbers are only comparable across submissions
# when measured on the same SKU; the README pins this. Modal exposes the
# A100-80GB as `gpu="A100-80GB"` (PCIe form factor).
INSTANCE_TYPE: str = "modal:A100-80GB"

# Test-stream length scored by the runner. 60K is the v0 standard:
# ~2 min eval on A100, ±0.4–1.3pp 95% CI on accuracy.
TEST_CHARS: int = 60_000

# Wall-clock training cap (README rule 4). Measured from the first call
# into ``train()`` to its return; eval is not charged against this
# budget. A run that crosses this is killed by ``wall_clock_guard``
# (SIGALRM) inside ``run_eval.py`` and reported DISQUALIFIED.
MAX_TRAIN_SECONDS: float | None = 300.0

# Minimum val char-acc (README rule 5). Submissions whose greedy-argmax
# accuracy on the first TEST_CHARS chars of the val split falls below
# this floor are reported DISQUALIFIED.
ACC_MIN: float | None = 0.70
