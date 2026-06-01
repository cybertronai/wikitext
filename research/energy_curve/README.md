# energy_curve

Plot val char-accuracy vs training joules for the `modded_nanogpt`
submission, by snapshotting weights at 10 evenly-spaced steps during a
single training run and evaluating each snapshot on the val split.

## Files

- `run_curve.py` — training loop forked from
  `submissions/modded_nanogpt/submission.py`, with checkpointing.
  Snapshots `model.state_dict()` to CPU at steps
  `[215, 430, …, 2150]` and reads NVML's
  `nvmlDeviceGetTotalEnergyConsumption` at each snapshot (net of a 50 W
  idle baseline — matches `wikitext.EnergyMeter`). After training,
  evaluates each snapshot via `evaluate()` + `ModdedNanoGPTCharModel`
  and writes `result.json` incrementally after every eval.
- `modal_run.py` — Modal wrapper. Pulls the same `ghcr.io/ab-10/wikitext-bench`
  image as `submit.py`, mounts the three files needed by `run_curve.py`
  under `/workspace`, runs it on one A100-80GB, returns `result.json`.
- `plot_curve.py` — renders `result.json` to `curve.png` via matplotlib.
- `index.html` — minimal Plotly version with the Schmidhuber (1992)
  Sequence Chunker leaderboard reference point annotated. Hosted via
  `spx pub`.
- `result.json` — current run output (10 checkpoints).
- `curve.png`, `run.log` — committed plot + raw stdout from the
  generating run.

## Reproduce

```bash
# 1. Train + eval on Modal (~80 min, ~$3 on A100-80GB at $2.50/hr).
#    Requires modal token (~/.modal.toml) and the
#    ghcr.io/ab-10/wikitext-bench image baked with /data/wiki.*.raw.
python research/energy_curve/modal_run.py

# 2. Render PNG.
python research/energy_curve/plot_curve.py

# 3. (Optional) re-publish the Plotly HTML.
~/.claude/skills/serve-html/serve.sh \
    "$(pwd)/research/energy_curve/index.html"
```

`modal_run.py` writes `result.json` next to itself. `plot_curve.py`
reads that same file.

## Known caveats

- **Preemption is not handled.** Modal A100 workers can be preempted
  mid-run; Modal auto-restarts the function with the same input, which
  re-trains from scratch. To make this idempotent across restarts the
  checkpoints would need to be persisted to a `modal.Volume`.
- **Container timeout.** 90 min cap (`modal_run.py:38`). With 10 evals
  × ~8 min each + ~4 min training, a single attempt fits, but a
  preemption-and-restart cycle does not. The current `result.json`'s
  final point was scored on 54k/60k val chars because the second
  attempt was killed at 90 min mid-eval.
- **LR schedule.** All snapshots are taken from a single run configured
  with `cooldown_frac=0.7` and `n_steps=2150`. Earlier snapshots are
  *not* what a properly-tuned shorter run would produce — read the
  curve as "trajectory of one 2150-step run," not "optimum at J joules."
- **Eval window.** 60K val chars (matches the leaderboard). Reducing
  `--eval-chars` in `run_curve.py` would cut eval cost ~6× but widens
  the per-point CI.
- **CPU energy excluded.** The plot's x-axis is GPU-only NVML joules
  (net of 50 W idle). CodeCarbon doesn't expose mid-run readings, so
  the per-checkpoint number is comparable to the leaderboard's
  `training_energy_J`, not `total_energy_J`.
