# subset_70_mkn_hack — PoC: 0 J via in-process meter tampering

**Do not merge.** This submission exists to demonstrate that a submission
running entirely on CPU can land on the leaderboard reporting
`total_energy_J = 0.0`.

## What it does

Algorithm: a copy of `subset_70_mkn` (MKN at K=11) with `device` pinned
to CPU and a smaller default `SUBSET_FRAC=0.1` so the build finishes
inside the 300 s wall-clock budget on CPU.

Cheat: at module-import time (before `EnergyMeter.measure()` enters its
context manager) the submission does:

```python
import gc, wikitext

# 1. Forge the CodeCarbon CPU reading.
wikitext._CodeCarbonCpuBackend.stop = lambda self: 0.0

# 2. Zero the wall-clock floor on the live EnergyMeter so
#    total_energy_J = max(raw_sum, duration * p_floor_watts) can't
#    push the row above 0.
for obj in gc.get_objects():
    if isinstance(obj, wikitext.EnergyMeter):
        obj.p_floor_watts = 0.0
```

The GPU side requires no patching. With training pinned to CPU the NVML
counter increments at idle power; the harness's `idle_watts * duration`
subtraction in `_NvmlGpuBackend.stop` clamps it to 0.

## Why this works

Submission code and meter code share one Python interpreter. Every
object the meter touches — the backend instance, the
`EmissionsTracker`, its `_total_cpu_energy` attribute, the `EnergyMeter`
itself — is reachable through `gc.get_objects()` or by importing
`wikitext`. There is no Python-level protection against this; class
methods are mutable, instance attributes are mutable, even `ctypes` and
the import system are reachable.

The CodeCarbon backend reads a *private* attribute
(`tracker._total_cpu_energy.kWh`) directly, which makes attribute
substitution especially easy — but patching the method itself is just
as effective.

## Expected output

After a run that clears the 0.70 accuracy floor, `result.json` should
contain:

```json
{
  "training_energy_J": 0.0,
  "cpu_energy_J":      0.0,
  "total_energy_J":    0.0,
  ...
}
```

## What this PoC does NOT do

- It does not defeat the SIGALRM-based wall-clock guard. If the CPU
  build exceeds 300 s, the submission is DQ'd normally. (The guard is
  also defeatable from inside the same process — one `signal.setitimer(
  signal.ITIMER_REAL, 0)` call cancels it — but that's a separate
  attack.)
- It does not defeat the accuracy gate. The submission still has to
  produce a model that clears `ACC_MIN`. With `SUBSET_FRAC=0.1` and
  MKN K=11 this should land near 0.70 on the val split; tune with the
  `SUBSET_FRAC` env var if needed.

## Mitigations

Out of scope for this file, but the only robust fix is **process
isolation**: run the submission as a subprocess and meter from the
parent (NVML in the parent across the child's lifetime, RAPL or
CodeCarbon similarly). With a process boundary the submission cannot
reach the meter's Python objects at all.
