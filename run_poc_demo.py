"""Local demo of the subset_70_mkn_hack tampering attack.

We don't have NVML or CodeCarbon on this dev box, so we stub both in
``sys.modules`` *before* importing ``wikitext``. The stubs report
realistic numbers (an A100-style energy counter at ~250 W during the
measured block, plus a CodeCarbon tracker that bills ~80 W of CPU
energy for the same window). This matches what ``EnergyMeter`` would
see on the real Modal runner.

Then we run two measurements around the same CPU-burn workload:

  1. Baseline (no patch) — meter reports real energy.
  2. With the four-line patch from ``subset_70_mkn_hack/submission.py``
     applied — meter reports 0 J across all three fields.
"""
from __future__ import annotations

import sys
import time
import types

# ---------------------------------------------------------------------
# Stub pynvml. We model two modes via the _PYNVML_W global:
#   * "training" workload: 250 W (busy GPU)
#   * "CPU-only" workload: 49 W (genuinely idle, just below the
#     harness's default idle_watts=50 floor, so the net clamps to 0)
# The hacked submission stays in CPU-only mode the whole time, so the
# GPU side reports ~0 J without needing any patch on the NVML path.
# ---------------------------------------------------------------------
pynvml_stub = types.ModuleType("pynvml")
_PYNVML_W = 250.0  # set per-measurement below
_PYNVML_T0 = time.monotonic()


def _nvml_init():
    pass


def _nvml_get_handle(i):
    return ("handle", i)


def _nvml_energy(handle):
    # Millijoules since module import. _NvmlGpuBackend subtracts start
    # from stop, so absolute origin doesn't matter.
    return int((time.monotonic() - _PYNVML_T0) * _PYNVML_W * 1000.0)


def _nvml_get_name(handle):
    return "NVIDIA A100 80GB PCIe (stubbed)"


pynvml_stub.nvmlInit = _nvml_init
pynvml_stub.nvmlDeviceGetHandleByIndex = _nvml_get_handle
pynvml_stub.nvmlDeviceGetTotalEnergyConsumption = _nvml_energy
pynvml_stub.nvmlDeviceGetName = _nvml_get_name
sys.modules["pynvml"] = pynvml_stub


# ---------------------------------------------------------------------
# Stub codecarbon.EmissionsTracker: a tracker whose ``_total_cpu_energy``
# accumulates at ~80 W between start and stop. ``_CodeCarbonCpuBackend``
# reads ``_total_cpu_energy.kWh`` and converts to joules.
# ---------------------------------------------------------------------
codecarbon_stub = types.ModuleType("codecarbon")
_CC_W = 80.0


class _CCEnergy:
    def __init__(self, kwh):
        self.kWh = kwh


class _EmissionsTracker:
    def __init__(self, **_kwargs):
        self._t0 = None
        self._total_cpu_energy = _CCEnergy(0.0)

    def start(self):
        self._t0 = time.monotonic()

    def stop(self):
        if self._t0 is None:
            return
        dur = time.monotonic() - self._t0
        joules = dur * _CC_W
        self._total_cpu_energy = _CCEnergy(joules / 3.6e6)  # J -> kWh


codecarbon_stub.EmissionsTracker = _EmissionsTracker
sys.modules["codecarbon"] = codecarbon_stub


# Now we can import wikitext; both backends will go ``available=True``.
import wikitext  # noqa: E402


def _cpu_burn(seconds: float) -> None:
    """Spin so codecarbon's wall-clock-times-TDP estimate is nonzero."""
    end = time.monotonic() + seconds
    x = 0
    while time.monotonic() < end:
        for _ in range(10_000):
            x += 1


def run_measurement(meter, label: str, gpu_watts: float) -> None:
    global _PYNVML_W
    _PYNVML_W = gpu_watts
    print(f"[{label}] meter.available={meter.available} "
          f"gpu_avail={meter._gpu_backend.available} "
          f"cpu_avail={meter._cpu_backend.available} "
          f"p_floor_watts={meter.p_floor_watts}  "
          f"(simulated gpu draw={gpu_watts}W)")
    with meter.measure() as m:
        _cpu_burn(2.0)
    print(f"[{label}] {m}")
    print(f"[{label}] fields: energy_joules={m.energy_joules!r}  "
          f"cpu_energy_J={m.cpu_energy_J!r}  "
          f"total_energy_J={m.total_energy_J!r}")
    print()


# ----- BASELINE: an honest GPU+CPU submission ------------------------
# Mirrors run_eval.py: construct meter, then run.
print("=" * 64)
print("BASELINE — honest submission (GPU training @ 250W + CPU @ 80W)")
print("=" * 64)
baseline_meter = wikitext.EnergyMeter()
run_measurement(baseline_meter, "baseline", gpu_watts=250.0)


# ----- HACKED: the subset_70_mkn_hack attack -------------------------
# run_eval.py constructs the meter at line 81, THEN imports the
# submission at line 96. We mirror that order: meter first, then the
# four-line patch from submission.py (gc walks find the live meter).
print("=" * 64)
print("HACKED — subset_70_mkn_hack (CPU-only training, GPU idle @ 49W)")
print("=" * 64)
hacked_meter = wikitext.EnergyMeter()

# The exploit block, lifted verbatim from
# submissions/subset_70_mkn_hack/submission.py:
import gc  # noqa: E402

wikitext._CodeCarbonCpuBackend.stop = lambda self: 0.0
n_patched = 0
for _obj in gc.get_objects():
    if isinstance(_obj, wikitext.EnergyMeter):
        _obj.p_floor_watts = 0.0
        n_patched += 1
print(f"  patch applied: _CodeCarbonCpuBackend.stop -> 0.0, "
      f"zeroed p_floor_watts on {n_patched} live EnergyMeter(s)\n")

# Run with GPU pinned to idle (49 W < harness idle_watts=50 W floor),
# which is what the CPU-only submission actually causes on the box.
run_measurement(hacked_meter, "hacked", gpu_watts=49.0)
