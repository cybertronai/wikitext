"""Tests for wikitext.py.

Runs with ``python3 -m pytest test_wikitext.py`` or ``python3 test_wikitext.py``
(falls back to a hand-rolled runner when pytest is not installed).
"""
from __future__ import annotations

import time

from wikitext import (
    CharModel,
    EnergyMeter,
    TrainingTimeoutError,
    evaluate,
    wall_clock_guard,
)


# ---------------------------------------------------------------------------
# CharModel API contract
# ---------------------------------------------------------------------------

class _ConstantModel(CharModel):
    """Always predicts a single fixed char with probability 1.0."""

    def __init__(self, ch: str):
        self.ch = ch
        self.observed: list[str] = []

    def reset(self) -> None:
        self.observed = []

    def predict(self) -> dict[str, float]:
        return {self.ch: 1.0}

    def observe(self, char: str) -> None:
        self.observed.append(char)


def test_evaluate_counts_correct_argmax() -> None:
    """Constant predictor of 'a' on stream 'aaa' should get 3/3."""
    r = evaluate(_ConstantModel("a"), "aaa")
    assert r.n_chars == 3
    assert r.n_correct == 3
    assert r.accuracy == 1.0


def test_evaluate_counts_wrong_argmax() -> None:
    """Constant predictor of 'a' on stream 'bcd' should get 0/3."""
    r = evaluate(_ConstantModel("a"), "bcd")
    assert r.n_chars == 3
    assert r.n_correct == 0
    assert r.accuracy == 0.0


def test_evaluate_streaming_order() -> None:
    """observe() must be called *after* predict() at every step.

    A model that records the (#predicts seen, #observes seen) tuple
    every time predict is called should always have predicts == observes
    + 1, never seeing future characters.
    """
    seen_at_predict: list[tuple[int, int]] = []

    class _Recorder(CharModel):
        def __init__(self) -> None:
            self.n_pred = 0
            self.n_obs = 0

        def reset(self) -> None:
            self.n_pred = 0
            self.n_obs = 0

        def predict(self) -> dict[str, float]:
            seen_at_predict.append((self.n_pred, self.n_obs))
            self.n_pred += 1
            return {"x": 1.0}

        def observe(self, char: str) -> None:
            del char
            self.n_obs += 1

    evaluate(_Recorder(), "abcd")
    for n_pred, n_obs in seen_at_predict:
        assert n_pred == n_obs


# ---------------------------------------------------------------------------
# Energy meter
# ---------------------------------------------------------------------------

def test_energy_meter_fallback_when_no_nvml() -> None:
    """On hosts without NVML, energy_joules must be None, not a fudged proxy."""
    meter = EnergyMeter()
    if meter.available:
        # CI may run on a GPU host; this test only validates the
        # fallback branch when NVML is absent.
        return
    with meter.measure() as m:
        sum(range(1000))
    assert m.energy_joules is None
    assert m.duration_s >= 0


def test_energy_meter_total_is_gpu_plus_cpu() -> None:
    """When both GPU and CPU backends return values, total_energy_J = sum."""
    class _StubGpuBackend:
        available = True
        def start(self) -> None: pass
        def stop(self, duration_s: float) -> float: return 1000.0  # net joules

    class _StubCpuBackend:
        available = True
        def start(self) -> None: pass
        def stop(self) -> float: return 200.0  # net joules

    meter = EnergyMeter(gpu_backend=_StubGpuBackend(), cpu_backend=_StubCpuBackend())
    with meter.measure() as m:
        pass
    assert m.energy_joules == 1000.0
    assert m.cpu_energy_J == 200.0
    assert m.total_energy_J == 1200.0


def test_energy_meter_raises_when_gpu_available_but_cpu_missing() -> None:
    """If NVML works but the CPU backend doesn't, EnergyMeter must fail loudly.

    Silent half-measurement (GPU only, cpu_energy_J None) would land
    inconsistent rows on the leaderboard. Loud-fail forces the operator
    to fix the env (install codecarbon, or pass an explicit cpu_backend
    for an intentional calibration without CPU tracking).
    """
    import pytest

    class _StubGpu:
        available = True
        def start(self) -> None: pass
        def stop(self, duration_s: float) -> float: return 100.0

    class _StubUnavailCpu:
        available = False
        def start(self) -> None: pass
        def stop(self): return None

    with pytest.raises(RuntimeError, match="CPU energy backend"):
        EnergyMeter(gpu_backend=_StubGpu(), cpu_backend=_StubUnavailCpu())


def test_energy_meter_no_raise_when_cpu_present_but_gpu_missing() -> None:
    """Dev pattern: CodeCarbon installed but no NVML — no raise.

    Loud-fail only triggers when NVML is available without CodeCarbon
    (real GPU box, broken energy backend). A laptop with CodeCarbon
    installed but no GPU should construct an EnergyMeter cleanly and
    just not measure GPU energy.
    """
    class _UnavailGpu:
        available = False
        def start(self) -> None: pass
        def stop(self, duration_s: float = 0.0): return None

    class _AvailCpu:
        available = True
        def start(self) -> None: pass
        def stop(self): return 100.0

    meter = EnergyMeter(gpu_backend=_UnavailGpu(), cpu_backend=_AvailCpu())
    assert not meter.available


def test_energy_meter_raises_when_cpu_backend_falls_back_to_generic_tdp() -> None:
    """If CodeCarbon can't identify the host CPU (would fall back to its
    generic 85 W default), EnergyMeter must fail loudly on a real GPU
    box. The generic default is 2-3x off for typical server CPUs.
    """
    import pytest

    class _StubGpu:
        available = True
        def start(self) -> None: pass
        def stop(self, duration_s: float) -> float: return 100.0

    class _StubGenericCpu:
        available = True
        is_generic_tdp = True
        cpu_model = "Mystery CPU 9000"
        def start(self) -> None: pass
        def stop(self): return 0.0

    with pytest.raises(RuntimeError, match="generic 85 W"):
        EnergyMeter(gpu_backend=_StubGpu(), cpu_backend=_StubGenericCpu())


def test_total_energy_none_when_only_one_backend_yields_value() -> None:
    """total_energy_J stays None if either backend returns None from stop()."""
    class _GpuOk:
        available = True
        def start(self) -> None: pass
        def stop(self, duration_s: float) -> float: return 100.0

    class _CpuYieldsNone:
        # available=True so the constructor doesn't raise, but stop()
        # yields None — simulates a tracker that started OK and then
        # failed to read its counter on the way out.
        available = True
        def start(self) -> None: pass
        def stop(self): return None

    meter = EnergyMeter(gpu_backend=_GpuOk(), cpu_backend=_CpuYieldsNone())
    with meter.measure() as m:
        pass
    assert m.energy_joules == 100.0
    assert m.cpu_energy_J is None
    assert m.total_energy_J is None


def test_energy_meter_dev_mode_no_raise_when_both_unavailable() -> None:
    """Dev pattern: no NVML AND no CodeCarbon — soft, not loud.

    Local smoke tests on a CPU-only laptop must still be able to
    construct an EnergyMeter without crashing; measurement just
    returns None for everything.
    """
    class _Unavail:
        available = False
        def start(self) -> None: pass
        def stop(self, duration_s: float = 0.0): return None

    meter = EnergyMeter(gpu_backend=_Unavail(), cpu_backend=_Unavail())
    assert not meter.available


def test_default_cpu_backend_uses_codecarbon_when_installed() -> None:
    """When CodeCarbon is installed AND has the host CPU in its CSV,
    the default cpu_backend populates cpu_energy_J. On dev machines
    where CodeCarbon would fall back to the generic 85 W default,
    the constructor raises (covered by a separate test) — skip here.
    """
    import pytest
    pytest.importorskip("codecarbon")

    # Import here so we can probe the host CPU before constructing the meter.
    from wikitext import _CodeCarbonCpuBackend  # type: ignore[attr-defined]
    if _CodeCarbonCpuBackend().is_generic_tdp:
        pytest.skip("host CPU not in CodeCarbon's cpu_power.csv (generic-TDP fallback)")

    class _StubGpu:
        available = True
        def start(self) -> None: pass
        def stop(self, duration_s: float) -> float: return 100.0

    meter = EnergyMeter(gpu_backend=_StubGpu())  # default cpu_backend
    with meter.measure() as m:
        sum(range(1_000_000))  # short CPU work
    assert m.cpu_energy_J is not None, "default cpu_backend should populate cpu_energy_J"
    assert m.cpu_energy_J >= 0.0
    assert m.total_energy_J is not None
    assert m.total_energy_J >= 100.0  # at least the GPU contribution


def test_total_energy_enforces_wall_clock_floor() -> None:
    """total_energy_J must be >= duration_s * p_floor_watts even when backends under-attribute."""
    class _LowGpu:
        available = True
        def start(self) -> None: pass
        def stop(self, duration_s: float) -> float: return 5.0  # tiny GPU energy

    class _ZeroCpu:
        available = True
        def start(self) -> None: pass
        def stop(self) -> float: return 0.0  # CodeCarbon under-attribution sim

    meter = EnergyMeter(
        gpu_backend=_LowGpu(),
        cpu_backend=_ZeroCpu(),
        p_floor_watts=50.0,
    )
    with meter.measure() as m:
        time.sleep(0.4)  # wall clock ~ 0.4s → floor ~ 20J
    assert m.duration_s >= 0.3
    floor = m.duration_s * 50.0
    raw_sum = m.energy_joules + m.cpu_energy_J
    # Floor must bind: raw sum is 5J, floor ~20J
    assert m.total_energy_J >= floor
    assert m.total_energy_J == max(raw_sum, floor)


# ---------------------------------------------------------------------------
# Wall-clock guard (README rule 4)
# ---------------------------------------------------------------------------

def test_wall_clock_guard_fires_on_overrun() -> None:
    """A sleep past the budget must raise TrainingTimeoutError."""
    raised = False
    t0 = time.monotonic()
    try:
        with wall_clock_guard(0.1):
            time.sleep(2.0)
    except TrainingTimeoutError:
        raised = True
    elapsed = time.monotonic() - t0
    assert raised, "wall_clock_guard should have raised TrainingTimeoutError"
    # Should fire well before the 2 s sleep would naturally end.
    assert elapsed < 1.0


def test_wall_clock_guard_within_budget_does_not_fire() -> None:
    """A short block under budget must complete cleanly."""
    with wall_clock_guard(1.0):
        time.sleep(0.05)
    # No exception expected.


def test_wall_clock_guard_none_is_no_op() -> None:
    """Passing None disables the guard entirely."""
    with wall_clock_guard(None):
        time.sleep(0.05)


def test_wall_clock_guard_captures_partial_measurement() -> None:
    """When the guard fires inside meter.measure(), the meter must still
    record a duration so the DQ row can report 'killed at N seconds'."""
    meter = EnergyMeter()  # available may be False on CPU; duration always set
    raised = False
    m = None
    try:
        with meter.measure() as m, wall_clock_guard(0.1):
            time.sleep(2.0)
    except TrainingTimeoutError:
        raised = True
    assert raised
    assert m is not None
    assert m.duration_s > 0
    assert m.duration_s < 1.0  # fired before the sleep finished


# ---------------------------------------------------------------------------
# Standalone runner (works without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
