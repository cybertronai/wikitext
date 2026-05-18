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
