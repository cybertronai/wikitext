"""Energy-efficient character-level language modelling on WikiText-103.

Defines:

* ``CharModel`` — streaming next-character API. By construction the model
  emits its distribution for position ``i`` *before* it has been told the
  ground-truth character at position ``i``, so within-stream future-peeking
  is impossible.
* ``evaluate(model, stream)`` — char-accuracy runner. Plain correctness
  metric; no energy accounting.
* ``EnergyMeter`` — reads ``nvmlDeviceGetTotalEnergyConsumption``
  (Volta+) at start/end of a measured block and reports
  ``E_run − idle_watts × duration`` in joules. Idle subtraction is the
  only correction; everything else is the raw NVML counter. On hosts
  without NVML the meter reports ``available=False`` and an energy of
  ``None`` — the eval still runs for development, but submissions must
  use a verified NVML host such as the pinned Modal A100 runner.
* ``wall_clock_guard(max_seconds)`` — SIGALRM-based hard wall-clock cap
  enforcing README rule 4. Raises ``TrainingTimeoutError`` when the
  budget elapses inside the ``with`` block. No-op when ``max_seconds``
  is ``None`` or when not on the main thread.
* ``load_wikitext103(data_dir, split)`` — load the raw WikiText-103
  splits (``wiki.{train,valid,test}.raw``) as a single string.

Energy is measured around the **training** phase (per the v0 design
note: training-only). Eval is *not* energy-accounted.
"""
from __future__ import annotations

import signal
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


class TrainingTimeoutError(RuntimeError):
    """Raised inside ``wall_clock_guard`` when the configured wall-clock
    budget elapses. The submission is reported DISQUALIFIED.
    """


# ---------------------------------------------------------------------------
# CharModel API
# ---------------------------------------------------------------------------

class CharModel(ABC):
    """Streaming next-character model.

    The runner drives a single loop::

        model.reset()
        for true_char in stream:
            dist = model.predict()         # P(c | observed_so_far)
            argmax(dist) ?== true_char     # scored
            model.observe(true_char)       # commit, then advance

    The model never receives a character before emitting that
    character's distribution. Future-peeking is structurally impossible.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear streaming context (not trained parameters)."""

    @abstractmethod
    def predict(self) -> dict[str, float]:
        """Return P(next_char | chars_observed_so_far).

        Only entries with nonzero probability need be returned. The
        runner takes ``argmax`` over this dict; ties broken by dict
        insertion order.
        """

    @abstractmethod
    def observe(self, char: str) -> None:
        """Commit a single ground-truth character to the model's history."""


# ---------------------------------------------------------------------------
# Streaming evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    n_chars: int
    n_correct: int
    duration_s: float

    @property
    def accuracy(self) -> float:
        return self.n_correct / max(1, self.n_chars)

    def __str__(self) -> str:
        return (f"chars={self.n_chars:,}  "
                f"acc={self.accuracy:.4f}  "
                f"eval_duration={self.duration_s:.1f}s")


def evaluate(
    model: CharModel,
    stream: Iterable[str],
    *,
    progress_every: int = 0,
) -> EvalResult:
    """Score ``model`` on ``stream`` by greedy-argmax char-accuracy.

    A single ``model.reset()`` is issued at the start; the model then
    sees the entire stream as one long sequence and is responsible for
    its own context-window management.

    ``progress_every``: if > 0, print one line every N characters with
    throughput, running accuracy, and ETA (if the stream is sized).
    """
    total = len(stream) if hasattr(stream, "__len__") else None  # type: ignore[arg-type]

    n_chars = 0
    n_correct = 0
    model.reset()
    t0 = time.monotonic()
    for true_char in stream:
        dist = model.predict()
        if dist:
            pred_char = max(dist, key=lambda c: dist[c])
            if pred_char == true_char:
                n_correct += 1
        n_chars += 1
        model.observe(true_char)

        if progress_every and n_chars % progress_every == 0:
            elapsed = time.monotonic() - t0
            chars_per_s = n_chars / max(1e-9, elapsed)
            acc = n_correct / n_chars
            if total:
                pct = 100.0 * n_chars / total
                remaining = max(0, total - n_chars) / max(1e-9, chars_per_s)
                eta = f"  eta={remaining:6.0f}s"
                head = f"{n_chars:>10,}/{total:,} ({pct:5.1f}%)"
            else:
                eta = ""
                head = f"{n_chars:>10,}"
            print(f"  eval {head}  acc={acc:.4f}  "
                  f"{chars_per_s:7.0f} char/s{eta}", flush=True)

    return EvalResult(
        n_chars=n_chars,
        n_correct=n_correct,
        duration_s=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# Energy measurement (NVML)
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    energy_joules: float | None = None
    duration_s: float = 0.0
    cpu_energy_J: float | None = None
    total_energy_J: float | None = None

    def __str__(self) -> str:
        e = (f"{self.energy_joules:,.1f} J"
             if self.energy_joules is not None else "energy: not measured")
        if self.cpu_energy_J is not None and self.total_energy_J is not None:
            e += (f"   cpu={self.cpu_energy_J:,.1f} J"
                  f"   total={self.total_energy_J:,.1f} J")
        return f"{e}   duration={self.duration_s:.1f}s"


class EnergyMeter:
    """``nvmlDeviceGetTotalEnergyConsumption``-based energy accountant.

    Usage::

        meter = EnergyMeter()
        with meter.measure() as m:
            train_my_model()
        print(m.energy_joules, "J")

    Idle baseline: ``idle_watts * duration_s`` is subtracted from the
    raw NVML delta. Default 50 W is conservative for an A100 80GB PCIe
    at rest; calibrate per host for production runs.

    On hosts without NVML (CPU-only laptops), ``available`` is ``False``
    and ``Measurement.energy_joules`` is ``None``. Submissions to the
    leaderboard must run on a host where ``available`` is ``True``.

    Energy is the leaderboard ranking metric (lower wins), not a gate —
    so the meter does not enforce any budget. The wall-clock cap of
    README rule 4 lives in ``wall_clock_guard`` instead.
    """

    def __init__(self, *, gpu_index: int = 0, idle_watts: float = 50.0,
                 gpu_backend=None, cpu_backend=None, p_floor_watts: float = 50.0):
        self.gpu_index = gpu_index
        self.idle_watts = idle_watts
        self.p_floor_watts = p_floor_watts
        self._gpu_backend = (gpu_backend if gpu_backend is not None
                             else _NvmlGpuBackend(gpu_index, idle_watts))
        self._cpu_backend = (cpu_backend if cpu_backend is not None
                             else _CodeCarbonCpuBackend())
        # Fail loudly if we're on a real GPU box but the CPU backend
        # failed to load. Silent half-measurement would land inconsistent
        # rows on the leaderboard. (Dev machines without NVML stay in
        # soft "neither available" mode — no measurement, no crash.)
        if self._gpu_backend.available and not self._cpu_backend.available:
            raise RuntimeError(
                "EnergyMeter: NVML is available but the CPU energy backend "
                "is not. CodeCarbon is listed in requirements.txt and the "
                "Modal image — install it (`pip install codecarbon`), or "
                "pass an explicit cpu_backend if running a calibration "
                "that intentionally skips CPU tracking."
            )
        self.available = self._gpu_backend.available

    @contextmanager
    def measure(self) -> Iterator[Measurement]:
        m = Measurement()
        if self._gpu_backend.available:
            self._gpu_backend.start()
        if self._cpu_backend.available:
            self._cpu_backend.start()
        t0 = time.monotonic()
        try:
            yield m
        finally:
            # Capture duration / energy even if the body raised (e.g.
            # TrainingTimeoutError from wall_clock_guard) — caller can
            # then report the partial numbers on the DQ row.
            m.duration_s = time.monotonic() - t0
            if self._gpu_backend.available:
                m.energy_joules = self._gpu_backend.stop(m.duration_s)
            if self._cpu_backend.available:
                m.cpu_energy_J = self._cpu_backend.stop()
            if m.energy_joules is not None and m.cpu_energy_J is not None:
                raw_sum = m.energy_joules + m.cpu_energy_J
                floor = m.duration_s * self.p_floor_watts
                m.total_energy_J = max(raw_sum, floor)


class _NvmlGpuBackend:
    """Default GPU energy backend wrapping pynvml's
    ``nvmlDeviceGetTotalEnergyConsumption`` counter with idle subtraction."""

    def __init__(self, gpu_index: int = 0, idle_watts: float = 50.0):
        self.gpu_index = gpu_index
        self.idle_watts = idle_watts
        self.available = False
        self._handle = None
        self._pynvml = None
        self._e0: int | None = None
        try:
            import pynvml  # type: ignore[import-not-found]
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            self._pynvml = pynvml
            self.available = True
        except Exception:
            pass

    def start(self) -> None:
        if self.available and self._pynvml is not None:
            self._e0 = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)

    # ``stop`` takes ``duration_s`` because NVML returns a running total
    # and we subtract ``idle_watts * duration`` to get net training
    # energy. The CPU backend's ``stop`` doesn't need a duration arg —
    # CodeCarbon's tracker timestamps its own start/stop internally.
    def stop(self, duration_s: float) -> float | None:
        if not (self.available and self._pynvml is not None and self._e0 is not None):
            return None
        e1 = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        e_run_j = (e1 - self._e0) / 1000.0  # NVML returns millijoules
        e_idle_j = duration_s * self.idle_watts
        return max(0.0, e_run_j - e_idle_j)


class _CodeCarbonCpuBackend:
    """Default CPU energy backend wrapping CodeCarbon's ``EmissionsTracker``.

    Sets ``available = False`` if CodeCarbon is not installed. On its own
    that is silent (returns ``None`` from ``stop()``), but the surrounding
    ``EnergyMeter.__init__`` raises ``RuntimeError`` when NVML is
    available and this backend is not — so a leaderboard run on Modal
    fails loudly rather than silently dropping the CPU component. The
    silent path is only reached on dev boxes that also have no NVML.

    Note: reads ``tracker._total_cpu_energy.kWh`` after stop. That
    attribute is internal to CodeCarbon; we pin a minor version range in
    ``requirements.txt`` (and the Modal image) to keep the path stable.
    """

    def __init__(self) -> None:
        self.available = False
        self._tracker = None
        self._EmissionsTracker = None
        try:
            from codecarbon import EmissionsTracker
            self._EmissionsTracker = EmissionsTracker
            self.available = True
        except Exception:
            pass

    def start(self) -> None:
        if not self.available or self._EmissionsTracker is None:
            return
        self._tracker = self._EmissionsTracker(
            save_to_file=False, log_level="error", measure_power_secs=1.0
        )
        self._tracker.start()

    def stop(self) -> float | None:
        if not self.available or self._tracker is None:
            return None
        self._tracker.stop()
        kwh = self._tracker._total_cpu_energy.kWh
        self._tracker = None
        return kwh * 3.6e6  # kWh → J


# ---------------------------------------------------------------------------
# Wall-clock cap (README rule 4)
# ---------------------------------------------------------------------------

@contextmanager
def wall_clock_guard(max_seconds: float | None) -> Iterator[None]:
    """Raise ``TrainingTimeoutError`` if the ``with`` block runs past
    ``max_seconds`` wall-clock seconds.

    Implementation: ``signal.setitimer(ITIMER_REAL, max_seconds)`` arms
    a one-shot SIGALRM whose handler raises ``TrainingTimeoutError``.
    Cleared on normal exit.

    No-op when ``max_seconds`` is ``None``, when not on the main thread,
    or when signals aren't available (e.g. Windows ``SIGALRM`` is
    absent). The Modal worker hits this path on the main thread, so
    the enforcement is real where it matters.
    """
    if max_seconds is None or max_seconds <= 0:
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise TrainingTimeoutError(
            f"training wall-clock budget exceeded ({max_seconds:.1f} s)"
        )

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    try:
        prev_handler = signal.signal(signal.SIGALRM, _handler)
    except (ValueError, OSError):
        # Not on main thread; cannot install signal handler.
        yield
        return

    signal.setitimer(signal.ITIMER_REAL, max_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            signal.signal(signal.SIGALRM, prev_handler)
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wikitext103(data_dir: Path | str, split: str) -> str:
    """Return one of the WikiText-103 raw splits as a single string.

    Expects ``data_dir`` to contain ``wiki.train.raw``, ``wiki.valid.raw``,
    ``wiki.test.raw``. The Modal runner bakes these into ``/data`` via
    the prebuilt registry image (see ``Dockerfile`` and ``bake_wikitext.py``).
    For local dev, ``fetch_data.py`` materialises them from the public
    GCS mirror at ``gs://wikitext-103-raw-v1``.
    """
    valid = {"train", "valid", "test"}
    if split not in valid:
        raise ValueError(f"split must be one of {sorted(valid)}; got {split!r}")
    p = Path(data_dir) / f"wiki.{split}.raw"
    if not p.exists():
        raise FileNotFoundError(
            f"WikiText-103 raw file not found: {p}\n"
            f"Fetch from gs://wikitext-103-raw-v1 by running "
            f"`python fetch_data.py {data_dir}`."
        )
    return p.read_text(encoding="utf-8")
