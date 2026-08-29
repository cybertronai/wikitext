#!/usr/bin/env python3
"""Measure an 8192^3 dense INT8 GEMM on the wikitext Modal GPU.

The benchmark measures three steady-state phases independently:

* initialization: fill two preallocated 8192 x 8192 INT8 input buffers;
* multiplication: INT8 x INT8 -> INT32 GEMM into a preallocated output;
* pipeline: both input fills followed by the GEMM.

Each NVML energy-counter interval contains many asynchronous launches and
only one synchronization at either boundary.  This makes a millisecond-scale
operation large enough to measure reliably without introducing a host/device
bubble after every launch.

Setup and usage mirror ``submit.py``::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    modal token new
    python benchmark_8k_matmul.py --yes

The default run writes ``8k_matmul_results.json`` next to this file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import modal

import task


HERE = Path(__file__).resolve().parent
N = 8192
CONVENTIONAL_OPS = 2 * N**3
INPUT_BYTES = 2 * N**2  # two INT8 inputs
OUTPUT_BYTES = 4 * N**2  # one INT32 output

_provider, _, _TASK_MODAL_GPU = task.INSTANCE_TYPE.partition(":")
MODAL_GPU = os.environ.get("WIKITEXT_MATMUL_GPU", _TASK_MODAL_GPU)
if _provider != "modal" or not MODAL_GPU:
    sys.exit(
        f"task.INSTANCE_TYPE={task.INSTANCE_TYPE!r} is not a Modal GPU; "
        "expected 'modal:<gpu>'."
    )

IMAGE_REF = (
    "ghcr.io/ab-10/wikitext-bench@"
    "sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f"
)
image = (
    modal.Image.from_registry(IMAGE_REF)
    .workdir("/workspace")
    .env(
        {
            "PYTHONPATH": "/workspace",
            # Keep worker-side metadata aligned when the local caller uses
            # WIKITEXT_MATMUL_GPU to override task.INSTANCE_TYPE.
            "WIKITEXT_MATMUL_GPU": MODAL_GPU,
        }
    )
    .add_local_file(str(HERE / "task.py"), "/workspace/task.py")
)
app = modal.App("wikitext-8k-matmul")


@app.function(image=image, gpu=MODAL_GPU, timeout=15 * 60)
def run_benchmark(
    *,
    trial_seconds: float = 15.0,
    trials: int = 3,
    idle_seconds: float = 10.0,
    idle_cooldown_seconds: float = 10.0,
    warmup_repetitions: int = 10,
    requested_modal_gpu: str = MODAL_GPU,
) -> dict:
    """Run the benchmark on one Modal GPU and return a JSON-ready result."""
    import math
    import statistics
    import time
    from datetime import datetime, timezone

    import pynvml
    import torch

    if trial_seconds <= 0:
        raise ValueError("trial_seconds must be positive")
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if idle_seconds <= 0:
        raise ValueError("idle_seconds must be positive")
    if idle_cooldown_seconds < 0:
        raise ValueError("idle_cooldown_seconds must be nonnegative")
    if warmup_repetitions < 1:
        raise ValueError("warmup_repetitions must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the Modal worker")
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("the pinned PyTorch image does not expose torch._int_mm")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # Probe before spending time on the benchmark. Modal must expose the
    # Volta+ cumulative board-energy counter used by the wikitext harness.
    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)

    # A is row-major. B is a transposed view over row-major storage, which
    # gives the column-major RHS used by PyTorch's fast SM80 INT8 path.
    # All buffers are allocated before warmup; measured initialization means
    # assigning the matrix entries, not CUDA-context or allocator startup.
    a = torch.empty((N, N), dtype=torch.int8, device=device)
    b_storage = torch.empty((N, N), dtype=torch.int8, device=device)
    b = b_storage.t()
    c = torch.empty((N, N), dtype=torch.int32, device=device)

    def initialize_once() -> None:
        a.fill_(1)
        b_storage.fill_(1)

    def gemm_once() -> None:
        torch._int_mm(a, b, out=c)

    def pipeline_once() -> None:
        initialize_once()
        gemm_once()

    initialize_once()
    for _ in range(warmup_repetitions):
        gemm_once()
    torch.cuda.synchronize(device)

    # Constant, nonzero inputs are fully dense and make an inexpensive full
    # correctness check possible. Every output cell must equal N.
    output_correct = bool(torch.all(c == N).item())
    if not output_correct:
        sample = [
            int(c[0, 0].item()),
            int(c[N // 2, N // 2].item()),
            int(c[-1, -1].item()),
        ]
        raise RuntimeError(
            f"torch._int_mm correctness check failed: expected {N}, got {sample}"
        )

    def energy_millijoules() -> int:
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))

    def idle_measurement(seconds: float) -> dict:
        torch.cuda.synchronize(device)
        e0 = energy_millijoules()
        t0 = time.monotonic()
        time.sleep(seconds)
        e1 = energy_millijoules()
        if e1 < e0:
            raise RuntimeError(
                f"NVML energy counter moved backwards during idle: {e0} -> {e1}"
            )
        duration = time.monotonic() - t0
        raw_joules = (e1 - e0) / 1000.0
        return {
            "duration_s": duration,
            "raw_gpu_energy_J": raw_joules,
            "average_gpu_power_W": raw_joules / duration,
        }

    def timed_probe(workload, repetitions: int) -> float:
        torch.cuda.synchronize(device)
        t0 = time.monotonic()
        for _ in range(repetitions):
            workload()
        torch.cuda.synchronize(device)
        return time.monotonic() - t0

    def calibrated_repetitions(workload) -> tuple[int, dict]:
        # Double until the pilot is long enough to see stable launch and
        # steady-clock behavior, then project to the requested trial length.
        repetitions = 1
        duration = timed_probe(workload, repetitions)
        while duration < 0.5 and repetitions < 1_048_576:
            repetitions *= 2
            duration = timed_probe(workload, repetitions)
        projected = math.ceil(trial_seconds * repetitions / max(duration, 1e-9))
        projected = max(100, min(projected, 1_048_576))
        return projected, {
            "pilot_repetitions": repetitions,
            "pilot_duration_s": duration,
            "projected_repetitions": projected,
        }

    def energy_trial(workload, repetitions: int) -> dict:
        # Synchronization only at aggregate boundaries is deliberate: a sync
        # per GEMM would add bubbles and cease to represent normal throughput.
        torch.cuda.synchronize(device)
        e0 = energy_millijoules()
        t0 = time.monotonic()
        for _ in range(repetitions):
            workload()
        torch.cuda.synchronize(device)
        e1 = energy_millijoules()
        if e1 < e0:
            raise RuntimeError(
                f"NVML energy counter moved backwards during trial: {e0} -> {e1}"
            )
        duration = time.monotonic() - t0
        return {
            "repetitions": repetitions,
            "duration_s": duration,
            "raw_gpu_energy_J": (e1 - e0) / 1000.0,
        }

    # Let clocks return to their unloaded state before measuring the baseline.
    time.sleep(idle_cooldown_seconds)
    idle_before = idle_measurement(idle_seconds)

    # Use one repeat count for GEMM-only and pipeline trials, selected from
    # the slower pipeline, so their aggregate intervals are directly paired.
    init_repetitions, init_pilot = calibrated_repetitions(initialize_once)
    compute_repetitions, compute_pilot = calibrated_repetitions(pipeline_once)
    stage_specs = (
        ("initialization", initialize_once, init_repetitions),
        ("multiplication", gemm_once, compute_repetitions),
        ("pipeline", pipeline_once, compute_repetitions),
    )

    raw_stages: dict[str, dict] = {}
    for label, workload, repetitions in stage_specs:
        print(
            f"[8k-matmul] {label}: {trials} trial(s) x "
            f"{repetitions:,} repetitions",
            flush=True,
        )
        phase_trials = []
        for trial_index in range(trials):
            sample = energy_trial(workload, repetitions)
            sample["trial"] = trial_index + 1
            phase_trials.append(sample)
            per_run = sample["raw_gpu_energy_J"] / repetitions
            print(
                f"  trial {trial_index + 1}: {sample['duration_s']:.3f} s, "
                f"{sample['raw_gpu_energy_J']:.3f} J total, "
                f"{per_run:.6f} J/run",
                flush=True,
            )
        raw_stages[label] = {
            "repetitions_per_trial": repetitions,
            "trials": phase_trials,
        }

    time.sleep(idle_cooldown_seconds)
    idle_after = idle_measurement(idle_seconds)
    idle_samples_W = [
        idle_before["average_gpu_power_W"],
        idle_after["average_gpu_power_W"],
    ]
    idle_watts = statistics.mean(idle_samples_W)
    idle_watts_low = min(idle_samples_W)
    idle_watts_high = max(idle_samples_W)

    def summarize_stage(label: str, raw: dict) -> dict:
        samples = raw["trials"]
        repetitions = raw["repetitions_per_trial"]
        total_repetitions = repetitions * len(samples)
        duration = sum(item["duration_s"] for item in samples)
        raw_joules = sum(item["raw_gpu_energy_J"] for item in samples)
        idle_joules = idle_watts * duration
        net_joules = max(0.0, raw_joules - idle_joules)
        net_joules_low = max(0.0, raw_joules - idle_watts_high * duration)
        net_joules_high = max(0.0, raw_joules - idle_watts_low * duration)
        per_run_raw = [item["raw_gpu_energy_J"] / repetitions for item in samples]
        per_run_net = [
            max(
                0.0,
                item["raw_gpu_energy_J"] - idle_watts * item["duration_s"],
            )
            / repetitions
            for item in samples
        ]
        seconds_per_run = duration / total_repetitions
        summary = {
            "label": label,
            "repetitions_per_trial": repetitions,
            "trial_count": len(samples),
            "total_repetitions": total_repetitions,
            "total_duration_s": duration,
            "total_raw_gpu_energy_J": raw_joules,
            "total_idle_baseline_energy_J": idle_joules,
            "total_idle_adjusted_gpu_energy_J": net_joules,
            "seconds_per_run": seconds_per_run,
            "raw_gpu_energy_J_per_run": raw_joules / total_repetitions,
            "idle_adjusted_gpu_energy_J_per_run": net_joules / total_repetitions,
            "idle_adjusted_gpu_energy_J_per_run_baseline_range": [
                net_joules_low / total_repetitions,
                net_joules_high / total_repetitions,
            ],
            "raw_gpu_energy_J_per_run_trial_stdev": (
                statistics.stdev(per_run_raw) if len(per_run_raw) > 1 else 0.0
            ),
            "idle_adjusted_gpu_energy_J_per_run_trial_stdev": (
                statistics.stdev(per_run_net) if len(per_run_net) > 1 else 0.0
            ),
            "average_gpu_power_W": raw_joules / duration,
            "trials": samples,
        }
        if label == "initialization":
            summary["initialized_bytes_per_run"] = INPUT_BYTES
            summary["effective_initialization_bandwidth_GB_per_s"] = (
                INPUT_BYTES / seconds_per_run / 1e9
            )
        else:
            summary["conventional_operations_per_run"] = CONVENTIONAL_OPS
            summary["effective_throughput_TOPS"] = (
                CONVENTIONAL_OPS / seconds_per_run / 1e12
            )
        return summary

    stages = {
        label: summarize_stage(label, raw_stages[label]) for label in raw_stages
    }
    initialization = stages["initialization"]
    multiplication = stages["multiplication"]
    pipeline = stages["pipeline"]

    isolated_raw_sum = (
        initialization["raw_gpu_energy_J_per_run"]
        + multiplication["raw_gpu_energy_J_per_run"]
    )
    isolated_net_sum = (
        initialization["idle_adjusted_gpu_energy_J_per_run"]
        + multiplication["idle_adjusted_gpu_energy_J_per_run"]
    )
    direct_raw = pipeline["raw_gpu_energy_J_per_run"]
    direct_net = pipeline["idle_adjusted_gpu_energy_J_per_run"]
    breakdown = {
        "direct_pipeline_raw_gpu_energy_J_per_run": direct_raw,
        "direct_pipeline_idle_adjusted_gpu_energy_J_per_run": direct_net,
        "isolated_stage_raw_sum_J_per_run": isolated_raw_sum,
        "isolated_stage_idle_adjusted_sum_J_per_run": isolated_net_sum,
        "raw_direct_minus_multiplication_J_per_run": (
            direct_raw - multiplication["raw_gpu_energy_J_per_run"]
        ),
        "idle_adjusted_direct_minus_multiplication_J_per_run": (
            direct_net
            - multiplication["idle_adjusted_gpu_energy_J_per_run"]
        ),
        "raw_initialization_fraction_of_isolated_sum": (
            initialization["raw_gpu_energy_J_per_run"] / isolated_raw_sum
        ),
        "idle_adjusted_initialization_fraction_of_isolated_sum": (
            initialization["idle_adjusted_gpu_energy_J_per_run"]
            / isolated_net_sum
            if isolated_net_sum > 0
            else None
        ),
    }

    gpu_name = pynvml.nvmlDeviceGetName(handle)
    driver_version = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    if isinstance(driver_version, bytes):
        driver_version = driver_version.decode()

    result = {
        "benchmark": "full 8192^3 dense INT8 GEMM",
        "date_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "matrix": {
            "m": N,
            "n": N,
            "k": N,
            "lhs_dtype": "int8",
            "rhs_dtype": "int8",
            "output_dtype": "int32",
            "lhs_stride": [int(value) for value in a.stride()],
            "rhs_stride": [int(value) for value in b.stride()],
            "output_stride": [int(value) for value in c.stride()],
            "input_bytes": INPUT_BYTES,
            "output_bytes": OUTPUT_BYTES,
            "initialization": "preallocated A and B filled with constant 1",
        },
        "operation_count": {
            "multiply_accumulates": N**3,
            "conventional_ops_2mnk": CONVENTIONAL_OPS,
            "conventional_teraops": CONVENTIONAL_OPS / 1e12,
        },
        "hardware": {
            "requested_modal_gpu": requested_modal_gpu,
            "gpu_name": gpu_name,
            "gpu_memory_bytes": int(props.total_memory),
            "multiprocessor_count": int(props.multi_processor_count),
            "nvml_power_limit_W": (
                pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            ),
            "driver_version": driver_version,
            # torch.__version__ is a TorchVersion object (a str subclass)
            # whose pickle requires torch in the local client environment.
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "image_ref": IMAGE_REF,
        },
        "measurement": {
            "counter": "nvmlDeviceGetTotalEnergyConsumption",
            "counter_unit": "millijoules",
            "scope": "GPU board energy only",
            "trial_target_seconds": trial_seconds,
            "trial_count": trials,
            "warmup_repetitions": warmup_repetitions,
            "synchronization": "once before and once after each aggregate trial",
            "idle_before": idle_before,
            "idle_after": idle_after,
            "idle_baseline_W": idle_watts,
            "idle_baseline_range_W": [idle_watts_low, idle_watts_high],
            "idle_cooldown_seconds": idle_cooldown_seconds,
            "initialization_pilot": init_pilot,
            "compute_pipeline_pilot": compute_pilot,
            "excluded": [
                "Modal image pull and worker boot",
                "CUDA context creation",
                "device-buffer allocation",
                "warmup and correctness validation",
                "host CPU and facility energy",
            ],
        },
        "correctness": {
            "method": "torch.all(C == 8192) after warmup",
            "passed": output_correct,
        },
        "stages": stages,
        "pipeline_breakdown": breakdown,
    }
    pynvml.nvmlShutdown()
    return result


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _print_summary(result: dict) -> None:
    print("\n8192^3 dense INT8 GEMM")
    print(f"  GPU: {result['hardware']['gpu_name']}")
    print(
        "  conventional operations: "
        f"{result['operation_count']['conventional_ops_2mnk']:,}"
    )
    print(
        f"  measured idle baseline: "
        f"{result['measurement']['idle_baseline_W']:.2f} W"
    )
    print("\n  phase             runs      ms/run    raw J/run    net J/run")
    for key in ("initialization", "multiplication", "pipeline"):
        stage = result["stages"][key]
        print(
            f"  {key:<14} "
            f"{stage['total_repetitions']:>9,}  "
            f"{stage['seconds_per_run'] * 1e3:>10.4f}  "
            f"{stage['raw_gpu_energy_J_per_run']:>11.6f}  "
            f"{stage['idle_adjusted_gpu_energy_J_per_run']:>11.6f}"
        )
    multiply = result["stages"]["multiplication"]
    print(
        f"\n  GEMM throughput: {multiply['effective_throughput_TOPS']:.2f} TOPS"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--trial-seconds",
        type=_positive_float,
        default=15.0,
        help="target duration of each aggregate trial (default: 15)",
    )
    parser.add_argument(
        "--trials",
        type=_positive_int,
        default=3,
        help="aggregate trials per phase (default: 3)",
    )
    parser.add_argument(
        "--idle-seconds",
        type=_positive_float,
        default=10.0,
        help="duration of each before/after idle calibration (default: 10)",
    )
    parser.add_argument(
        "--idle-cooldown-seconds",
        type=float,
        default=10.0,
        help="unmeasured cooldown before each idle calibration (default: 10)",
    )
    parser.add_argument(
        "--warmup-repetitions",
        type=_positive_int,
        default=10,
        help="unmeasured GEMMs before validation and measurement (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "8k_matmul_results.json",
        help="local JSON result path",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the Modal cost confirmation"
    )
    args = parser.parse_args()

    if args.idle_cooldown_seconds < 0:
        parser.error("--idle-cooldown-seconds must be nonnegative")
    measured_seconds = (
        3 * args.trials * args.trial_seconds
        + 2 * args.idle_seconds
        + 2 * args.idle_cooldown_seconds
    )
    print(f"╭─ Modal {MODAL_GPU} 8192^3 INT8 GEMM ──────")
    print(f"│  aggregate measurement: at least ~{measured_seconds:.0f} s")
    print("│  plus image startup, pilots, warmup, and validation")
    print(f"│  output: {args.output.resolve()}")
    print("╰───────────────────────────────────────")
    if not args.yes and input("proceed? [Y/n] ").strip().lower() not in ("", "y", "yes"):
        print("aborted")
        return 1

    with modal.enable_output(), app.run():
        result = run_benchmark.remote(
            trial_seconds=args.trial_seconds,
            trials=args.trials,
            idle_seconds=args.idle_seconds,
            idle_cooldown_seconds=args.idle_cooldown_seconds,
            warmup_repetitions=args.warmup_repetitions,
            requested_modal_gpu=MODAL_GPU,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    _print_summary(result)
    print(f"\nwrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
