#!/usr/bin/env python3
"""Measure an 8192^3 dense FP8 GEMM on a Modal NVIDIA B200.

The benchmark measures three steady-state phases independently:

* initialization: fill two preallocated 8192 x 8192 FP8 input buffers;
* multiplication: FP8 E4M3FN x FP8 E4M3FN -> FP32 GEMM into a
  preallocated output;
* pipeline: both input fills followed by the GEMM.

Each NVML energy-counter interval contains many asynchronous launches and
only one synchronization at either boundary. This makes a sub-millisecond
operation large enough to measure reliably without introducing a host/device
bubble after every launch.

The Modal GPU is deliberately fixed to ``B200`` rather than ``B200+`` so a
B300 upgrade cannot silently change the hardware being compared. Setup and
usage mirror ``submit.py``::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    modal token new
    python benchmark_8k_fp8_matmul.py --yes

The default run writes ``8k_fp8_blackwell_results.json`` next to this file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
N = 8192
CONVENTIONAL_OPS = 2 * N**3
INPUT_BYTES = 2 * N**2  # two one-byte FP8 inputs
OUTPUT_BYTES = 4 * N**2  # one FP32 output

# Keep this exact. Modal's B200+ selector may upgrade the worker to B300.
MODAL_GPU = "B200"
USE_FAST_ACCUM = False
MAX_PLANNED_MEASUREMENT_SECONDS = 210.0
MIN_TRIAL_SECONDS = 0.1
MAX_TRIALS = 10
MAX_WARMUP_REPETITIONS = 1_000
EXECUTION_CAP_SECONDS = 5 * 60
STARTUP_CAP_SECONDS = 3 * 60
CALL_WALL_CAP_SECONDS = 8 * 60

# Use the exact Blackwell-capable PyTorch/CUDA environment verified by the
# pilot. Package versions and the CUDA wheel index are pinned explicitly.
IMAGE_REF = "modal.Image.debian_slim(python_version='3.11')"
TORCH_PACKAGE = "torch==2.12.1"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
PYNVML_PACKAGE = "nvidia-ml-py==13.580.82"
NUMPY_PACKAGE = "numpy==2.3.5"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(TORCH_PACKAGE, index_url=TORCH_INDEX_URL)
    .uv_pip_install(PYNVML_PACKAGE, NUMPY_PACKAGE)
    .workdir("/workspace")
)
app = modal.App("wikitext-8k-fp8-matmul")

REFERENCE_URL = (
    "https://chatgpt.com/share/6a922e47-aae0-83e8-81f4-d72ddf648a11"
)
REFERENCE_ACCELERATOR = "NVIDIA B200"
REFERENCE_DENSE_FP8_OPS_PER_S = 4.50e15
REFERENCE_POWER_W = 1000.0
REFERENCE_IDEAL_TIME_S = CONVENTIONAL_OPS / REFERENCE_DENSE_FP8_OPS_PER_S
REFERENCE_MODULE_ENERGY_J = REFERENCE_POWER_W * REFERENCE_IDEAL_TIME_S


@app.function(
    image=image,
    gpu=MODAL_GPU,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
    single_use_containers=True,
    retries=0,
    timeout=EXECUTION_CAP_SECONDS,
    startup_timeout=STARTUP_CAP_SECONDS,
)
def run_benchmark(
    *,
    trial_seconds: float = 15.0,
    trials: int = 3,
    idle_seconds: float = 10.0,
    idle_cooldown_seconds: float = 10.0,
    warmup_repetitions: int = 10,
    requested_modal_gpu: str = MODAL_GPU,
) -> dict:
    """Run the benchmark on one Modal B200 and return a JSON-ready result."""
    import math
    import statistics
    import time
    from datetime import datetime, timezone

    import pynvml
    import torch

    if not math.isfinite(trial_seconds) or trial_seconds < MIN_TRIAL_SECONDS:
        raise ValueError(f"trial_seconds must be at least {MIN_TRIAL_SECONDS}")
    if not 1 <= trials <= MAX_TRIALS:
        raise ValueError(f"trials must be between 1 and {MAX_TRIALS}")
    if not math.isfinite(idle_seconds) or idle_seconds <= 0:
        raise ValueError("idle_seconds must be finite and positive")
    if (
        not math.isfinite(idle_cooldown_seconds)
        or idle_cooldown_seconds < 0
    ):
        raise ValueError("idle_cooldown_seconds must be finite and nonnegative")
    if not 1 <= warmup_repetitions <= MAX_WARMUP_REPETITIONS:
        raise ValueError(
            "warmup_repetitions must be between 1 and "
            f"{MAX_WARMUP_REPETITIONS}"
        )
    planned_seconds = (
        3 * trials * trial_seconds
        + 2 * idle_seconds
        + 2 * idle_cooldown_seconds
    )
    if planned_seconds > MAX_PLANNED_MEASUREMENT_SECONDS:
        raise ValueError(
            f"measurement plan is {planned_seconds:.1f} s; the cost guard "
            f"allows at most {MAX_PLANNED_MEASUREMENT_SECONDS:.0f} s"
        )
    if requested_modal_gpu != MODAL_GPU:
        raise ValueError(
            f"this benchmark is fixed to Modal {MODAL_GPU}, got "
            f"{requested_modal_gpu!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the Modal worker")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("this PyTorch build does not expose float8_e4m3fn")
    try:
        scaled_mm_out = torch.ops.aten._scaled_mm.out
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "this PyTorch build does not expose aten::_scaled_mm.out"
        ) from exc

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # Probe before spending time on the benchmark. Modal must expose the
    # cumulative board-energy counter used by the wikitext harness.
    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    compute_capability = torch.cuda.get_device_capability(device)
    if compute_capability != (10, 0):
        raise RuntimeError(
            "expected a B200 with CUDA compute capability 10.0, got "
            f"{compute_capability[0]}.{compute_capability[1]} "
            f"({props.name})"
        )

    fp8_dtype = torch.float8_e4m3fn

    # A is row-major. B is a transposed view over row-major storage, giving
    # the column-major RHS accepted by the cuBLASLt scaled-GEMM path. All
    # buffers and scalar scales are allocated before warmup. Measured
    # initialization means writing the matrix entries, not allocator startup.
    a = torch.empty((N, N), dtype=fp8_dtype, device=device)
    b_storage = torch.empty((N, N), dtype=fp8_dtype, device=device)
    b = b_storage.t()
    c = torch.empty((N, N), dtype=torch.float32, device=device)
    scale_a = torch.ones((1,), dtype=torch.float32, device=device)
    scale_b = torch.ones((1,), dtype=torch.float32, device=device)

    # Float8 is a shell dtype in PyTorch, so fill the underlying bytes. Derive
    # the representation at runtime and assert the standard E4M3 encoding of
    # 1.0. This is a full 128 MiB device write on every initialization.
    one_fp8 = torch.tensor([1.0], dtype=torch.float32, device=device).to(fp8_dtype)
    one_bits = int(one_fp8.view(torch.uint8).item())
    if one_bits != 0x38:
        raise RuntimeError(
            f"unexpected float8_e4m3fn representation for 1.0: 0x{one_bits:02x}"
        )
    a_bytes = a.view(torch.uint8)
    b_storage_bytes = b_storage.view(torch.uint8)

    def initialize_once() -> None:
        a_bytes.fill_(one_bits)
        b_storage_bytes.fill_(one_bits)

    def gemm_once() -> None:
        # The out overload keeps the 256 MiB FP32 result buffer preallocated.
        # Keep full accumulation enabled. The B200 pilot found this path at
        # least as fast as fast-accum while retaining its better accuracy.
        scaled_mm_out(
            a,
            b,
            scale_a,
            scale_b,
            bias=None,
            scale_result=None,
            out_dtype=torch.float32,
            use_fast_accum=USE_FAST_ACCUM,
            out=c,
        )

    def pipeline_once() -> None:
        initialize_once()
        gemm_once()

    initialize_once()
    for _ in range(warmup_repetitions):
        gemm_once()
    torch.cuda.synchronize(device)

    # The FP8 inputs and FP32 output exactly represent this constant-input
    # result. Check every one of the 67,108,864 output entries.
    output_correct = bool(torch.all(c == float(N)).item())
    if not output_correct:
        sample = [
            float(c[0, 0].item()),
            float(c[N // 2, N // 2].item()),
            float(c[-1, -1].item()),
        ]
        raise RuntimeError(
            f"FP8 scaled GEMM correctness check failed: expected {N}, got {sample}"
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
            f"[8k-fp8-matmul] {label}: {trials} trial(s) x "
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
            summary["effective_throughput_TFLOP_per_s"] = (
                CONVENTIONAL_OPS / seconds_per_run / 1e12
            )
            summary["effective_throughput_PFLOP_per_s"] = (
                CONVENTIONAL_OPS / seconds_per_run / 1e15
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

    measured_multiply_s = multiplication["seconds_per_run"]
    measured_multiply_raw_J = multiplication["raw_gpu_energy_J_per_run"]
    comparison = {
        "like_for_like_scope": (
            "isolated multiplication raw board energy; the reference excludes "
            "input initialization and uses whole-module peak power"
        ),
        "measured_multiplication_seconds_per_run": measured_multiply_s,
        "measured_multiplication_raw_gpu_energy_J_per_run": (
            measured_multiply_raw_J
        ),
        "measured_time_over_reference_ideal_ratio": (
            measured_multiply_s / REFERENCE_IDEAL_TIME_S
        ),
        "measured_time_delta_from_reference_percent": (
            (measured_multiply_s / REFERENCE_IDEAL_TIME_S - 1.0) * 100.0
        ),
        "measured_raw_energy_over_reference_ratio": (
            measured_multiply_raw_J / REFERENCE_MODULE_ENERGY_J
        ),
        "measured_raw_energy_delta_from_reference_percent": (
            (measured_multiply_raw_J / REFERENCE_MODULE_ENERGY_J - 1.0)
            * 100.0
        ),
        "measured_throughput_over_reference_peak_ratio": (
            multiplication["effective_throughput_PFLOP_per_s"]
            / (REFERENCE_DENSE_FP8_OPS_PER_S / 1e15)
        ),
        "direct_pipeline_raw_gpu_energy_J_per_run": direct_raw,
        "direct_pipeline_raw_over_prepacked_reference_ratio": (
            direct_raw / REFERENCE_MODULE_ENERGY_J
        ),
        "direct_pipeline_comparison_caveat": (
            "not like-for-like: the measured pipeline includes two 64 MiB "
            "input fills, while the reference assumes prepacked inputs"
        ),
    }

    gpu_name = pynvml.nvmlDeviceGetName(handle)
    driver_version = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    if isinstance(driver_version, bytes):
        driver_version = driver_version.decode()

    result = {
        "benchmark": "full 8192^3 dense FP8 E4M3FN GEMM",
        "date_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "matrix": {
            "m": N,
            "n": N,
            "k": N,
            "lhs_dtype": "float8_e4m3fn",
            "rhs_dtype": "float8_e4m3fn",
            "accumulation_dtype": "float32",
            "output_dtype": "float32",
            "lhs_stride": [int(value) for value in a.stride()],
            "rhs_stride": [int(value) for value in b.stride()],
            "output_stride": [int(value) for value in c.stride()],
            "input_bytes": INPUT_BYTES,
            "output_bytes": OUTPUT_BYTES,
            "logical_minimum_traffic_bytes": INPUT_BYTES + OUTPUT_BYTES,
            "initialization": (
                "preallocated A and B raw FP8 storage filled with the E4M3FN "
                "encoding of constant 1.0"
            ),
            "constant_one_fp8_bits_hex": f"0x{one_bits:02x}",
            "scale_a": 1.0,
            "scale_b": 1.0,
        },
        "operation_count": {
            "multiply_accumulates": N**3,
            "conventional_ops_2mnk": CONVENTIONAL_OPS,
            "conventional_teraops": CONVENTIONAL_OPS / 1e12,
        },
        "operator": {
            "name": "aten::_scaled_mm.out",
            "output_preallocated": True,
            "tensorwise_fp32_scales": True,
            "use_fast_accum": USE_FAST_ACCUM,
        },
        "hardware": {
            "requested_modal_gpu": requested_modal_gpu,
            "gpu_name": gpu_name,
            "gpu_memory_bytes": int(props.total_memory),
            "multiprocessor_count": int(props.multi_processor_count),
            "compute_capability": [
                int(compute_capability[0]),
                int(compute_capability[1]),
            ],
            "nvml_power_limit_W": (
                pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            ),
            "driver_version": driver_version,
            # torch.__version__ is a TorchVersion object (a str subclass)
            # whose pickle requires torch in the local client environment.
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "torch_cuda_arch_list": list(torch.cuda.get_arch_list()),
            "image_ref": IMAGE_REF,
            "torch_package": TORCH_PACKAGE,
            "torch_index_url": TORCH_INDEX_URL,
            "pynvml_package": PYNVML_PACKAGE,
            "numpy_package": NUMPY_PACKAGE,
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
                "device-buffer and scale allocation",
                "warmup and correctness validation",
                "host CPU and facility energy",
            ],
        },
        "cost_controls": {
            "exact_gpu_selector": MODAL_GPU,
            "min_containers": 0,
            "max_containers": 1,
            "buffer_containers": 0,
            "scaledown_window_seconds": 2,
            "single_use_containers": True,
            "retries": 0,
            "execution_cap_seconds": EXECUTION_CAP_SECONDS,
            "startup_cap_seconds": STARTUP_CAP_SECONDS,
            "caller_wall_cap_seconds": CALL_WALL_CAP_SECONDS,
            "max_planned_measurement_seconds": (
                MAX_PLANNED_MEASUREMENT_SECONDS
            ),
        },
        "correctness": {
            "method": "torch.all(C == 8192.0) after warmup",
            "checked_output_entries": N**2,
            "passed": output_correct,
        },
        "stages": stages,
        "pipeline_breakdown": breakdown,
        "reference_estimate": {
            "source_url": REFERENCE_URL,
            "table": "One full 8192^3 dense FP8 GEMM",
            "accelerator": REFERENCE_ACCELERATOR,
            "dense_fp8_rate_ops_per_s": REFERENCE_DENSE_FP8_OPS_PER_S,
            "power_basis_W": REFERENCE_POWER_W,
            "reported_ideal_time_ms_rounded": 0.244,
            "reported_accelerator_module_energy_J_rounded": 0.244,
            "recomputed_ideal_time_s": REFERENCE_IDEAL_TIME_S,
            "recomputed_accelerator_module_energy_J": (
                REFERENCE_MODULE_ENERGY_J
            ),
            "calculation": "energy = power * conventional_ops / dense_rate",
            "scope": (
                "warm repeated GEMM with prepacked matrices in device HBM; "
                "whole accelerator-module power; initialization, host, cooling, "
                "and facility overhead excluded"
            ),
            "nature": "engineering peak-power estimate, not a measurement",
        },
        "estimate_comparison": comparison,
    }
    pynvml.nvmlShutdown()
    return result


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _print_summary(result: dict) -> None:
    print("\n8192^3 dense FP8 E4M3FN GEMM")
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
    comparison = result["estimate_comparison"]
    estimate = result["reference_estimate"]
    print(
        f"\n  GEMM throughput: "
        f"{multiply['effective_throughput_PFLOP_per_s']:.3f} PFLOP/s"
    )
    print(
        "  raw GEMM energy vs reference: "
        f"{multiply['raw_gpu_energy_J_per_run']:.6f} J / "
        f"{estimate['recomputed_accelerator_module_energy_J']:.6f} J = "
        f"{comparison['measured_raw_energy_over_reference_ratio']:.3f}x"
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
        default=HERE / "8k_fp8_blackwell_results.json",
        help="local JSON result path",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the Modal cost confirmation"
    )
    args = parser.parse_args()

    if (
        not math.isfinite(args.idle_cooldown_seconds)
        or args.idle_cooldown_seconds < 0
    ):
        parser.error("--idle-cooldown-seconds must be finite and nonnegative")
    if args.trial_seconds < MIN_TRIAL_SECONDS:
        parser.error(f"--trial-seconds must be at least {MIN_TRIAL_SECONDS}")
    if args.trials > MAX_TRIALS:
        parser.error(f"--trials cannot exceed {MAX_TRIALS}")
    if args.warmup_repetitions > MAX_WARMUP_REPETITIONS:
        parser.error(
            f"--warmup-repetitions cannot exceed {MAX_WARMUP_REPETITIONS}"
        )
    measured_seconds = (
        3 * args.trials * args.trial_seconds
        + 2 * args.idle_seconds
        + 2 * args.idle_cooldown_seconds
    )
    if measured_seconds > MAX_PLANNED_MEASUREMENT_SECONDS:
        parser.error(
            f"measurement plan is {measured_seconds:.1f} s; the cost guard "
            f"allows at most {MAX_PLANNED_MEASUREMENT_SECONDS:.0f} s"
        )
    print(f"╭─ Modal {MODAL_GPU} 8192^3 FP8 GEMM ──────")
    print(f"│  aggregate measurement: at least ~{measured_seconds:.0f} s")
    print("│  one container, one input, no configured retries")
    print(f"│  hard function cap: {EXECUTION_CAP_SECONDS} s")
    print(f"│  output: {args.output.resolve()}")
    print("╰───────────────────────────────────────")
    if not args.yes and input("proceed? [Y/n] ").strip().lower() not in (
        "",
        "y",
        "yes",
    ):
        print("aborted")
        return 1

    call = None
    completed = False
    with modal.enable_output(), app.run(detach=False):
        print(
            f"emergency stop: modal app stop -y {app.app_id}",
            file=sys.stderr,
            flush=True,
        )
        call = run_benchmark.spawn(
            trial_seconds=args.trial_seconds,
            trials=args.trials,
            idle_seconds=args.idle_seconds,
            idle_cooldown_seconds=args.idle_cooldown_seconds,
            warmup_repetitions=args.warmup_repetitions,
            requested_modal_gpu=MODAL_GPU,
        )
        try:
            result = call.get(timeout=CALL_WALL_CAP_SECONDS)
            completed = True
        finally:
            if not completed:
                call.cancel(terminate_containers=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    _print_summary(result)
    print(f"\nwrote {args.output.resolve()}")
    print("Modal app exited; the single-use container is not kept warm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
