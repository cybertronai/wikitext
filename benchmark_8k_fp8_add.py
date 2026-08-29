#!/usr/bin/env python3
"""Measure out-of-place addition of two dense 8192 x 8192 FP8 matrices.

The measured operation is::

    C_fp8 = round_to_e4m3fn(float32(A_fp8) + float32(B_fp8))

All three buffers are preallocated on one Modal NVIDIA B200. The benchmark
measures input initialization, addition, and the direct initialization-plus-add
pipeline independently. Thousands of asynchronous launches are amortized in
each NVML interval, with synchronization only at aggregate boundaries.

The Modal function is intentionally single-use, allows only one container,
does not retry, and has tight execution and caller timeouts. The default run
writes ``8k_fp8_add_blackwell_results.json`` next to this file.
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
ELEMENTS = N**2
ELEMENTWISE_ADDITIONS = ELEMENTS
INPUT_BYTES = 2 * ELEMENTS  # two one-byte FP8 inputs
OUTPUT_BYTES = ELEMENTS  # one one-byte FP8 output
ADD_LOGICAL_TRAFFIC_BYTES = INPUT_BYTES + OUTPUT_BYTES
PIPELINE_LOGICAL_TRAFFIC_BYTES = INPUT_BYTES + ADD_LOGICAL_TRAFFIC_BYTES

MODAL_GPU = "B200"
TRITON_BLOCK_SIZE = 1024
TRITON_NUM_WARPS = 8

# A short, one-shot measurement is enough because each aggregate interval
# contains tens of thousands of adds and hundreds of joules of board energy.
DEFAULT_TRIAL_SECONDS = 2.0
DEFAULT_TRIALS = 3
DEFAULT_IDLE_SECONDS = 3.0
DEFAULT_IDLE_COOLDOWN_SECONDS = 2.0
DEFAULT_WARMUP_REPETITIONS = 100
MAX_PLANNED_MEASUREMENT_SECONDS = 60.0
MIN_TRIAL_SECONDS = 0.1
MAX_TRIALS = 10
MAX_WARMUP_REPETITIONS = 1_000
MAX_CALIBRATED_REPETITIONS = 1_048_576
EXECUTION_CAP_SECONDS = 3 * 60
STARTUP_CAP_SECONDS = 3 * 60
CALL_WALL_CAP_SECONDS = 6 * 60

IMAGE_DESCRIPTION = "modal.Image.debian_slim(python_version='3.11')"
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
app = modal.App("wikitext-8k-fp8-add")

# A simple B200 HBM roofline proxy. This is deliberately separate from the
# Sutro model: it includes the logical byte width but still assumes ideal
# bandwidth and peak whole-board power.
B200_HBM_BANDWIDTH_BYTES_PER_S = 8.0e12
B200_POWER_BASIS_W = 1000.0
ROOFLINE_ADD_TIME_S = (
    ADD_LOGICAL_TRAFFIC_BYTES / B200_HBM_BANDWIDTH_BYTES_PER_S
)
ROOFLINE_ADD_ENERGY_J = B200_POWER_BASIS_W * ROOFLINE_ADD_TIME_S
ROOFLINE_PIPELINE_TIME_S = (
    PIPELINE_LOGICAL_TRAFFIC_BYTES / B200_HBM_BANDWIDTH_BYTES_PER_S
)
ROOFLINE_PIPELINE_ENERGY_J = B200_POWER_BASIS_W * ROOFLINE_PIPELINE_TIME_S

# Original Sutro scorer: every paid source/final-output read at positive
# address a costs ceil(sqrt(a)). At 1 fJ per unit, a direct out-of-place add
# reads each of the 3*N^2 contiguous cells exactly once.
SUTRO_REPORT_URL = (
    "https://cybertronai.github.io/sutro-problems/matmul/energy-report/"
    "#eightk"
)
SUTRO_ENERGY_J_PER_SCORE_UNIT = 1.0e-15


def _sutro_prefix_score(last_address: int) -> int:
    """Return sum(ceil(sqrt(a)) for a in 1..last_address) exactly."""
    if last_address < 0:
        raise ValueError("last_address must be nonnegative")
    q = math.isqrt(last_address)
    return (
        q * (q + 1) * (4 * q - 1) // 6
        + (last_address - q**2) * (q + 1)
    )


SUTRO_OUT_OF_PLACE_SCORE = _sutro_prefix_score(3 * ELEMENTS)
SUTRO_OUT_OF_PLACE_ENERGY_J = (
    SUTRO_OUT_OF_PLACE_SCORE * SUTRO_ENERGY_J_PER_SCORE_UNIT
)
SUTRO_IN_PLACE_SCORE = _sutro_prefix_score(ELEMENTS) + _sutro_prefix_score(
    2 * ELEMENTS
)
SUTRO_IN_PLACE_ENERGY_J = (
    SUTRO_IN_PLACE_SCORE * SUTRO_ENERGY_J_PER_SCORE_UNIT
)


def _planned_measurement_seconds(
    trial_seconds: float,
    trials: int,
    idle_seconds: float,
    idle_cooldown_seconds: float,
) -> float:
    return (
        3 * trials * trial_seconds
        + 2 * idle_seconds
        + 2 * idle_cooldown_seconds
    )


def _validate_plan(
    *,
    trial_seconds: float,
    trials: int,
    idle_seconds: float,
    idle_cooldown_seconds: float,
    warmup_repetitions: int,
) -> float:
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
    planned = _planned_measurement_seconds(
        trial_seconds, trials, idle_seconds, idle_cooldown_seconds
    )
    if planned > MAX_PLANNED_MEASUREMENT_SECONDS:
        raise ValueError(
            f"measurement plan is {planned:.1f} s; the cost guard allows at "
            f"most {MAX_PLANNED_MEASUREMENT_SECONDS:.0f} s"
        )
    return planned


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
    trial_seconds: float = DEFAULT_TRIAL_SECONDS,
    trials: int = DEFAULT_TRIALS,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS,
    warmup_repetitions: int = DEFAULT_WARMUP_REPETITIONS,
) -> dict:
    """Run one cost-bounded benchmark input on one Modal B200."""
    import statistics
    import time
    from datetime import datetime, timezone

    import pynvml
    import torch
    import triton
    import triton.language as tl

    planned_seconds = _validate_plan(
        trial_seconds=trial_seconds,
        trials=trials,
        idle_seconds=idle_seconds,
        idle_cooldown_seconds=idle_cooldown_seconds,
        warmup_repetitions=warmup_repetitions,
    )
    remote_started = time.monotonic()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the Modal worker")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("this PyTorch build does not expose float8_e4m3fn")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    compute_capability = torch.cuda.get_device_capability(device)
    if compute_capability != (10, 0):
        raise RuntimeError(
            "expected a B200 with CUDA compute capability 10.0, got "
            f"{compute_capability[0]}.{compute_capability[1]} ({props.name})"
        )

    @triton.jit
    def fp8_add_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        n_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        a_values = tl.load(a_ptr + offsets, mask=mask).to(tl.float32)
        b_values = tl.load(b_ptr + offsets, mask=mask).to(tl.float32)
        result = (a_values + b_values).to(
            tl.float8e4nv, fp_downcast_rounding="rtne"
        )
        tl.store(c_ptr + offsets, result, mask=mask)

    fp8_dtype = torch.float8_e4m3fn
    a = torch.empty((N, N), dtype=fp8_dtype, device=device)
    b = torch.empty((N, N), dtype=fp8_dtype, device=device)
    c = torch.empty((N, N), dtype=fp8_dtype, device=device)

    one_fp8 = torch.tensor([1.0], dtype=torch.float32, device=device).to(fp8_dtype)
    two_fp8 = torch.tensor([2.0], dtype=torch.float32, device=device).to(fp8_dtype)
    one_bits = int(one_fp8.view(torch.uint8).item())
    two_bits = int(two_fp8.view(torch.uint8).item())
    if one_bits != 0x38 or two_bits != 0x40:
        raise RuntimeError(
            "unexpected E4M3FN encodings: "
            f"1.0=0x{one_bits:02x}, 2.0=0x{two_bits:02x}"
        )
    a_bytes = a.view(torch.uint8)
    b_bytes = b.view(torch.uint8)
    c_bytes = c.view(torch.uint8)

    def initialize_once() -> None:
        a_bytes.fill_(one_bits)
        b_bytes.fill_(one_bits)

    grid = (triton.cdiv(ELEMENTS, TRITON_BLOCK_SIZE),)

    def add_once() -> None:
        fp8_add_kernel[grid](
            a,
            b,
            c,
            ELEMENTS,
            block_size=TRITON_BLOCK_SIZE,
            num_warps=TRITON_NUM_WARPS,
        )

    def pipeline_once() -> None:
        initialize_once()
        add_once()

    initialize_once()
    for _ in range(warmup_repetitions):
        add_once()
    torch.cuda.synchronize(device)

    output_correct = bool(torch.all(c_bytes == two_bits).item())
    if not output_correct:
        sample = [
            float(c[0, 0].item()),
            float(c[N // 2, N // 2].item()),
            float(c[-1, -1].item()),
        ]
        raise RuntimeError(
            f"FP8 addition correctness check failed: expected 2.0, got {sample}"
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
        repetitions = 1
        duration = timed_probe(workload, repetitions)
        while duration < 0.25 and repetitions < MAX_CALIBRATED_REPETITIONS:
            repetitions *= 2
            duration = timed_probe(workload, repetitions)
        projected = math.ceil(trial_seconds * repetitions / max(duration, 1e-9))
        projected = max(100, min(projected, MAX_CALIBRATED_REPETITIONS))
        return projected, {
            "pilot_repetitions": repetitions,
            "pilot_duration_s": duration,
            "projected_repetitions": projected,
        }

    def energy_trial(workload, repetitions: int) -> dict:
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

    time.sleep(idle_cooldown_seconds)
    idle_before = idle_measurement(idle_seconds)

    stage_workloads = {
        "initialization": initialize_once,
        "addition": add_once,
        "pipeline": pipeline_once,
    }
    stage_pilots = {}
    stage_repetitions = {}
    for label, workload in stage_workloads.items():
        repetitions, pilot = calibrated_repetitions(workload)
        stage_repetitions[label] = repetitions
        stage_pilots[label] = pilot

    raw_stages = {
        label: {"repetitions_per_trial": stage_repetitions[label], "trials": []}
        for label in stage_workloads
    }
    # Interleave stage trials so mild temperature/clock drift affects each
    # phase across the run rather than only the final phase.
    for trial_index in range(trials):
        for label, workload in stage_workloads.items():
            repetitions = stage_repetitions[label]
            print(
                f"[8k-fp8-add] trial {trial_index + 1}/{trials} {label}: "
                f"{repetitions:,} repetitions",
                flush=True,
            )
            sample = energy_trial(workload, repetitions)
            sample["trial"] = trial_index + 1
            raw_stages[label]["trials"].append(sample)
            print(
                f"  {sample['duration_s']:.3f} s, "
                f"{sample['raw_gpu_energy_J']:.3f} J total, "
                f"{sample['raw_gpu_energy_J'] / repetitions:.6f} J/run",
                flush=True,
            )

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
        logical_bytes = {
            "initialization": INPUT_BYTES,
            "addition": ADD_LOGICAL_TRAFFIC_BYTES,
            "pipeline": PIPELINE_LOGICAL_TRAFFIC_BYTES,
        }[label]
        summary["logical_traffic_bytes_per_run"] = logical_bytes
        summary["effective_logical_bandwidth_GB_per_s"] = (
            logical_bytes / seconds_per_run / 1e9
        )
        if label != "initialization":
            summary["elementwise_additions_per_run"] = ELEMENTWISE_ADDITIONS
        return summary

    stages = {
        label: summarize_stage(label, raw_stages[label]) for label in raw_stages
    }
    initialization = stages["initialization"]
    addition = stages["addition"]
    pipeline = stages["pipeline"]
    isolated_raw_sum = (
        initialization["raw_gpu_energy_J_per_run"]
        + addition["raw_gpu_energy_J_per_run"]
    )
    isolated_net_sum = (
        initialization["idle_adjusted_gpu_energy_J_per_run"]
        + addition["idle_adjusted_gpu_energy_J_per_run"]
    )
    direct_raw = pipeline["raw_gpu_energy_J_per_run"]
    direct_net = pipeline["idle_adjusted_gpu_energy_J_per_run"]

    pipeline_breakdown = {
        "direct_pipeline_raw_gpu_energy_J_per_run": direct_raw,
        "direct_pipeline_idle_adjusted_gpu_energy_J_per_run": direct_net,
        "isolated_stage_raw_sum_J_per_run": isolated_raw_sum,
        "isolated_stage_idle_adjusted_sum_J_per_run": isolated_net_sum,
        "raw_direct_minus_addition_J_per_run": (
            direct_raw - addition["raw_gpu_energy_J_per_run"]
        ),
        "idle_adjusted_direct_minus_addition_J_per_run": (
            direct_net - addition["idle_adjusted_gpu_energy_J_per_run"]
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

    grid_comparison = {
        "like_for_like_stage": "addition",
        "measured_raw_gpu_energy_J_per_add": addition[
            "raw_gpu_energy_J_per_run"
        ],
        "measured_idle_adjusted_gpu_energy_J_per_add": addition[
            "idle_adjusted_gpu_energy_J_per_run"
        ],
        "measured_raw_over_sutro_out_of_place_ratio": (
            addition["raw_gpu_energy_J_per_run"] / SUTRO_OUT_OF_PLACE_ENERGY_J
        ),
        "measured_idle_adjusted_over_sutro_out_of_place_ratio": (
            addition["idle_adjusted_gpu_energy_J_per_run"]
            / SUTRO_OUT_OF_PLACE_ENERGY_J
        ),
        "measured_raw_over_hbm_roofline_ratio": (
            addition["raw_gpu_energy_J_per_run"] / ROOFLINE_ADD_ENERGY_J
        ),
        "direct_pipeline_raw_over_hbm_roofline_ratio": (
            direct_raw / ROOFLINE_PIPELINE_ENERGY_J
        ),
        "scope_warning": (
            "Sutro is a widthless source/output-read movement proxy. NVML is "
            "whole GPU-board energy and must not be added to the model value."
        ),
    }

    gpu_name = pynvml.nvmlDeviceGetName(handle)
    driver_version = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    if isinstance(driver_version, bytes):
        driver_version = driver_version.decode()

    result = {
        "benchmark": "out-of-place addition of two dense 8192 x 8192 FP8 matrices",
        "date_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "matrix": {
            "rows": N,
            "columns": N,
            "elements": ELEMENTS,
            "lhs_dtype": "float8_e4m3fn",
            "rhs_dtype": "float8_e4m3fn",
            "arithmetic_dtype": "float32",
            "output_dtype": "float8_e4m3fn",
            "rounding": "round to nearest, ties to even",
            "input_bytes": INPUT_BYTES,
            "output_bytes": OUTPUT_BYTES,
            "addition_logical_traffic_bytes": ADD_LOGICAL_TRAFFIC_BYTES,
            "pipeline_logical_traffic_bytes": PIPELINE_LOGICAL_TRAFFIC_BYTES,
            "initialization": "preallocated A and B filled with constant 1.0",
        },
        "operation_count": {
            "elementwise_additions": ELEMENTWISE_ADDITIONS,
        },
        "operator": {
            "name": "custom Triton FP8 elementwise add",
            "block_size": TRITON_BLOCK_SIZE,
            "num_warps": TRITON_NUM_WARPS,
            "output_preallocated": True,
            "source": (
                "FP8 loads -> FP32 add -> FP8 E4M3FN RTNE store"
            ),
        },
        "hardware": {
            "requested_modal_gpu": MODAL_GPU,
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
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "triton_version": str(triton.__version__),
            "image_description": IMAGE_DESCRIPTION,
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
            "planned_measurement_and_cooldown_seconds": planned_seconds,
            "warmup_repetitions": warmup_repetitions,
            "synchronization": "once before and once after each aggregate trial",
            "trial_order": "initialization, addition, pipeline; interleaved",
            "idle_before": idle_before,
            "idle_after": idle_after,
            "idle_baseline_W": idle_watts,
            "idle_baseline_range_W": [idle_watts_low, idle_watts_high],
            "idle_cooldown_seconds": idle_cooldown_seconds,
            "stage_pilots": stage_pilots,
            "post_import_benchmark_duration_s": time.monotonic() - remote_started,
            "excluded": [
                "Modal image pull and worker boot",
                "CUDA context and Triton compilation",
                "device-buffer allocation",
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
            "max_planned_measurement_seconds": MAX_PLANNED_MEASUREMENT_SECONDS,
        },
        "correctness": {
            "method": "all output bytes equal E4M3FN encoding of 2.0",
            "checked_output_entries": ELEMENTS,
            "constant_one_fp8_bits_hex": f"0x{one_bits:02x}",
            "expected_two_fp8_bits_hex": f"0x{two_bits:02x}",
            "passed": output_correct,
        },
        "stages": stages,
        "pipeline_breakdown": pipeline_breakdown,
        "physical_roofline_estimate": {
            "nature": "logical-traffic HBM bandwidth and peak-board-power proxy",
            "b200_hbm_bandwidth_bytes_per_s": B200_HBM_BANDWIDTH_BYTES_PER_S,
            "b200_power_basis_W": B200_POWER_BASIS_W,
            "addition_logical_traffic_bytes": ADD_LOGICAL_TRAFFIC_BYTES,
            "addition_ideal_time_s": ROOFLINE_ADD_TIME_S,
            "addition_peak_power_energy_J": ROOFLINE_ADD_ENERGY_J,
            "pipeline_logical_traffic_bytes": PIPELINE_LOGICAL_TRAFFIC_BYTES,
            "pipeline_ideal_time_s": ROOFLINE_PIPELINE_TIME_S,
            "pipeline_peak_power_energy_J": ROOFLINE_PIPELINE_ENERGY_J,
            "limitations": (
                "Assumes each logical byte reaches HBM once, full 8 TB/s, and "
                "1,000 W throughout; it is not a measurement or lower bound."
            ),
        },
        "sutro_2d_grid_model": {
            "source_url": SUTRO_REPORT_URL,
            "distance": "d(a) = ceil(sqrt(a)) for positive address a",
            "energy_J_per_score_unit": SUTRO_ENERGY_J_PER_SCORE_UNIT,
            "out_of_place_schedule": (
                "direct C_i = A_i + B_i; contiguous A, B, C; every cell paid once"
            ),
            "out_of_place_score_units": SUTRO_OUT_OF_PLACE_SCORE,
            "out_of_place_energy_J": SUTRO_OUT_OF_PLACE_ENERGY_J,
            "optimal_in_place_schedule": (
                "A_i = A_i + B_i with overwritten/output A in nearest cells"
            ),
            "optimal_in_place_score_units": SUTRO_IN_PLACE_SCORE,
            "optimal_in_place_energy_J": SUTRO_IN_PLACE_ENERGY_J,
            "scope": (
                "Synthetic widthless source/final-output-read movement only; "
                "input placement, writes, arithmetic, HBM/cache endpoints, "
                "clocking, launch, idle, and leakage are unpriced."
            ),
            "initialization_energy": (
                "zero/undefined because the scorer makes initial placement free"
            ),
        },
        "model_comparison": grid_comparison,
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
    print("\n8192 x 8192 dense FP8 elementwise addition")
    print(f"  GPU: {result['hardware']['gpu_name']}")
    print(
        f"  elements/additions: "
        f"{result['operation_count']['elementwise_additions']:,}"
    )
    print(
        f"  measured idle baseline: "
        f"{result['measurement']['idle_baseline_W']:.2f} W"
    )
    print("\n  phase             runs      us/run    raw J/run    net J/run")
    for key in ("initialization", "addition", "pipeline"):
        stage = result["stages"][key]
        print(
            f"  {key:<14} "
            f"{stage['total_repetitions']:>9,}  "
            f"{stage['seconds_per_run'] * 1e6:>10.3f}  "
            f"{stage['raw_gpu_energy_J_per_run']:>11.6f}  "
            f"{stage['idle_adjusted_gpu_energy_J_per_run']:>11.6f}"
        )
    addition = result["stages"]["addition"]
    model = result["sutro_2d_grid_model"]
    comparison = result["model_comparison"]
    print(
        f"\n  add logical bandwidth: "
        f"{addition['effective_logical_bandwidth_GB_per_s']:.1f} GB/s"
    )
    print(
        "  raw add energy vs Sutro: "
        f"{addition['raw_gpu_energy_J_per_run']:.6f} J / "
        f"{model['out_of_place_energy_J']:.9f} J = "
        f"{comparison['measured_raw_over_sutro_out_of_place_ratio']:.2f}x"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--trial-seconds",
        type=_positive_float,
        default=DEFAULT_TRIAL_SECONDS,
        help=f"target aggregate trial duration (default: {DEFAULT_TRIAL_SECONDS:g})",
    )
    parser.add_argument(
        "--trials",
        type=_positive_int,
        default=DEFAULT_TRIALS,
        help=f"aggregate trials per phase (default: {DEFAULT_TRIALS})",
    )
    parser.add_argument(
        "--idle-seconds",
        type=_positive_float,
        default=DEFAULT_IDLE_SECONDS,
        help=f"before/after idle sample duration (default: {DEFAULT_IDLE_SECONDS:g})",
    )
    parser.add_argument(
        "--idle-cooldown-seconds",
        type=float,
        default=DEFAULT_IDLE_COOLDOWN_SECONDS,
        help=(
            "unmeasured cooldown before each idle sample "
            f"(default: {DEFAULT_IDLE_COOLDOWN_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--warmup-repetitions",
        type=_positive_int,
        default=DEFAULT_WARMUP_REPETITIONS,
        help=f"unmeasured additions before validation (default: {DEFAULT_WARMUP_REPETITIONS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "8k_fp8_add_blackwell_results.json",
        help="local JSON result path",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the Modal cost confirmation"
    )
    args = parser.parse_args()

    try:
        planned_seconds = _validate_plan(
            trial_seconds=args.trial_seconds,
            trials=args.trials,
            idle_seconds=args.idle_seconds,
            idle_cooldown_seconds=args.idle_cooldown_seconds,
            warmup_repetitions=args.warmup_repetitions,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"╭─ One-shot Modal {MODAL_GPU} 8192 x 8192 FP8 add ──────")
    print(f"│  planned measurement/cooldown: ~{planned_seconds:.0f} s")
    print("│  one container, one input, no configured retries")
    print(f"│  hard function cap: {EXECUTION_CAP_SECONDS} s")
    print(f"│  output: {args.output.resolve()}")
    print("╰───────────────────────────────────────────────────")
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
