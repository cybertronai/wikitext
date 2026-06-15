"""Submission runner.

Trains the user's submission on the WikiText-103 train split (energy
measured, wall-clock capped by README rule 4), then evaluates greedy
char-acc on the val split (gated by README rule 5 and recorded on the
leaderboard). Energy is not measured during eval, per the v0 design —
training-only accounting.

Example::

    python3 run_eval.py --data-dir /data/wikitext-103 \\
        --submission submissions/modded_nanogpt/submission.py

The output line is the (energy, val acc) pair the runner uses to
populate the record table. The test split is held out — it is not
scored or recorded by this runner.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from wikitext import (
    CharModel,
    EnergyMeter,
    TrainingTimeoutError,
    evaluate,
    load_wikitext103,
    wall_clock_guard,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory containing wiki.{train,valid,test}.raw")
    p.add_argument("--max-test-chars", type=int, default=60_000,
                   help="Scoring-window length on the val split. The "
                        "runner sets this from task.TEST_CHARS (currently "
                        "60,000); local dev may override. Pass 0 to score "
                        "the full ~250K-char val split. (Flag name is "
                        "historical — test split is no longer scored.)")
    p.add_argument("--progress-every", type=int, default=None,
                   help="Print eval progress every N chars. Pass 0 to "
                        "disable progress output. Default: about 50 updates.")
    p.add_argument("--submission", type=Path, required=True,
                   help="Path to a Python file exposing "
                        "`train(train_text, valid_text=None) -> CharModel`.")
    p.add_argument("--results-json", type=Path, default=None,
                   help="Write the final (energy, accuracy, gpu, …) tuple "
                        "as a JSON object to this path. Used by submit.py.")
    p.add_argument("--max-train-seconds", type=float, default=None,
                   help="Wall-clock training cap in seconds (README rule 4). "
                        "The runner sets this from task.MAX_TRAIN_SECONDS; "
                        "submissions cannot vary it. If set, a SIGALRM-based "
                        "guard kills the training run when the budget elapses "
                        "and the submission is reported as DISQUALIFIED.")
    p.add_argument("--acc-min", type=float, default=None,
                   help="Minimum val char-accuracy (README rule 5). The "
                        "runner sets this from task.ACC_MIN; submissions "
                        "cannot vary it. If the val score falls below the "
                        "floor, the submission is reported as DISQUALIFIED.")
    p.add_argument("--ce-max", type=float, default=None,
                   help="Maximum native val cross-entropy in bits/char "
                        "for the CE track. The runner sets this from "
                        "task.CE_MAX; submissions cannot vary it. CE-track "
                        "gating: if the submission exposes ``predict_dist()`` "
                        "AND native CE on val is at or below this floor, "
                        "``ce_track_status`` = pass. If it "
                        "exposes ``predict_dist()`` but exceeds the floor, "
                        "status = fail. If ``predict_dist()`` is not "
                        "implemented, status = n_a (CE track not entered).")
    args = p.parse_args()

    print(f"loading WikiText-103 from {args.data_dir} ...")
    train_text = load_wikitext103(args.data_dir, "train")
    valid_text = load_wikitext103(args.data_dir, "valid")
    # Cap val at --max-test-chars so accuracy noise is comparable across
    # runs. The full val split is ~250K chars; using all of it would add
    # ~6 min eval per run. The test split is intentionally not loaded —
    # it's held out from the harness for now to keep runs short.
    val_score_text = valid_text[: args.max_test_chars] if args.max_test_chars else valid_text
    print(f"  train chars: {len(train_text):,}")
    print(f"  val   chars: {len(val_score_text):,}  (scored, gated by --acc-min)")

    meter = EnergyMeter()
    if not meter.available:
        print("WARNING: NVML energy counter not available on this host; "
              "energy will not be measured. Submissions must run on a "
              "host with NVML access (Modal A100, etc.).")
    if args.max_train_seconds is not None:
        print(f"train wall-clock cap: {args.max_train_seconds:.0f} s")
    if args.acc_min is not None:
        print(f"val accuracy floor : {args.acc_min:.4f}")
    if args.ce_max is not None:
        print(f"CE-track floor (bits/char, native): {args.ce_max:.4f}")

    submission_name = args.submission.stem
    spec = importlib.util.spec_from_file_location("user_submission", args.submission)
    if spec is None or spec.loader is None:
        sys.exit(f"could not import submission file: {args.submission}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    train_fn = getattr(mod, "train", None)
    if not callable(train_fn):
        sys.exit(f"submission must define train(train_text, valid_text=None) -> "
                 f"CharModel: {args.submission}")
    # Pass valid_text only if the user's signature accepts it.
    accepts_valid = "valid_text" in inspect.signature(train_fn).parameters

    print(f"training submission {args.submission} ...")
    m = None
    timed_out = False
    try:
        with meter.measure() as m, wall_clock_guard(args.max_train_seconds):
            model = train_fn(train_text, valid_text=valid_text) if accepts_valid \
                else train_fn(train_text)
        if not isinstance(model, CharModel):
            sys.exit(f"submission's train() returned {type(model).__name__}, "
                     f"expected a CharModel subclass")
    except TrainingTimeoutError as e:
        timed_out = True
        print("---")
        print(f"DISQUALIFIED: {e}")
        print(f"submission         : {submission_name}")
        if m is not None:
            print(f"training duration  : {m.duration_s:.1f}s")
            if m.energy_joules is not None:
                print(f"training energy (J): {_fmt_training_energy(m)}  (at kill)")
        if args.results_json is not None:
            payload = {
                "submission": submission_name,
                "disqualified": True,
                "reason": "train_time_exceeded",
                "max_train_seconds": args.max_train_seconds,
                "training_energy_J": m.energy_joules if m is not None else None,
                "training_duration_s": m.duration_s if m is not None else None,
                "cpu_energy_J": m.cpu_energy_J if m is not None else None,
                "total_energy_J": m.total_energy_J if m is not None else None,
                "gpu_name": _gpu_name(),
                "date_utc": _utc_now(),
            }
            args.results_json.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote {args.results_json}")
        sys.exit(2)

    print(f"training: {m}")

    print(f"evaluating on val split ...")
    val_progress = (
        args.progress_every
        if args.progress_every is not None
        else max(1, len(val_score_text) // 50)
    )
    val_result = evaluate(model, val_score_text, progress_every=val_progress)
    print(val_result)

    # Val accuracy gate (README rule 5).
    if args.acc_min is not None and val_result.accuracy < args.acc_min:
        print("---")
        print(f"DISQUALIFIED: val accuracy {val_result.accuracy:.4f} "
              f"below floor {args.acc_min:.4f}")
        print(f"submission         : {submission_name}")
        if m.energy_joules is not None:
            print(f"training energy (J): {_fmt_training_energy(m)}")
        print(f"training duration  : {m.duration_s:.1f}s")
        if args.results_json is not None:
            payload = {
                "submission": submission_name,
                "disqualified": True,
                "reason": "val_accuracy_below_floor",
                "acc_min": args.acc_min,
                "val_char_accuracy": val_result.accuracy,
                "val_chars": val_result.n_chars,
                "val_bits_per_char": val_result.bits_per_char,
                "val_ce_chars": val_result.n_ce_chars,
                "training_energy_J": m.energy_joules,
                "training_duration_s": m.duration_s,
                "cpu_energy_J": m.cpu_energy_J,
                "total_energy_J": m.total_energy_J,
                "gpu_name": _gpu_name(),
                "date_utc": _utc_now(),
            }
            args.results_json.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote {args.results_json}")
        sys.exit(2)

    print("---")
    print(f"submission         : {submission_name}")
    print(f"training energy (J): {_fmt_training_energy(m)}")
    print(f"training duration  : {m.duration_s:.1f}s")
    print(f"val  char-accuracy : {val_result.accuracy:.4f}")
    print(f"val  chars         : {val_result.n_chars:,}")

    # CE-track status. Independent of the acc-track gate: a submission
    # can pass the acc gate but fail CE, or pass CE while DQ'd on acc
    # (the acc DQ already exited above). "n_a" means the submission
    # didn't expose predict_dist, so it never entered the CE track.
    ce = val_result.bits_per_char
    if ce is None:
        ce_track_status = "n_a"
    elif args.ce_max is None:
        ce_track_status = "n_a"  # no threshold supplied
    elif ce <= args.ce_max:
        ce_track_status = "pass"
    else:
        ce_track_status = "fail"
    if ce_track_status != "n_a":
        print(f"CE  track          : {ce_track_status}  "
              f"(native={ce:.4f} bits/char, "
              f"floor={args.ce_max:.4f})")

    if args.results_json is not None:
        payload = {
            "submission": submission_name,
            "training_energy_J": m.energy_joules,
            "training_duration_s": m.duration_s,
            "cpu_energy_J": m.cpu_energy_J,
            "total_energy_J": m.total_energy_J,
            "val_char_accuracy": val_result.accuracy,
            "val_chars": val_result.n_chars,
            "val_bits_per_char": val_result.bits_per_char,
            "val_ce_chars": val_result.n_ce_chars,
            "ce_track_status": ce_track_status,
            "ce_max": args.ce_max,
            "gpu_name": _gpu_name(),
            "date_utc": _utc_now(),
        }
        args.results_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.results_json}")


def _gpu_name() -> str | None:
    try:
        import pynvml  # type: ignore[import-not-found]
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return n.decode() if isinstance(n, bytes) else n
    except Exception:
        return None


def _utc_now() -> str:
    return (datetime.now(timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"))


def _fmt_training_energy(m) -> str:
    if (m.total_energy_J is not None
            and m.energy_joules is not None
            and m.cpu_energy_J is not None):
        return (f"{m.total_energy_J:,.1f} "
                f"({m.energy_joules:,.1f} GPU + {m.cpu_energy_J:,.1f} CPU)")
    if m.energy_joules is not None:
        return f"{m.energy_joules:,.1f}"
    return "NOT MEASURED"


if __name__ == "__main__":
    main()
