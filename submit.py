#!/usr/bin/env python3
"""End-to-end submission runner for the wikitext energy benchmark on Modal.

What it does:
  1. AST-parses the user's submission file locally as a precheck (catches
     SyntaxError and missing `train` before any cloud spend; works without
     the submission's heavy deps installed in the local env).
  2. Defines a Modal App that pulls a prebuilt public image from ghcr.io
     containing PyTorch + nvidia-ml-py + pyarrow and the WikiText-103 raw
     splits already baked into /data, and a single A100-40GB function.
  3. The remote function verifies NVML, writes the user's submission to
     disk, runs run_eval, and returns the result dict.
  4. Saves result.json + nvml.json + run.log into the submission directory
     and appends a row to the Record History table in README.md.

Setup (once, from wip-wikitext/):
  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
  modal token new       # opens browser, writes ~/.modal.toml

Usage:
  python submit.py path/to/my_submission_dir/
  python submit.py submissions/modded_nanogpt --yes

The submission directory must contain a `submission.py` that defines
`train(train_text, valid_text=None) -> CharModel`.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

import task  # task-pinned constants — single source of truth

HERE = Path(__file__).resolve().parent

# Modal A100-40GB list price as of 2026-05: $2.10/hr.
# A typical run is ~15 min (image cold-start + verify_nvml + ~5 min
# training capped by task.MAX_TRAIN_SECONDS + ~2 min eval over val+test).
# The image is the prebuilt public ghcr.io artifact with torch +
# WikiText-103 already baked in, so cold start is just the registry
# pull (~85s); no GCS download or pip install at run time.
EST_RUNTIME_MIN = 15
EST_RATE_USD_PER_HR = 2.10
EST_COST_USD = round(EST_RUNTIME_MIN * EST_RATE_USD_PER_HR / 60, 2)

# task.INSTANCE_TYPE is the leaderboard-pinned hardware string. The
# "modal:" prefix is informational; Modal's gpu= kwarg takes the bare
# SKU. Strip and validate so a future change to task.INSTANCE_TYPE
# (different provider, different GPU) fails loudly here instead of
# silently launching on the wrong hardware.
_provider, _, MODAL_GPU = task.INSTANCE_TYPE.partition(":")
if _provider != "modal" or not MODAL_GPU:
    sys.exit(
        f"task.INSTANCE_TYPE = {task.INSTANCE_TYPE!r} not understood by "
        f"submit.py — expected 'modal:<gpu>' (e.g. 'modal:A100-40GB')."
    )


# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------
#
# Harness files are added to /workspace inside the container. The user's
# submission is *not* baked in — it's passed as bytes to the function
# call, so we don't rebuild the image per submission. Image-level deps
# (torch, nvidia-ml-py, pyarrow) and the WikiText-103 raw splits in /data
# are baked into the public ghcr.io image pulled via from_registry below.

HARNESS_FILES = (
    "wikitext.py",
    "run_eval.py",
    "verify_nvml.py",
    "task.py",
)


app = modal.App("wikitext-bench")

# Public prebuilt image:
#   python 3.11, torch 2.5.1+cu124, numpy 2.1.3, nvidia-ml-py 12.560.30,
#   pyarrow 18.1.0, /data/wiki.{train,valid,test}.raw
#
# Source: wip-wikitext/Dockerfile in this repo; rebuild + push via
#   docker build -t ghcr.io/ab-10/wikitext-bench:latest -f Dockerfile .
#   docker push ghcr.io/ab-10/wikitext-bench:latest
#
# We pin to :latest deliberately — submitters always pick up the
# newest deps. Bump to a dated tag (e.g. :wkt-YYYY-MM-DD) if you need
# reproducibility across record history.
#
# No add_python= : the registry image already has /usr/local/bin/python
# in place. Passing add_python="3.11" makes Modal try `ln -s python3
# python` and fail because the symlink already exists.
WIKITEXT_IMAGE_REF = "ghcr.io/ab-10/wikitext-bench:latest"

image = (
    modal.Image.from_registry(WIKITEXT_IMAGE_REF)
    .workdir("/workspace")
    # Modal re-imports submit.py inside the container to resolve the
    # remote function. submit.py does a top-level `import task`, so
    # /workspace (where task.py lands via add_local_file) must be on
    # sys.path before that import runs.
    #
    # PYTHONUNBUFFERED forces line-buffered stdout/stderr for the remote
    # function *and* any python child it spawns — without it, print()
    # output gets batched into ~8 KB blocks and shows up only at exit,
    # which makes `modal.enable_output()` look broken.
    .env({"PYTHONPATH": "/workspace", "PYTHONUNBUFFERED": "1"})
)
for _f in HARNESS_FILES:
    image = image.add_local_file(str(HERE / _f), f"/workspace/{_f}")


@app.function(
    image=image,
    gpu=MODAL_GPU,
    # Hard wall-clock cap on the whole function. Training itself is
    # bounded by task.MAX_TRAIN_SECONDS (SIGALRM-based wall_clock_guard
    # inside run_eval.py); the remaining budget covers image cold-start
    # plus 2×60K-char autoregressive eval. Observed eval throughput
    # varies ~1.4× across Modal workers (88–125 char/s); on a slow
    # worker the full pipeline is ~29 min, so 50 min gives headroom.
    timeout=50 * 60,
)
def run_submission(
    submission_bytes: bytes,
    submission_name: str,
    seed: int | None = None,
) -> dict:
    """Run a submission end-to-end on the pinned Modal GPU and return
    the result dict that submit.py will save and record.

    ``seed`` is exposed to the submission via the ``SEED`` env var (read
    by submission.py if present). Used by the floor-calibration sweep;
    None for normal record-class runs.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    workspace = Path("/workspace")
    os.chdir(workspace)
    if seed is not None:
        os.environ["SEED"] = str(seed)
        print(f"[modal] SEED={seed} (calibration run)")

    # WikiText-103 is baked into /data inside the prebuilt registry image
    # (see Dockerfile + WIKITEXT_IMAGE_REF above).

    # NVML probe — bail before training cycles if the energy counter
    # isn't exposed on this host.
    #
    # Stream stdout line-by-line so probe output shows up live in the
    # Modal log feed instead of being held until the process exits.
    # verify_nvml.py prints its JSON summary on the last stdout line, so
    # we remember the last non-empty line as we tee.
    print("[modal] verifying NVML energy counter ...")
    proc = subprocess.Popen(
        [sys.executable, "verify_nvml.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1,
    )
    last_line = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        s = line.strip()
        if s:
            last_line = s
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"verify_nvml.py failed (rc={rc}).")
    nvml_summary = json.loads(last_line)

    # Drop the user's submission next to the harness files. Keep the
    # original stem so run_eval.py's submission_name (== Path(...).stem)
    # propagates into the result JSON and the README record row.
    if submission_name in {Path(f).stem for f in HARNESS_FILES} | {"submission"}:
        raise RuntimeError(
            f"submission name {submission_name!r} collides with a harness "
            f"file; rename your submission file."
        )
    sub_path = workspace / f"{submission_name}.py"
    sub_path.write_bytes(submission_bytes)

    # Pull task constants from the in-image task.py (single source of
    # truth — submitters do not get to vary these).
    sys.path.insert(0, str(workspace))
    import importlib
    task_mod = importlib.import_module("task")
    test_chars = task_mod.TEST_CHARS
    max_train_seconds = task_mod.MAX_TRAIN_SECONDS
    acc_min = task_mod.ACC_MIN

    eval_args = [
        sys.executable, "run_eval.py",
        "--data-dir", "/data",
        "--submission", str(sub_path),
        "--results-json", "/tmp/result.json",
        "--max-test-chars", str(test_chars),
    ]
    if max_train_seconds is not None:
        eval_args += ["--max-train-seconds", str(max_train_seconds)]
    if acc_min is not None:
        eval_args += ["--acc-min", str(acc_min)]

    print(f"[modal] running submission "
          f"(TEST_CHARS={test_chars} "
          f"MAX_TRAIN_SECONDS={max_train_seconds} "
          f"ACC_MIN={acc_min}) ...")
    rc = subprocess.run(eval_args).returncode

    # run_eval exits 2 on either DQ path (wall-clock or val accuracy)
    # *with* a written result.json — that's a valid leaderboard outcome
    # and we ship it. Anything else missing the JSON is a harness failure.
    rj = Path("/tmp/result.json")
    if not rj.exists():
        raise RuntimeError(f"run_eval.py failed (rc={rc}); no result.json written")

    result = json.loads(rj.read_text())
    result["_nvml"] = nvml_summary
    if seed is not None:
        result["seed"] = seed
    return result


# ---------------------------------------------------------------------------
# Local orchestration
# ---------------------------------------------------------------------------

def precheck_submission(submission_path: Path) -> str:
    """AST-parse the submission file. Fails fast on SyntaxError and on
    ``train`` not being defined — both would otherwise surface mid-run
    on the Modal host, after Modal billing started.

    Returns the submission's ``__author__`` string if defined, else
    ``"@you"`` — used to credit the contributor in the Record History.

    AST-only on purpose: importing the submission would require its deps
    (torch etc.) in the local venv, which requirements.txt deliberately
    does not pin. The Modal container has them.
    """
    src = submission_path.read_text()
    try:
        tree = ast.parse(src, filename=str(submission_path))
    except SyntaxError as e:
        sys.exit(f"submission has a syntax error: {submission_path}\n  {e}")

    has_train = False
    author = ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "train":
            has_train = True
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "train":
                    has_train = True
                if isinstance(target, ast.Name) and target.id == "__author__":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        author = node.value.value
    if not has_train:
        sys.exit(
            f"submission must define train(train_text, valid_text=None) -> CharModel "
            f"at module level: {submission_path}"
        )
    return author or "@you"


def _suffixed(name: str, seed: int | None) -> str:
    """``"result.json"`` → ``"result.json"`` or ``"result.seed3.json"``."""
    if seed is None:
        return name
    stem, _, ext = name.partition(".")
    return f"{stem}.seed{seed}.{ext}" if ext else f"{stem}.seed{seed}"


def save_result(result: dict, sub_dir: Path, *, seed: int | None = None) -> Path:
    out_path = sub_dir / _suffixed("result.json", seed)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return out_path


def save_nvml_artifact(
    result: dict, sub_dir: Path, *, seed: int | None = None,
) -> Path | None:
    """Write ``<sub_dir>/nvml.json`` evidence from the embedded
    ``_nvml`` field returned by the Modal function.

    Returns the path written, or None if absent.
    """
    nvml = result.get("_nvml")
    if not nvml:
        return None
    out_path = sub_dir / _suffixed("nvml.json", seed)
    out_path.write_text(json.dumps(nvml, indent=2) + "\n")
    return out_path


def append_record(result: dict, dir_relpath: str) -> None:
    """Append one row to the Record History table in README.md.

    Replaces the placeholder dash row if present, otherwise appends.
    Disqualified rows render their accuracy cell as ``DQ`` so they
    don't pollute the leaderboard sort.
    """
    readme = HERE / "README.md"
    text = readme.read_text()
    energy = result.get("training_energy_J")
    energy_cell = f"{energy:>10,.0f}" if energy is not None else "         —"
    if result.get("disqualified"):
        acc_cell = "      DQ"
    else:
        acc_cell = f"{result['val_char_accuracy']:.4f}"
    contributor = result.get("contributor") or "@you"
    row = (
        f"| {result['date_utc'][:10]} "
        f"| {energy_cell} "
        f"| {acc_cell} "
        f"| {result['submission']} "
        f"| [dir]({dir_relpath}) "
        f"| {contributor} |\n"
    )
    placeholder = "| —    |          — |        — | —      | —          | —           |\n"
    if placeholder in text:
        text = text.replace(placeholder, row, 1)
    else:
        text = text.rstrip() + "\n" + row
    readme.write_text(text)


class _Tee(io.TextIOBase):
    """Mirror writes to multiple text streams. Used to capture submit.py
    stdout into <sub_dir>/run.log while still showing it in the terminal.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("submission_dir", type=Path,
                   help="Directory containing submission.py "
                        "(exposing train(train_text, valid_text=None) -> CharModel). "
                        "Run artifacts (result.json, nvml.json, run.log) are written "
                        "into this directory.")
    p.add_argument("--yes", action="store_true",
                   help="Skip cost confirmation prompt")
    p.add_argument("--seed", type=int, default=None,
                   help="If set, exposed to the submission as SEED env var "
                        "and treated as a floor-calibration run: artifacts "
                        "are written with a .seed{N} suffix and the result "
                        "is NOT appended to the Record History table.")
    args = p.parse_args()

    sub_dir = args.submission_dir.resolve()
    if not sub_dir.is_dir():
        sys.exit(f"submission directory not found: {args.submission_dir}")
    sub_path = sub_dir / "submission.py"
    if not sub_path.is_file():
        sys.exit(f"missing submission.py inside {sub_dir}")

    submission_name = sub_dir.name

    print(f"╭─ Modal {MODAL_GPU} wikitext submission ───────")
    print(f"│  submission:    {sub_dir}")
    print(f"│  est. runtime:  ~{EST_RUNTIME_MIN} min  (cold start + train + eval)")
    print(f"│  est. cost:     ~${EST_COST_USD:.2f}  (@ ${EST_RATE_USD_PER_HR:.2f}/hr)")
    print(f"╰───────────────────────────────────────────────")

    if not args.yes and input("proceed? [Y/n] ").strip().lower() not in ("", "y", "yes"):
        sys.exit("aborted")

    contributor = precheck_submission(sub_path)
    submission_bytes = sub_path.read_bytes()

    log_path = sub_dir / _suffixed("run.log", args.seed)
    log_f = log_path.open("w")
    seed_tag = f"  SEED={args.seed}" if args.seed is not None else ""
    log_f.write(
        f"# wikitext submit.py log — {submission_name}{seed_tag} — "
        f"{datetime.now(timezone.utc).replace(microsecond=0).isoformat()}Z\n"
    )

    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, log_f)
    try:
        print(f"[modal] launching {MODAL_GPU}{seed_tag} ...")
        # modal.enable_output() writes to the real stdout fd, so its
        # progress feed is visible in the terminal but not captured into
        # run.log. Our prints + the result block are captured.
        with modal.enable_output(), app.run():
            result = run_submission.remote(
                submission_bytes, submission_name, seed=args.seed,
            )
    finally:
        sys.stdout = real_stdout

    result["contributor"] = contributor
    out_path = save_result(result, sub_dir, seed=args.seed)
    save_nvml_artifact(result, sub_dir, seed=args.seed)
    # Seeded calibration runs don't enter the leaderboard.
    if args.seed is None:
        rel_dir = sub_dir.relative_to(HERE).as_posix()
        append_record(result, dir_relpath=rel_dir)

    log_f.write("\n# final result\n")
    log_f.write(json.dumps(result, indent=2) + "\n")
    log_f.close()

    if result.get("disqualified"):
        reason = result.get("reason", "unknown")
        dur = result.get("training_duration_s")
        energy = result.get("training_energy_J")
        print(f"[done] DISQUALIFIED — {reason}")
        if reason == "train_time_exceeded":
            cap = result.get("max_train_seconds")
            if cap is not None:
                print(f"       cap      = {cap:.0f} s")
            if dur is not None:
                print(f"       at_kill  = {dur:.1f} s")
        elif reason == "val_accuracy_below_floor":
            floor = result.get("acc_min")
            val_acc = result.get("val_char_accuracy")
            if floor is not None:
                print(f"       floor    = {floor:.4f}")
            if val_acc is not None:
                print(f"       val_acc  = {val_acc:.4f}")
            if dur is not None:
                print(f"       train_s  = {dur:.1f}")
        if energy is not None:
            print(f"       energy   = {energy:,.0f} J")
        print(f"       result   = {out_path}")
        # Non-zero exit so CI scripts can distinguish DQ from a clean
        # leaderboard submission, even though the harness ran cleanly.
        return 2

    print(f"[done] {out_path}")
    energy = result.get("training_energy_J")
    if energy is not None:
        print(f"       energy = {energy:,.0f} J")
    else:
        print(f"       energy = NOT MEASURED")
    print(f"       acc    = {result['val_char_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
