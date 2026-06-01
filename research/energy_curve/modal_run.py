"""Modal driver for run_curve.py.

Pulls the same prebuilt wikitext-bench image as submit.py (torch + nvml +
/data baked in), mounts wikitext.py / submission.py / run_curve.py into
/workspace, runs run_curve.py on a single A100, and pulls result.json
back to the local research/energy_curve/ dir.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# REPO is only needed on the local side for the add_local_file calls
# below. Inside the Modal container the script lands at /root/modal_run.py
# (no grandparent), so guard the parents[1] lookup.
REPO = HERE.parents[1] if len(HERE.parents) >= 2 else HERE

WIKITEXT_IMAGE_REF = "ghcr.io/ab-10/wikitext-bench:latest"

image = (
    modal.Image.from_registry(WIKITEXT_IMAGE_REF)
    .workdir("/workspace")
    .env({"PYTHONPATH": "/workspace", "PYTHONUNBUFFERED": "1"})
    .add_local_file(str(REPO / "wikitext.py"), "/workspace/wikitext.py")
    .add_local_file(
        str(REPO / "submissions" / "modded_nanogpt" / "submission.py"),
        "/workspace/submissions/modded_nanogpt/submission.py",
    )
    .add_local_file(str(HERE / "run_curve.py"),
                    "/workspace/research/energy_curve/run_curve.py")
)

app = modal.App("wikitext-energy-curve")


@app.function(image=image, gpu="A100-80GB", timeout=90 * 60)
def run_curve() -> dict:
    # Each eval is ~6 min at 60K chars on A100; train ~4 min; total ~64
    # min for 10 checkpoints. 90 min gives headroom for cold-start +
    # variance. run_curve.py writes result.json incrementally after each
    # eval, so a timeout still yields a partial-checkpoint JSON.
    rc = subprocess.run(
        [sys.executable, "/workspace/research/energy_curve/run_curve.py",
         "--data-dir", "/data",
         "--out", "/tmp/result.json"]
    ).returncode
    out = Path("/tmp/result.json")
    if not out.exists():
        raise RuntimeError(f"run_curve.py failed (rc={rc}); no result.json")
    return json.loads(out.read_text())


def main():
    with modal.enable_output(), app.run():
        result = run_curve.remote()
    out_path = HERE / "result.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_path}")
    for r in result["checkpoints"]:
        print(f"  step={r['step']:5d}  J={r['joules_J']:>8,.0f}  "
              f"acc={r['val_char_accuracy']:.4f}")


if __name__ == "__main__":
    main()
