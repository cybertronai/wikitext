"""Modal launcher for the diffusionblocks_ar geneval debug.

Reuses the prebuilt wikitext-bench image (torch + /data baked in), adds
``transformers`` for the GPT-2 scorer, then runs run.py on an A100-80GB.

Usage:
    modal run research/debug/diffusionblocks_ar_geneval/run_modal.py \\
        --steps 12000

Artifacts (result.json, loss_curve.csv, samples.txt) are streamed back to
the local debug directory.
"""
from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent

WIKITEXT_IMAGE_REF = "ghcr.io/ab-10/wikitext-bench:latest"

image = (
    modal.Image.from_registry(WIKITEXT_IMAGE_REF)
    .workdir("/workspace")
    .pip_install("transformers==4.46.3")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file(str(HERE / "run.py"), "/workspace/run.py")
)

app = modal.App("dblocks-ar-geneval")


@app.function(image=image, gpu="A100-80GB", timeout=4 * 60 * 60)
def run(steps: int = 12000, n_samples: int = 50, seq_len: int = 256,
        n_inference_steps: int = 50) -> dict:
    import json
    import subprocess
    import sys
    from pathlib import Path

    out = Path("/tmp/geneval_out")
    out.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run([
        sys.executable, "/workspace/run.py",
        "--data-dir", "/data",
        "--out", str(out),
        "--steps", str(steps),
        "--n-samples", str(n_samples),
        "--seq-len", str(seq_len),
        "--n-inference-steps", str(n_inference_steps),
    ]).returncode
    if rc != 0:
        raise RuntimeError(f"run.py failed (rc={rc})")
    return {
        "result.json": (out / "result.json").read_text(),
        "loss_curve.csv": (out / "loss_curve.csv").read_text(),
        "samples.txt": (out / "samples.txt").read_text(),
    }


@app.local_entrypoint()
def main(steps: int = 12000, n_samples: int = 50, seq_len: int = 256,
         n_inference_steps: int = 50):
    artifacts = run.remote(steps=steps, n_samples=n_samples, seq_len=seq_len,
                           n_inference_steps=n_inference_steps)
    for name, content in artifacts.items():
        path = HERE / name
        path.write_text(content)
        print(f"wrote {path} ({len(content)} bytes)")
