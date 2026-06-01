"""Modal launcher for the diffusionblocks_ar BPE + MAUVE debug.

Reuses the prebuilt wikitext-bench image (torch + /data baked in), adds
``transformers`` for the GPT-2 tokenizer/featurizer and ``mauve-text`` for
the MAUVE eval, then runs run.py on an A100-80GB.

Usage:
    modal run research/debug/diffusionblocks_ar_mauve/run_modal.py \\
        --steps 12000

Artifacts (result.json, loss_curve.csv, samples.txt) are streamed back to
the local debug directory.
"""
from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

WIKITEXT_IMAGE_REF = "ghcr.io/ab-10/wikitext-bench:latest"

image = (
    modal.Image.from_registry(WIKITEXT_IMAGE_REF)
    .workdir("/workspace")
    .pip_install(
        "transformers==4.46.3",
        "mauve-text==0.4.0",
        "scikit-learn",
        "faiss-cpu",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file(str(HERE / "run.py"), "/workspace/run.py")
)

app = modal.App("dblocks-ar-mauve")


@app.function(image=image, gpu="A100-80GB", timeout=4 * 60 * 60)
def run(steps: int = 12000, batch_size: int = 32, max_len: int = 256,
        n_prompts: int = 250, prompt_len: int = 32, continuation_len: int = 50,
        n_inference_steps: int = 50) -> dict:
    import subprocess
    import sys
    from pathlib import Path

    out = Path("/tmp/mauve_out")
    out.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run([
        sys.executable, "/workspace/run.py",
        "--data-dir", "/data",
        "--out", str(out),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--max-len", str(max_len),
        "--n-prompts", str(n_prompts),
        "--prompt-len", str(prompt_len),
        "--continuation-len", str(continuation_len),
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
def main(steps: int = 12000, batch_size: int = 32, max_len: int = 256,
         n_prompts: int = 250, prompt_len: int = 32, continuation_len: int = 50,
         n_inference_steps: int = 50):
    artifacts = run.remote(
        steps=steps, batch_size=batch_size, max_len=max_len,
        n_prompts=n_prompts, prompt_len=prompt_len,
        continuation_len=continuation_len, n_inference_steps=n_inference_steps,
    )
    for name, content in artifacts.items():
        path = HERE / name
        path.write_text(content)
        print(f"wrote {path} ({len(content)} bytes)")
