"""One-off Modal runner for submissions/mha_alpha05/test_kernel.py.

Uses the same A100-80GB image as submit.py (ghcr.io/ab-10/wikitext-bench:latest)
so the smoke test runs against the exact PyTorch / CUDA stack the leaderboard
uses. Does NOT engage the wikitext training/eval harness — this is purely a
correctness + perf check on the custom MHA kernel.

Usage:
    python run_modal_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

WIKITEXT_IMAGE_REF = "ghcr.io/ab-10/wikitext-bench:latest"

app = modal.App("mha-kernel-test")

image = (
    modal.Image.from_registry(WIKITEXT_IMAGE_REF)
    # Triton (used by torch._inductor / FlexAttention) JIT-compiles a C
    # wrapper at first kernel call. The base wikitext-bench image ships
    # PyTorch without a host C toolchain; without gcc the FlexAttention
    # call fails with "Failed to find C compiler". Add a minimal one.
    .apt_install("gcc", "g++")
    .workdir("/workspace")
    .env({"PYTHONPATH": "/workspace", "PYTHONUNBUFFERED": "1"})
    # Ship submission.py and test_kernel.py up.
    .add_local_file(
        str(HERE / "submissions/mha_alpha05/submission.py"),
        "/workspace/submission.py",
    )
    .add_local_file(
        str(HERE / "submissions/mha_alpha05/test_kernel.py"),
        "/workspace/test_kernel.py",
    )
    # wikitext.py is needed because submission.py imports CharModel from it.
    .add_local_file(str(HERE / "wikitext.py"), "/workspace/wikitext.py")
)


@app.function(image=image, gpu="A100-80GB", timeout=10 * 60)
def run_test() -> int:
    import subprocess, sys
    rc = subprocess.run(
        [sys.executable, "/workspace/test_kernel.py"],
        cwd="/workspace",
    ).returncode
    return rc


if __name__ == "__main__":
    with modal.enable_output(), app.run():
        rc = run_test.remote()
    print(f"\n[modal] test_kernel.py exit code: {rc}")
    sys.exit(rc)
