"""CPU smoke test: tiny config, ~10 steps. Shake out shape bugs.

Run from repo root:
    PYTHONPATH=. python submissions/diffusionblocks_ar/_smoke.py
"""
from __future__ import annotations

import sys
import torch

import submissions.diffusionblocks_ar.submission as sub


def main() -> int:
    torch.manual_seed(0)
    # Tiny corpus: enough bytes for a few batches at max_len=64.
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump! "
    ) * 200

    cfg = sub.TrainConfig(
        model_dim=64,
        num_layers=4,
        head_dim=16,
        num_blocks=2,
        cond_dim=32,
        max_len=64,
        batch_size=4,
        baseline_steps=5,
        gamma=0.10,
        lambda_ce=1.0,
        lr=5e-4,
        log_every=2,
    )
    device = torch.device("cpu")
    print(f"[smoke] training tiny model on CPU: {cfg}")
    model = sub._train(text, cfg, device)

    # Test inference contract.
    print("[smoke] testing CharModel contract ...")
    cm = sub.DBlocksCharModel(model, device=device)
    cm.reset()
    seen = ""
    correct = 0
    n = 0
    for ch in "The quick brown fox":
        pred = cm.predict()
        if not isinstance(pred, str):
            print(f"[smoke] FAIL: predict() returned {type(pred)}, not str")
            return 1
        if pred == ch:
            correct += 1
        n += 1
        cm.observe(ch)
        seen += ch
    print(f"[smoke] streaming acc: {correct}/{n} (random ≈ 1/256 — accuracy "
          f"not meaningful at 10 steps)")

    # Confirm cache-trim path doesn't crash by forcing it.
    print("[smoke] exercising cache-trim path ...")
    cm.max_len = 16  # shrink to trip the trim branch
    for ch in "a" * 32:
        cm.predict()
        cm.observe(ch)
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
