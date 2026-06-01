"""CPU smoke test for the BPE-tokenised DiffusionBlocks AR submission.

Tiny config, ~10 steps, exercises the BPE marginalisation path in
predict() and the incremental-token-commit path in observe().

Requires tiktoken (the Modal image ships it; locally install via
``pip install tiktoken``).

Run from repo root:
    PYTHONPATH=. python submissions/diffusionblocks_ar_bpe/_smoke.py
"""
from __future__ import annotations

import sys
import torch

import submissions.diffusionblocks_ar_bpe.submission as sub


def main() -> int:
    torch.manual_seed(0)
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        print("[smoke] tiktoken not installed; install via `pip install tiktoken`")
        return 0

    # Enough chars that GPT-2 BPE produces > max_len tokens after the
    # cfg.max_len threshold below.
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
        max_len=32,
        batch_size=4,
        baseline_steps=5,
        gamma=0.10,
        lambda_ce=1.0,
        lr=5e-4,
        log_every=2,
    )
    device = torch.device("cpu")
    print(f"[smoke] training tiny BPE model on CPU: {cfg}")
    model, encoding, token_bytes_arr, token_lens = sub._train(text, cfg, device)

    print("[smoke] testing CharModel contract ...")
    cm = sub.DBlocksBPECharModel(
        model, encoding, token_bytes_arr, token_lens, device=device,
    )
    cm.reset()
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
    print(f"[smoke] streaming acc: {correct}/{n} (not meaningful at 10 steps)")

    print("[smoke] exercising cache-trim path ...")
    cm.max_len = 8
    for ch in "a" * 64:
        cm.predict()
        cm.observe(ch)
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
