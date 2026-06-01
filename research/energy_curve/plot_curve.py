"""Plot val char-accuracy vs training joules from run_curve.py output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path,
                   default=Path(__file__).with_name("result.json"))
    p.add_argument("--out", type=Path,
                   default=Path(__file__).with_name("curve.png"))
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    rows = data["checkpoints"]
    joules = [r["joules_J"] for r in rows]
    acc = [r["val_char_accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(joules, acc, marker="o")
    ax.set_xlabel("Training energy (J, GPU NVML net of idle baseline)")
    ax.set_ylabel(f"Val char-accuracy (first {data['eval_chars']:,} chars)")
    ax.set_title("modded_nanogpt: char-acc vs joules")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
