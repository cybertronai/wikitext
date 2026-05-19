"""Char-bigram baselines on WikiText-103: unigram floor and bigram-table ceiling.

These bound any deterministic prev_byte -> next_byte predictor:
  unigram      : argmax of marginal byte frequency in train  (lower bound)
  bigram-table : per-prev-byte argmax of empirical conditional  (upper bound
                 for any deterministic single-byte-context predictor)

Any single-byte-context model — including an NBB with 256-way one-hot input —
cannot exceed bigram-table on a held-out set. The diagnostic passes if NBB
reaches a noticeable fraction of the gap between unigram and bigram-table.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np


TRAIN = Path("/tmp/wikitext_data/wiki.train.raw")
VALID = Path("/tmp/wikitext_data/wiki.valid.raw")


def load_bytes(path: Path, limit_mb: float | None = None) -> np.ndarray:
    data = path.read_bytes()
    if limit_mb is not None:
        data = data[: int(limit_mb * 1_000_000)]
    return np.frombuffer(data, dtype=np.uint8)


def unigram_acc(train: np.ndarray, val: np.ndarray) -> tuple[float, int]:
    counts = np.bincount(train, minlength=256)
    modal = int(counts.argmax())
    return float((val == modal).mean()), modal


def bigram_table_acc(train: np.ndarray, val: np.ndarray) -> tuple[float, np.ndarray]:
    bigram = np.zeros((256, 256), dtype=np.int64)
    np.add.at(bigram, (train[:-1], train[1:]), 1)
    modal_next = bigram.argmax(axis=1).astype(np.uint8)  # (256,)
    pred = modal_next[val[:-1]]
    acc = float((pred == val[1:]).mean())
    return acc, modal_next


def main():
    print("loading data...")
    t0 = time.monotonic()
    train = load_bytes(TRAIN)
    val = load_bytes(VALID)
    print(f"  train: {train.size:,} bytes   val: {val.size:,} bytes   "
          f"({time.monotonic() - t0:.1f}s)")

    t0 = time.monotonic()
    u_acc, modal_byte = unigram_acc(train, val)
    print(f"unigram     acc = {u_acc:.4f}   (modal byte: {modal_byte!r} = {chr(modal_byte)!r}   "
          f"{time.monotonic() - t0:.1f}s)")

    t0 = time.monotonic()
    b_acc, modal_next = bigram_table_acc(train, val)
    print(f"bigram-table acc = {b_acc:.4f}   ({time.monotonic() - t0:.1f}s)")

    print(f"\ngap unigram→bigram-table = {b_acc - u_acc:.4f}")
    print(f"D2 pass threshold (≥0.25): {'TBD by NBB run'}")


if __name__ == "__main__":
    main()
