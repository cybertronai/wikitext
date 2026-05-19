"""Local PPMd-D runner. Loads WikiText-103, drives the C extension via
ctypes, trains on as much of the 541 MB train stream as fits in the
budget, evals on the first 60K chars of val (char-aware, matches what
the official CharModel runner would compute).

No Modal, no GPU, no submit pipeline — this is a "is PPM promising?"
spike. Use:

    python experiments/ppm_c/run.py --K 6 --max-seconds 280

Optional knobs:
    --train-bytes N      cap training data at N bytes (default: all)
    --max-nodes N        node arena size (default 20M)
    --max-entries N      entries arena size (default 200M)
    --no-online          disable observe()-time count updates at eval
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from ctypes import POINTER, c_double, c_int, c_int8, c_int16, c_int32, c_int64, c_uint8, c_void_p
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/tmp/wikitext_data")
DEFAULT_VAL_CHARS = 60_000  # matches README rule 5


def load_lib() -> ctypes.CDLL:
    so = HERE / "ppm_core.so"
    if not so.exists():
        raise SystemExit(f"missing {so}; build with experiments/ppm_c/build.sh")
    lib = ctypes.CDLL(str(so))

    lib.ppm_create.argtypes = [c_int, c_int64, c_int64]
    lib.ppm_create.restype = c_void_p

    lib.ppm_destroy.argtypes = [c_void_p]
    lib.ppm_destroy.restype = None

    lib.ppm_train_bulk.argtypes = [c_void_p, POINTER(c_uint8), c_int64]
    lib.ppm_train_bulk.restype = c_int64

    lib.ppm_eval_bulk.argtypes = [c_void_p, POINTER(c_uint8), c_int64, c_int]
    lib.ppm_eval_bulk.restype = c_int64

    lib.ppm_eval_chars.argtypes = [
        c_void_p, POINTER(c_uint8), c_int64,
        POINTER(c_int32), c_int64, c_int,
    ]
    lib.ppm_eval_chars.restype = c_int64

    lib.ppm_reset_path.argtypes = [c_void_p]
    lib.ppm_reset_path.restype = None

    for name in ("ppm_n_nodes", "ppm_entries_used", "ppm_bytes_seen",
                 "ppm_node_exhausted", "ppm_entries_exhausted"):
        getattr(lib, name).argtypes = [c_void_p]
        getattr(lib, name).restype = c_int64

    return lib


def load_split(data_dir: Path, split: str) -> str:
    return (data_dir / f"wiki.{split}.raw").read_text(encoding="utf-8")


def char_byte_offsets(s: str) -> tuple[bytes, np.ndarray]:
    """Return (utf8_bytes, offsets[n_chars+1]) — offsets[i] = byte
    index where char i starts; offsets[-1] = len(bytes)."""
    # Per-char UTF-8 length without iterating in pure Python: build via
    # a vectorized table.
    raw = s.encode("utf-8")
    cps = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)
    # UTF-8 byte length per codepoint
    lens = np.ones(len(cps), dtype=np.int32)
    lens[cps >= 0x80]     = 2
    lens[cps >= 0x800]    = 3
    lens[cps >= 0x10000]  = 4
    offsets = np.empty(len(cps) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(lens, out=offsets[1:])
    assert offsets[-1] == len(raw), (offsets[-1], len(raw))
    return raw, offsets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--max-seconds", type=float, default=280.0,
                    help="train wall-clock budget; we chunk training and "
                         "break when this is reached")
    ap.add_argument("--train-bytes", type=int, default=0,
                    help="0 = use entire train split")
    ap.add_argument("--max-nodes", type=int, default=20_000_000)
    ap.add_argument("--max-entries", type=int, default=200_000_000)
    ap.add_argument("--val-chars", type=int, default=DEFAULT_VAL_CHARS)
    ap.add_argument("--no-online", action="store_true")
    ap.add_argument("--chunk-bytes", type=int, default=10_000_000,
                    help="train chunk size between budget checks")
    ap.add_argument("--eval-every-bytes", type=int, default=0,
                    help="if > 0, run a clean val eval (online=off) "
                         "every N training bytes; emits a learning curve")
    args = ap.parse_args()

    lib = load_lib()

    data_dir = Path(args.data_dir)
    print(f"[run] loading splits from {data_dir}", flush=True)
    train_text = load_split(data_dir, "train")
    valid_text = load_split(data_dir, "valid")
    print(f"[run] train chars={len(train_text):,}  val chars={len(valid_text):,}",
          flush=True)

    train_bytes = train_text.encode("utf-8")
    if args.train_bytes and args.train_bytes < len(train_bytes):
        train_bytes = train_bytes[: args.train_bytes]
    print(f"[run] train bytes available={len(train_bytes):,}", flush=True)

    val_raw_full, val_offsets_full = char_byte_offsets(valid_text)
    n_val_chars = min(args.val_chars, len(val_offsets_full) - 1)
    val_byte_end = int(val_offsets_full[n_val_chars])
    val_raw = val_raw_full[:val_byte_end]
    val_offsets = val_offsets_full[: n_val_chars + 1].copy()
    print(f"[run] eval chars={n_val_chars:,}  eval bytes={val_byte_end:,}",
          flush=True)

    print(f"[run] creating PPM(K={args.K}, max_nodes={args.max_nodes:,}, "
          f"max_entries={args.max_entries:,})", flush=True)
    p = lib.ppm_create(args.K, args.max_nodes, args.max_entries)
    if not p:
        raise SystemExit("ppm_create failed (likely OOM)")

    n_total = len(train_bytes)
    chunk = max(1, args.chunk_bytes)
    t0 = time.monotonic()
    pos = 0
    last_print = t0
    train_array = (c_uint8 * 0)()  # placeholder for ptr cast

    # Pre-build val buffers once.
    val_buf = (c_uint8 * len(val_raw)).from_buffer_copy(val_raw)
    val_off_buf = val_offsets.ctypes.data_as(POINTER(c_int32))

    curve: list[tuple[int, float, float]] = []  # (bytes_seen, val_acc, elapsed_s)
    next_eval_at = args.eval_every_bytes if args.eval_every_bytes > 0 else None

    def snapshot_eval(tag: str) -> float:
        # Clean eval: online_updates=off so we don't modify the trie.
        n_corr = lib.ppm_eval_chars(p, val_buf, len(val_raw),
                                    val_off_buf, n_val_chars, 0)
        return n_corr / n_val_chars

    while pos < n_total:
        budget_left = args.max_seconds - (time.monotonic() - t0)
        if budget_left <= 0:
            break
        end = min(n_total, pos + chunk)
        # Direct ctypes pointer to bytes (no copy).
        buf = (c_uint8 * (end - pos)).from_buffer_copy(train_bytes[pos:end])
        lib.ppm_train_bulk(p, buf, end - pos)
        pos = end

        # Curve-mode: periodic clean eval at training milestones.
        if next_eval_at is not None and pos >= next_eval_at:
            acc = snapshot_eval(f"curve@{pos}")
            elapsed_now = time.monotonic() - t0
            curve.append((pos, acc, elapsed_now))
            print(f"[curve] bytes={pos:>11,}  val_char_acc={acc:.4f}  "
                  f"(elapsed={elapsed_now:.1f}s, "
                  f"nodes={lib.ppm_n_nodes(p):,})", flush=True)
            next_eval_at += args.eval_every_bytes

        now = time.monotonic()
        if now - last_print >= 5.0 or pos == n_total:
            elapsed = now - t0
            rate = pos / max(1e-9, elapsed)
            n_nodes = lib.ppm_n_nodes(p)
            entries = lib.ppm_entries_used(p)
            print(f"[train] {pos:>11,} / {n_total:,} bytes "
                  f"({100.0 * pos / n_total:5.1f}%)  "
                  f"{rate / 1e6:5.2f} MB/s  "
                  f"elapsed={elapsed:6.1f}s  "
                  f"nodes={n_nodes:>10,}  entries={entries:>11,}",
                  flush=True)
            last_print = now

    train_elapsed = time.monotonic() - t0
    bytes_ingested = lib.ppm_bytes_seen(p)
    n_nodes = lib.ppm_n_nodes(p)
    entries = lib.ppm_entries_used(p)
    n_ex = lib.ppm_node_exhausted(p)
    e_ex = lib.ppm_entries_exhausted(p)
    print(f"[train] done: {bytes_ingested:,} bytes in {train_elapsed:.1f}s "
          f"({bytes_ingested / train_elapsed / 1e6:.2f} MB/s)  "
          f"nodes={n_nodes:,}  entries={entries:,}  "
          f"node_exhausted={n_ex}  entries_exhausted={e_ex}",
          flush=True)

    # ----------------------------------------------------------------
    # Eval — char-aware (matches what the official runner sees).
    # ----------------------------------------------------------------
    do_online = 0 if args.no_online else 1
    t1 = time.monotonic()
    n_correct = lib.ppm_eval_chars(p, val_buf, len(val_raw),
                                   val_off_buf, n_val_chars, do_online)
    eval_elapsed = time.monotonic() - t1
    char_acc = n_correct / n_val_chars
    print(f"[eval ] char-acc = {char_acc:.4f}  "
          f"({n_correct:,} / {n_val_chars:,})  "
          f"online_updates={'on' if do_online else 'off'}  "
          f"eval_time={eval_elapsed:.1f}s",
          flush=True)

    # Also report byte-level acc on the same window for reference.
    p2 = lib.ppm_create(args.K, args.max_nodes, args.max_entries)
    # Re-run training so byte-eval starts from the same trained state.
    # (Cheaper: re-eval needs a fresh path; the char eval already
    # corrupted online state via observe(). To get a clean byte-eval
    # comparison we'd need a checkpoint — skip for now.)
    lib.ppm_destroy(p2)
    lib.ppm_destroy(p)

    print()
    print("==========  PPMd-D / WikiText-103 / local CPU  ==========")
    print(f"  K                = {args.K}")
    print(f"  train ingested   = {bytes_ingested:,} bytes "
          f"({100.0 * bytes_ingested / len(train_bytes):.1f}% of "
          f"{len(train_bytes):,})")
    print(f"  train wall time  = {train_elapsed:.1f} s")
    print(f"  train throughput = {bytes_ingested / train_elapsed / 1e6:.2f} MB/s")
    print(f"  trie nodes       = {n_nodes:,}")
    print(f"  trie entries     = {entries:,}")
    print(f"  val char-acc     = {char_acc:.4f}  (first {n_val_chars:,} chars)")
    print(f"  online updates   = {'on' if do_online else 'off'}")
    print("=========================================================")
    if curve:
        print()
        print("==========  Learning curve (clean eval, no online)  ==========")
        print(f"  {'bytes_seen':>13}  {'val_char_acc':>13}  {'elapsed_s':>10}")
        for b, a, e in curve:
            print(f"  {b:>13,}  {a:>13.4f}  {e:>10.1f}")
        print("==============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
