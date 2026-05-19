"""NBB bigram diagnostic (RESEARCH_DIRECTIONS.md D2).

Port of `schmidhuber-problems/nbb-xor/nbb_xor.py` to the WikiText byte-bigram
task: given the previous byte, predict the next byte.

Architecture:
  - 257 input units = bias + 256 one-hot byte                    (clamped)
  - n_hidden units, one WTA subset
  - 256 output units, one WTA subset                             (one-hot byte)

Per-pattern presentation: 5 ticks (vs XOR's 6). Activations reset between
patterns. The local bucket-brigade weight update fires at every tick.

Pass criterion: reach ≥ 0.25 val acc within 60 s of CPU training, beating
the unigram floor (0.1885) and approaching the bigram-table ceiling (0.2894).
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np


TRAIN = Path("/tmp/wikitext_data/wiki.train.raw")
VALID = Path("/tmp/wikitext_data/wiki.valid.raw")


# ----------------------------------------------------------------------
# NBB — single-sample online
# ----------------------------------------------------------------------

class NBB:
    """Two-WTA-subset NBB faithful to nbb-xor/nbb_xor.py, scaled up.

    Differences from XOR stub:
      - n_input = 1 (bias) + 256 (one-hot byte). Only bias and the prev_byte
        unit fire.
      - n_hidden = configurable (1024 by default), one WTA subset.
      - n_output = 256, one WTA subset (byte prediction).
      - Tick count is configurable (default 5).
    """

    def __init__(
        self,
        n_input: int = 257,
        n_hidden: int = 1024,
        n_output: int = 256,
        lam: float = 0.005,
        eta: float = 0.005,
        init_lo: float = 0.999,
        init_hi: float = 1.001,
        seed: int = 0,
    ):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lam = lam
        self.eta = eta
        rng = np.random.default_rng(seed)
        self.W_ih = rng.uniform(init_lo, init_hi, (n_input, n_hidden)).astype(np.float32)
        self.W_ho = rng.uniform(init_lo, init_hi, (n_hidden, n_output)).astype(np.float32)
        self._alloc_state()

    def _alloc_state(self) -> None:
        ni, nh, no = self.n_input, self.n_hidden, self.n_output
        self.x_i = np.zeros(ni, dtype=np.float32)
        self.x_h = np.zeros(nh, dtype=np.float32)
        self.x_o = np.zeros(no, dtype=np.float32)
        self.c_ih = np.zeros((ni, nh), dtype=np.float32)
        self.c_ho = np.zeros((nh, no), dtype=np.float32)
        self.c_ih_prev = np.zeros_like(self.c_ih)
        self.c_ho_prev = np.zeros_like(self.c_ho)
        self.x_i_prev = np.zeros(ni, dtype=np.float32)
        self.x_h_prev = np.zeros(nh, dtype=np.float32)
        self.x_o_prev = np.zeros(no, dtype=np.float32)

    def _reset_state(self) -> None:
        self.x_i[:] = 0
        self.x_h[:] = 0
        self.x_o[:] = 0
        self.c_ih[:] = 0
        self.c_ho[:] = 0
        self.c_ih_prev[:] = 0
        self.c_ho_prev[:] = 0
        self.x_i_prev[:] = 0
        self.x_h_prev[:] = 0
        self.x_o_prev[:] = 0

    @staticmethod
    def _wta(net: np.ndarray) -> np.ndarray:
        """Largest-positive WTA, deterministic tiebreak (lowest index)."""
        x = np.zeros_like(net)
        if net.max() > 0:
            x[int(np.argmax(net))] = 1.0
        return x

    def _step(self, input_clamp: np.ndarray) -> None:
        # Snapshot tick t-1.
        self.x_i_prev[:] = self.x_i
        self.x_h_prev[:] = self.x_h
        self.x_o_prev[:] = self.x_o
        self.c_ih_prev[:] = self.c_ih
        self.c_ho_prev[:] = self.c_ho

        # c_ij(t) = x_i(t-1) * w_ij(t-1)
        np.multiply(self.x_i_prev[:, None], self.W_ih, out=self.c_ih)
        np.multiply(self.x_h_prev[:, None], self.W_ho, out=self.c_ho)

        net_h = self.c_ih.sum(axis=0)
        net_o = self.c_ho.sum(axis=0)

        # Apply activation rule.
        self.x_i = input_clamp.copy()
        self.x_h = self._wta(net_h)
        self.x_o = self._wta(net_o)

    def _bb_update(self, target_out_idx: int) -> None:
        """Bucket-brigade update for the current tick (per nbb_xor.py)."""
        active_h = self.x_h > 0  # (nh,)
        active_o = self.x_o > 0  # (no,)

        # ----- input -> hidden -----
        # Term 1: pay-out where hidden fires.
        # Only the hidden winner is True → only one column of W_ih shrinks per tick.
        delta_ih = -self.lam * self.c_ih * active_h[None, :]
        # Term 2: redistribute output-side payments back through hidden.
        paid_by_h = (self.lam * self.c_ho * active_o[None, :]).sum(axis=1)  # (nh,)
        denom_h = self.c_ih_prev.sum(axis=0)
        safe = denom_h > 1e-12
        if safe.any():
            share = np.zeros_like(self.W_ih)
            share[:, safe] = self.c_ih_prev[:, safe] / denom_h[safe]
            delta_ih += share * paid_by_h[None, :]

        # ----- hidden -> output -----
        delta_ho = -self.lam * self.c_ho * active_o[None, :]
        # No successor for output layer → no redistribution.
        # Term 3 (Ext): reward correct-output column.
        if active_o[target_out_idx]:
            delta_ho[:, target_out_idx] += self.eta * self.c_ho[:, target_out_idx]

        self.W_ih += delta_ih
        self.W_ho += delta_ho

    def present(
        self,
        prev_byte: int,
        target_byte: int,
        n_ticks: int = 5,
        learn: bool = True,
    ) -> int:
        """Present one (prev_byte, target_byte) pair. Return final output index.

        Input clamp = [bias=1, one_hot(prev_byte)]. Same shape as XOR's clamp
        but bigger.
        """
        self._reset_state()
        clamp = np.zeros(self.n_input, dtype=np.float32)
        clamp[0] = 1.0  # bias
        clamp[1 + prev_byte] = 1.0

        last_out = -1
        for _ in range(n_ticks):
            self._step(clamp)
            if learn:
                self._bb_update(target_byte)
            if self.x_o.sum() > 0:
                last_out = int(np.argmax(self.x_o))
        return last_out


# ----------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------

def evaluate(nbb: NBB, val_arr: np.ndarray, n_ticks: int, max_eval: int = 60_000) -> float:
    """Frozen-eval bigram accuracy on the first max_eval val bytes."""
    n = min(max_eval, val_arr.size - 1)
    correct = 0
    for i in range(n):
        out = nbb.present(int(val_arr[i]), int(val_arr[i + 1]),
                          n_ticks=n_ticks, learn=False)
        if out == int(val_arr[i + 1]):
            correct += 1
    return correct / n


def train(
    train_arr: np.ndarray,
    val_arr: np.ndarray,
    n_hidden: int = 1024,
    lam: float = 0.005,
    eta: float = 0.005,
    n_ticks: int = 5,
    max_seconds: float = 60.0,
    eval_every_s: float = 10.0,
    eval_chars: int = 5_000,
    seed: int = 0,
    log_path: Path | None = None,
) -> tuple[NBB, list[dict]]:
    nbb = NBB(n_hidden=n_hidden, lam=lam, eta=eta, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n_train = train_arr.size

    print(f"[nbb] n_hidden={n_hidden} lam={lam} eta={eta} ticks={n_ticks}")
    print(f"[nbb] training cap {max_seconds:.0f}s")

    history: list[dict] = []
    t0 = time.monotonic()
    next_eval = eval_every_s
    presentations = 0
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= max_seconds:
            break

        # Online: present one random (prev, target) pair.
        idx = int(rng.integers(0, n_train - 1))
        prev_b = int(train_arr[idx])
        target_b = int(train_arr[idx + 1])
        nbb.present(prev_b, target_b, n_ticks=n_ticks, learn=True)
        presentations += 1

        if elapsed >= next_eval:
            t_eval0 = time.monotonic()
            acc = evaluate(nbb, val_arr, n_ticks=n_ticks, max_eval=eval_chars)
            t_eval = time.monotonic() - t_eval0
            entry = {
                "elapsed_s": elapsed,
                "presentations": presentations,
                "val_acc": acc,
                "eval_chars": eval_chars,
                "eval_s": t_eval,
            }
            history.append(entry)
            print(f"  t={elapsed:5.1f}s  pres={presentations:8d}  "
                  f"val_acc={acc:.4f}  (eval on {eval_chars} chars in {t_eval:.1f}s)",
                  flush=True)
            next_eval = elapsed + eval_every_s + t_eval

    # Final eval on a larger slice.
    print()
    print("[nbb] final eval on 60K val chars...")
    t_eval0 = time.monotonic()
    final_acc = evaluate(nbb, val_arr, n_ticks=n_ticks, max_eval=60_000)
    history.append({
        "elapsed_s": time.monotonic() - t0,
        "presentations": presentations,
        "val_acc": final_acc,
        "eval_chars": 60_000,
        "eval_s": time.monotonic() - t_eval0,
        "final": True,
    })
    print(f"[nbb] final val acc (60K chars): {final_acc:.4f}   "
          f"({time.monotonic() - t_eval0:.1f}s)")

    if log_path:
        import json
        log_path.write_text(json.dumps(history, indent=2))
        print(f"[nbb] history written to {log_path}")

    return nbb, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-hidden", type=int, default=1024)
    p.add_argument("--lam", type=float, default=0.005)
    p.add_argument("--eta", type=float, default=0.005)
    p.add_argument("--ticks", type=int, default=5)
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--eval-every", type=float, default=10.0)
    p.add_argument("--eval-chars", type=int, default=5_000)
    p.add_argument("--train-mb", type=float, default=20.0,
                   help="MB of train data to sample from")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", type=str, default=None)
    args = p.parse_args()

    print("loading data...")
    t0 = time.monotonic()
    train_bytes = TRAIN.read_bytes()[: int(args.train_mb * 1_000_000)]
    val_bytes = VALID.read_bytes()
    train_arr = np.frombuffer(train_bytes, dtype=np.uint8)
    val_arr = np.frombuffer(val_bytes, dtype=np.uint8)
    print(f"  train: {train_arr.size:,} bytes   val: {val_arr.size:,} bytes "
          f"({time.monotonic() - t0:.1f}s)")
    print(f"  unigram floor:    0.1885")
    print(f"  bigram-table cap: 0.2894")
    print(f"  D2 pass:          ≥ 0.25")
    print()

    log_path = Path(args.log) if args.log else None
    train(
        train_arr=train_arr,
        val_arr=val_arr,
        n_hidden=args.n_hidden,
        lam=args.lam,
        eta=args.eta,
        n_ticks=args.ticks,
        max_seconds=args.max_seconds,
        eval_every_s=args.eval_every,
        eval_chars=args.eval_chars,
        seed=args.seed,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
