"""NBB with k-byte context — the per-user follow-up to D2.

The 1-byte-context bigram diagnostic failed structurally: per-presentation
E[ΔW/W] = p_modal · η − λ, and the bigram p_modal distribution (weighted
average 0.288) sits far below η/λ = 1, guaranteeing dissipation. At k=4
context the weighted average p_modal climbs to 0.59; at k=8 to 0.82. If
NBB's failure is purely a function of target stochasticity, it should
work much better at k=4 or k=8.

Architecture (multi-slot one-hot input — the natural extension of the
XOR bias + x1 + x2 layout):

    1 bias unit + k·256 one-hot byte slots                  → 1 + 256k input units
    n_hidden WTA subset
    256 output WTA subset

Per presentation, active input units = 1 (bias) + k (one byte per slot).
Same sparse update structure as the bigram version: forward = (k+1) reads,
each per-tick weight update = (k+1) scalar mutations on the input side, ≤2
scalars on the output side.
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np


TRAIN = Path("/tmp/wikitext_data/wiki.train.raw")
VALID = Path("/tmp/wikitext_data/wiki.valid.raw")


class NBBKGram:
    """k-byte context NBB. n_input = 1 + 256·k. Update is O(k) scalars per tick."""

    def __init__(
        self,
        k: int,
        n_hidden: int = 4096,
        n_output: int = 256,
        lam: float = 0.005,
        eta: float = 0.005,
        init_lo: float = 0.999,
        init_hi: float = 1.001,
        seed: int = 0,
    ):
        self.k = k
        self.n_input = 1 + 256 * k
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lam = lam
        self.eta = eta
        rng = np.random.default_rng(seed)
        self.W_ih = rng.uniform(init_lo, init_hi, (self.n_input, n_hidden)).astype(np.float32)
        self.W_ho = rng.uniform(init_lo, init_hi, (n_hidden, n_output)).astype(np.float32)

    def _input_indices(self, context_bytes) -> list[int]:
        # bias + slot-0-byte + slot-1-byte + ...
        # context_bytes is a length-k sequence of byte values
        return [0] + [1 + 256 * s + int(context_bytes[s]) for s in range(self.k)]

    def present(self, context_bytes, target_byte: int, n_ticks: int = 5, learn: bool = True) -> int:
        active_in = self._input_indices(context_bytes)  # list of (k+1) ints
        n_h = self.n_hidden

        h_prev = -1
        o_prev = -1
        # W_ih values at t-1 on the (active_in × h_prev) cell — for term-2 share.
        w_at_h_prev_prev = [0.0] * len(active_in)

        last_out = -1
        for t in range(n_ticks):
            # forward: net_h = sum_{i in active} W_ih[i, :]
            net_h = self.W_ih[active_in[0]].copy()
            for i in active_in[1:]:
                net_h += self.W_ih[i]
            h_winner = int(np.argmax(net_h)) if net_h.max() > 0 else -1

            if h_winner >= 0:
                net_o = self.W_ho[h_winner]
                o_winner = int(np.argmax(net_o)) if net_o.max() > 0 else -1
            else:
                o_winner = -1

            if o_winner >= 0:
                last_out = o_winner

            if not learn:
                h_prev = h_winner
                o_prev = o_winner
                continue

            # input propagated only at t >= 1 (tick-1 has x_i_prev = 0)
            input_propagated = t >= 1

            if input_propagated and h_winner >= 0:
                # Term 1 input→hidden: pay-out (one update per active input)
                c_vals = [self.W_ih[i, h_winner] for i in active_in]  # c_ih[i, h_winner] = 1*W_ih[i, h_winner]
                deltas = [-self.lam * c for c in c_vals]

                # Term 2 redistribute (only if h_winner == h_prev and o fired)
                if h_winner == h_prev and h_prev >= 0 and o_winner >= 0:
                    sum_prev = sum(w_at_h_prev_prev)
                    if sum_prev > 1e-12:
                        paid = self.lam * self.W_ho[h_winner, o_winner]
                        for j, w_prev in enumerate(w_at_h_prev_prev):
                            deltas[j] += (w_prev / sum_prev) * paid

                # Apply input→hidden updates
                for i, d in zip(active_in, deltas):
                    self.W_ih[i, h_winner] += d

            if h_winner >= 0 and o_winner >= 0 and h_prev == h_winner:
                c_ho_val = self.W_ho[h_winner, o_winner]
                self.W_ho[h_winner, o_winner] -= self.lam * c_ho_val
                if o_winner == target_byte:
                    self.W_ho[h_winner, o_winner] += self.eta * c_ho_val

            # Cache for next tick
            if h_winner >= 0:
                w_at_h_prev_prev = [float(self.W_ih[i, h_winner]) for i in active_in]
            else:
                w_at_h_prev_prev = [0.0] * len(active_in)
            h_prev = h_winner
            o_prev = o_winner

        return last_out


def evaluate(nbb: NBBKGram, val_arr: np.ndarray, n_ticks: int, max_eval: int) -> float:
    k = nbb.k
    n = min(max_eval, val_arr.size - k)
    correct = 0
    for i in range(n):
        ctx = val_arr[i:i+k]
        out = nbb.present(ctx, int(val_arr[i + k]), n_ticks=n_ticks, learn=False)
        if out == int(val_arr[i + k]):
            correct += 1
    return correct / n


def train(
    train_arr, val_arr, k, n_hidden, lam, eta, n_ticks,
    max_seconds, eval_every_s, eval_chars, seed,
    log_path=None,
):
    nbb = NBBKGram(k=k, n_hidden=n_hidden, lam=lam, eta=eta, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n_train = train_arr.size

    print(f"[nbb-k] k={k}  n_input={nbb.n_input}  n_hidden={n_hidden}  "
          f"lam={lam}  eta={eta}  ticks={n_ticks}")
    print(f"[nbb-k] W_ih: {nbb.W_ih.nbytes/1e6:.0f} MB   W_ho: {nbb.W_ho.nbytes/1e6:.0f} MB")
    print(f"[nbb-k] training cap {max_seconds:.0f}s")

    history = []
    t0 = time.monotonic()
    next_eval = eval_every_s
    presentations = 0

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        overflow = False
        try:
            while True:
                elapsed = time.monotonic() - t0
                if elapsed >= max_seconds:
                    break
                idx = int(rng.integers(0, n_train - k - 1))
                ctx = train_arr[idx:idx+k]
                tgt = int(train_arr[idx + k])
                nbb.present(ctx, tgt, n_ticks=n_ticks, learn=True)
                presentations += 1

                if elapsed >= next_eval:
                    acc = evaluate(nbb, val_arr, n_ticks=n_ticks, max_eval=eval_chars)
                    history.append({"elapsed_s": elapsed, "presentations": presentations,
                                    "val_acc": acc, "eval_chars": eval_chars})
                    print(f"  t={elapsed:5.1f}s  pres={presentations:8d}  "
                          f"val_acc={acc:.4f}  ({presentations/elapsed:.0f} pres/s)",
                          flush=True)
                    next_eval += eval_every_s
        except RuntimeWarning as e:
            overflow = True
            print(f"  *** OVERFLOW at presentation {presentations} ***")

    print()
    final_acc = evaluate(nbb, val_arr, n_ticks=n_ticks, max_eval=10000)
    history.append({"elapsed_s": time.monotonic() - t0, "presentations": presentations,
                    "val_acc": final_acc, "eval_chars": 10000, "final": True})
    print(f"[nbb-k] final val acc (10K chars): {final_acc:.4f}")
    print(f"[nbb-k] overflow during training: {overflow}")
    print(f"[nbb-k] W_ih max final: {np.nanmax(nbb.W_ih):.4e}")
    print(f"[nbb-k] W_ho max final: {np.nanmax(nbb.W_ho):.4e}")

    if log_path:
        Path(log_path).write_text(json.dumps(history, indent=2))
    return nbb, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n-hidden", type=int, default=4096)
    p.add_argument("--lam", type=float, default=0.005)
    p.add_argument("--eta", type=float, default=0.005)
    p.add_argument("--ticks", type=int, default=5)
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--eval-every", type=float, default=15.0)
    p.add_argument("--eval-chars", type=int, default=3000)
    p.add_argument("--train-mb", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", type=str, default=None)
    args = p.parse_args()

    print("loading data...")
    train_bytes = TRAIN.read_bytes()[: int(args.train_mb * 1_000_000)]
    val_bytes = VALID.read_bytes()
    train_arr = np.frombuffer(train_bytes, dtype=np.uint8)
    val_arr = np.frombuffer(val_bytes, dtype=np.uint8)
    print(f"  train: {train_arr.size:,} bytes   val: {val_arr.size:,} bytes")
    print(f"  unigram floor:   0.1885")
    print(f"  bigram-table:    0.2894  (k=1 oracle)")
    print(f"  k=4 oracle (wt avg p_modal): ~0.59")
    print(f"  k=8 oracle (wt avg p_modal): ~0.82")
    print()

    train(train_arr=train_arr, val_arr=val_arr,
          k=args.k, n_hidden=args.n_hidden, lam=args.lam, eta=args.eta,
          n_ticks=args.ticks, max_seconds=args.max_seconds,
          eval_every_s=args.eval_every, eval_chars=args.eval_chars,
          seed=args.seed, log_path=Path(args.log) if args.log else None)


if __name__ == "__main__":
    main()
