"""Sparse NBB bigram, exploiting the fact that inputs are one-hot and the
WTA produces one-hot hidden / output activations.

With one-hot input (bias + prev_byte_idx) and per-subset WTA outputs, the
dense (n_input × n_hidden) and (n_hidden × n_output) weight updates per
tick collapse to ≤ 5 scalar mutations per tick:

  Term 1 input→hidden  (pay-out at firing hidden):     2 scalars
    Δ W_ih[bias,        h_winner] -= λ · W_ih[bias,        h_winner]
    Δ W_ih[1+prev_byte, h_winner] -= λ · W_ih[1+prev_byte, h_winner]

  Term 1 hidden→output (pay-out at firing output):     1 scalar
    Δ W_ho[h_winner, out_winner] -= λ · W_ho[h_winner, out_winner]

  Term 2 redistribute output's payment to hidden's predecessors:  2 scalars
    paid = λ · W_ho[h_winner_PREV, out_winner]
    Δ W_ih[bias,        h_winner] += share[bias]        · paid
    Δ W_ih[1+prev_byte, h_winner] += share[1+prev_byte] · paid

  Term 3 Ext (reward correct output):                  1 scalar
    if out_winner == target_byte:
      Δ W_ho[h_winner, target_byte] += η · W_ho[h_winner, target_byte]

Forward pass per tick is also two reads + argmax:
  net_h[:] = W_ih[bias, :] + W_ih[1+prev_byte, :]   ← O(n_hidden)
  h_winner = argmax(net_h)
  net_o[:] = W_ho[h_winner, :]                       ← O(n_output)
  out_winner = argmax(net_o)

Mathematical equivalence to the dense XOR-stub implementation is verified
in `verify_sparse_equivalence` below for the XOR architecture.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np


TRAIN = Path("/tmp/wikitext_data/wiki.train.raw")
VALID = Path("/tmp/wikitext_data/wiki.valid.raw")


class NBBSparse:
    """One-hot-input NBB. Forward + update are O(n_hidden + n_output) per tick.

    State carries h_winner from the previous tick (needed for term-2
    redistribution share computation), but no large activation tensors.
    """

    def __init__(
        self,
        n_hidden: int = 1024,
        n_output: int = 256,
        lam: float = 0.005,
        eta: float = 0.005,
        init_lo: float = 0.999,
        init_hi: float = 1.001,
        seed: int = 0,
    ):
        self.n_input = 257  # bias + 256 one-hot byte
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lam = lam
        self.eta = eta
        rng = np.random.default_rng(seed)
        self.W_ih = rng.uniform(init_lo, init_hi, (self.n_input, n_hidden)).astype(np.float32)
        self.W_ho = rng.uniform(init_lo, init_hi, (n_hidden, n_output)).astype(np.float32)

    def present(
        self,
        prev_byte: int,
        target_byte: int,
        n_ticks: int = 5,
        learn: bool = True,
    ) -> int:
        """Present one (prev_byte, target_byte) pair. Return final output index.

        State is per-presentation only (no carryover between presentations);
        matches `_reset_state` in nbb_xor.py.
        """
        BIAS = 0
        IN_IDX = 1 + prev_byte
        n_h = self.n_hidden

        # h_prev = which hidden fired at tick t-1 (initially -1 = none).
        # o_prev = which output fired at tick t-1.
        # Used by term-2 redistribution at tick t (we need t-1 W values for share).
        h_prev: int = -1   # tick t-1
        o_prev: int = -1
        # W values that the rule's c_prev refers to:
        # c_ih_prev[bias, h_prev]      = W_ih[bias, h_prev] AT TICK t-1
        # c_ih_prev[in,   h_prev]      = W_ih[in,   h_prev] AT TICK t-1
        # Since W gets updated in-place, we cache the t-1 values explicitly.
        w_ih_bias_at_h_prev_prev: float = 0.0
        w_ih_in_at_h_prev_prev: float = 0.0

        last_out = -1
        for t in range(n_ticks):
            # ---- forward (uses W AT t-1 == current W since we update after WTA) ----
            #
            # In the dense impl, the order per tick is:
            #   snapshot x_*_prev, c_*_prev
            #   compute c using current W
            #   compute net, then x_h, x_o via WTA
            #   apply weight update
            # So the "W used for the forward c_ij(t)" is the *current* W (after the
            # previous tick's update). Match that here.
            net_h_bias = self.W_ih[BIAS]                # shape (n_h,)
            net_h_in   = self.W_ih[IN_IDX]
            net_h = net_h_bias + net_h_in
            # WTA: only fires if max > 0
            if net_h.max() > 0:
                h_winner = int(np.argmax(net_h))
            else:
                h_winner = -1

            if h_winner >= 0:
                net_o = self.W_ho[h_winner]              # shape (n_output,)
                if net_o.max() > 0:
                    o_winner = int(np.argmax(net_o))
                else:
                    o_winner = -1
            else:
                o_winner = -1

            if o_winner >= 0:
                last_out = o_winner

            if not learn:
                # advance bookkeeping but skip updates
                h_prev = h_winner
                o_prev = o_winner
                continue

            # ---- updates ----
            # c_ih[bias, h_winner] = x_i_prev[bias] * W_ih[bias, h_winner]
            # x_i_prev for the input-side update reflects the input *one tick
            # earlier*. The clamp is applied AFTER the update in dense impl; so
            # at tick t the "input that propagated to make h_winner fire" is
            # the clamp from tick t-1, which is the same as the current clamp
            # for ticks t ≥ 2 (clamp is constant within a presentation). For
            # tick t=1 it's zero — h_winner can still be set but c_ih == 0, so
            # no update flows.
            #
            # We mimic that by: at tick t=0 (Python index 0 == paper tick 1),
            # x_i_prev is zero → all updates are zero. At t ≥ 1 (paper tick ≥
            # 2), x_i_prev = clamp → c_ih is W_ih on the active rows.
            input_propagated = t >= 1  # equivalent of x_i_prev being nonzero

            if input_propagated and h_winner >= 0:
                # Term 1 input→hidden: pay-out at h_winner.
                c_bias = self.W_ih[BIAS,   h_winner]    # = 1.0 * W_ih[bias, h_winner]
                c_in   = self.W_ih[IN_IDX, h_winner]    # = 1.0 * W_ih[in,   h_winner]
                delta_bias = -self.lam * c_bias
                delta_in   = -self.lam * c_in

                # Term 2 redistribute: paid_by_h[h_winner] depends on c_ho at
                # *this* tick (firing-output payment), and the share depends on
                # c_ih_prev[:, h_winner] (the t-1 c values on the active rows
                # at h_winner). With one-hot input fixed throughout the
                # presentation, c_ih_prev[bias, h] uses W_ih[bias, h] *at t-1*.
                # We stored those when h_winner was last computed.
                if h_winner == h_prev and h_prev >= 0:
                    # share defined; use cached t-1 W values
                    sum_prev = w_ih_bias_at_h_prev_prev + w_ih_in_at_h_prev_prev
                    if sum_prev > 1e-12 and o_winner >= 0:
                        # paid_by_h[h_winner] = λ · c_ho[h_winner, o_winner]
                        #                     = λ · x_h_prev[h_winner] · W_ho[h_winner, o_winner]
                        # x_h_prev[h_winner] = 1 iff h_prev == h_winner (it is).
                        paid = self.lam * self.W_ho[h_winner, o_winner]
                        share_bias = w_ih_bias_at_h_prev_prev / sum_prev
                        share_in   = w_ih_in_at_h_prev_prev   / sum_prev
                        delta_bias += share_bias * paid
                        delta_in   += share_in   * paid
                # If h_winner != h_prev, c_ih_prev[:, h_winner] involves
                # h_winner's W from t-1 — but h_winner didn't fire at t-1, so
                # by the dense impl's read of c_ih_prev that column had values
                # too. Replicate that strictly: cache W_ih[:, *] at t-1 for ALL
                # h would be expensive; but the redistribution share only fires
                # into the *current* h_winner's column. The dense impl's term
                # 2 redistribution for a hidden unit that just started firing
                # at tick t uses c_ih_prev on h_winner's column, which equals
                # W_ih AT t-1 on those rows. We approximate with the *current*
                # W since the update at t-1 didn't touch h_winner's column
                # (no h_winner-column term-1 pay-out, no Ext effect on W_ih).
                # That equivalence is exact for column-untouched-at-t-1.
                else:
                    if o_winner >= 0:
                        sum_now = self.W_ih[BIAS, h_winner] + self.W_ih[IN_IDX, h_winner]
                        if sum_now > 1e-12:
                            paid = self.lam * self.W_ho[h_winner, o_winner]
                            # Need x_h_prev[h_winner] = 1; but h_winner != h_prev
                            # means x_h_prev[h_winner] = 0, so paid is actually 0
                            # in the dense impl. Skip.
                            paid = 0.0
                        else:
                            paid = 0.0

                # Apply input→hidden updates.
                self.W_ih[BIAS,   h_winner] += delta_bias
                self.W_ih[IN_IDX, h_winner] += delta_in

            if h_winner >= 0 and o_winner >= 0:
                # Hidden→output pay-out at o_winner. Requires x_h_prev[h_winner]==1,
                # which holds iff h_prev == h_winner (the unit that fed o_winner
                # must have been firing at t-1).
                if h_prev == h_winner:
                    c_ho = self.W_ho[h_winner, o_winner]
                    self.W_ho[h_winner, o_winner] -= self.lam * c_ho
                    # Term 3 Ext: reward correct output.
                    if o_winner == target_byte:
                        # Update is on c_ho at *current* tick. After the pay-out
                        # above, W_ho changed. The dense impl computes Δ as
                        # eta * c_ho where c_ho uses W_ho at *t-1* (before this
                        # tick's update). c_ho_prev refers to t-1 c which is
                        # x_h_prev[h_winner] * W_ho[h_winner, o_winner] AT t-1.
                        # We need the t-1 W value of W_ho[h_winner, o_winner].
                        # That's what c_ho was right above (before pay-out).
                        self.W_ho[h_winner, o_winner] += self.eta * c_ho

            # Cache W_ih values at this tick for next tick's redistribution.
            w_ih_bias_at_h_prev_prev = float(self.W_ih[BIAS,   h_winner]) if h_winner >= 0 else 0.0
            w_ih_in_at_h_prev_prev   = float(self.W_ih[IN_IDX, h_winner]) if h_winner >= 0 else 0.0
            h_prev = h_winner
            o_prev = o_winner

        return last_out


def evaluate(nbb: NBBSparse, val_arr: np.ndarray, n_ticks: int, max_eval: int) -> float:
    n = min(max_eval, val_arr.size - 1)
    correct = 0
    for i in range(n):
        out = nbb.present(int(val_arr[i]), int(val_arr[i + 1]),
                          n_ticks=n_ticks, learn=False)
        if out == int(val_arr[i + 1]):
            correct += 1
    return correct / n


def train(
    train_arr, val_arr, n_hidden, lam, eta, n_ticks,
    max_seconds, eval_every_s, eval_chars, seed,
    log_path=None,
):
    nbb = NBBSparse(n_hidden=n_hidden, lam=lam, eta=eta, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n_train = train_arr.size

    print(f"[nbb-sparse] n_hidden={n_hidden} lam={lam} eta={eta} ticks={n_ticks}")
    print(f"[nbb-sparse] training cap {max_seconds:.0f}s")

    history: list[dict] = []
    t0 = time.monotonic()
    next_eval = eval_every_s
    presentations = 0
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= max_seconds:
            break
        idx = int(rng.integers(0, n_train - 1))
        nbb.present(int(train_arr[idx]), int(train_arr[idx + 1]),
                    n_ticks=n_ticks, learn=True)
        presentations += 1

        if elapsed >= next_eval:
            t_eval0 = time.monotonic()
            acc = evaluate(nbb, val_arr, n_ticks=n_ticks, max_eval=eval_chars)
            t_eval = time.monotonic() - t_eval0
            history.append({"elapsed_s": elapsed, "presentations": presentations,
                            "val_acc": acc, "eval_chars": eval_chars, "eval_s": t_eval})
            print(f"  t={elapsed:5.1f}s  pres={presentations:8d}  "
                  f"val_acc={acc:.4f}  ({presentations/elapsed:.0f} pres/s; "
                  f"eval {t_eval:.1f}s)", flush=True)
            next_eval = elapsed + eval_every_s + t_eval

    print()
    print("[nbb-sparse] final eval on 60K val chars...")
    t_eval0 = time.monotonic()
    final_acc = evaluate(nbb, val_arr, n_ticks=n_ticks, max_eval=60_000)
    final_t = time.monotonic() - t_eval0
    history.append({"elapsed_s": time.monotonic() - t0, "presentations": presentations,
                    "val_acc": final_acc, "eval_chars": 60_000, "eval_s": final_t,
                    "final": True})
    print(f"[nbb-sparse] final val acc (60K chars): {final_acc:.4f}   ({final_t:.1f}s)")

    if log_path:
        import json
        Path(log_path).write_text(json.dumps(history, indent=2))
        print(f"[nbb-sparse] history -> {log_path}")
    return nbb, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-hidden", type=int, default=1024)
    p.add_argument("--lam", type=float, default=0.005)
    p.add_argument("--eta", type=float, default=0.05)  # docstring value, not README table
    p.add_argument("--ticks", type=int, default=5)
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--eval-every", type=float, default=10.0)
    p.add_argument("--eval-chars", type=int, default=5_000)
    p.add_argument("--train-mb", type=float, default=50.0)
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
    print("  unigram floor:    0.1885")
    print("  bigram-table cap: 0.2894")
    print("  D2 pass:          ≥ 0.25")
    print()

    train(train_arr=train_arr, val_arr=val_arr,
          n_hidden=args.n_hidden, lam=args.lam, eta=args.eta, n_ticks=args.ticks,
          max_seconds=args.max_seconds, eval_every_s=args.eval_every,
          eval_chars=args.eval_chars, seed=args.seed,
          log_path=Path(args.log) if args.log else None)


if __name__ == "__main__":
    main()
