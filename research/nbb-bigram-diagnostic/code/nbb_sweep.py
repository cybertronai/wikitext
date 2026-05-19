"""Sweep over (λ, η) for NBB bigram to map the failure regime.

Per-presentation expected weight change on the modal-byte connection, for
a stochastic target with modal-byte probability p:

    E[Δ W / W] = p · (η - λ)  +  (1 - p) · (-λ)
              = p η - λ

  λ = η:       E[ΔW/W] = (p-1) λ < 0  for p < 1      ← always dissipates
  η > λ / p:   E[ΔW/W] > 0                            ← exponential blow-up
  η = λ / p:   E[ΔW/W] = 0                            ← unstable equilibrium

For bigrams the per-byte modal-byte probability ranges roughly 0.05–0.95 across
the 256 byte vocabulary, so no single η works for all bytes. This sweep
verifies the prediction empirically and records val_acc, W-norm trajectories,
and overflow flags. Output goes to results/sweep.json.
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np

from nbb_bigram_sparse import NBBSparse, TRAIN, VALID, evaluate


def run_one(
    train_arr,
    val_arr,
    n_hidden: int,
    lam: float,
    eta: float,
    n_ticks: int,
    max_seconds: float,
    eval_chars: int,
    seed: int,
) -> dict:
    nbb = NBBSparse(n_hidden=n_hidden, lam=lam, eta=eta, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n_train = train_arr.size

    accs: list[tuple[float, int, float]] = []  # (t, pres, acc)
    w_stats: list[tuple[float, float, float, float, float]] = []  # (t, ih_max, ih_min, ho_max, ho_min)
    overflow = False
    nan_at_pres: int | None = None

    t0 = time.monotonic()
    next_log = 5.0
    presentations = 0

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            while True:
                elapsed = time.monotonic() - t0
                if elapsed >= max_seconds:
                    break
                idx = int(rng.integers(0, n_train - 1))
                nbb.present(int(train_arr[idx]), int(train_arr[idx + 1]),
                            n_ticks=n_ticks, learn=True)
                presentations += 1

                if elapsed >= next_log:
                    acc = evaluate(nbb, val_arr, n_ticks=n_ticks, max_eval=eval_chars)
                    accs.append((elapsed, presentations, acc))
                    ih = nbb.W_ih
                    ho = nbb.W_ho
                    w_stats.append((elapsed,
                                    float(ih.max()), float(ih.min()),
                                    float(ho.max()), float(ho.min())))
                    next_log += 5.0
        except RuntimeWarning as e:
            overflow = True
            nan_at_pres = presentations

    # Final accuracy (may be nan-driven 0).
    if not (np.isnan(nbb.W_ih).any() or np.isnan(nbb.W_ho).any()):
        final = evaluate(nbb, val_arr, n_ticks=n_ticks, max_eval=eval_chars * 4)
    else:
        final = 0.0

    return {
        "lam": lam, "eta": eta, "n_hidden": n_hidden, "n_ticks": n_ticks,
        "seed": seed, "max_seconds": max_seconds,
        "presentations": presentations,
        "overflow": overflow, "nan_at_pres": nan_at_pres,
        "trajectory": [{"t": t, "pres": p, "acc": a} for t, p, a in accs],
        "w_stats": [{"t": t, "ih_max": a, "ih_min": b, "ho_max": c, "ho_min": d}
                    for t, a, b, c, d in w_stats],
        "final_acc": final,
        "w_ih_has_nan": bool(np.isnan(nbb.W_ih).any()),
        "w_ho_has_nan": bool(np.isnan(nbb.W_ho).any()),
        "w_ih_max_final": float(np.nanmax(nbb.W_ih)) if not np.isnan(nbb.W_ih).all() else float("nan"),
        "w_ho_max_final": float(np.nanmax(nbb.W_ho)) if not np.isnan(nbb.W_ho).all() else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-hidden", type=int, default=1024)
    p.add_argument("--ticks", type=int, default=5)
    p.add_argument("--max-seconds", type=float, default=20.0)
    p.add_argument("--eval-chars", type=int, default=2000)
    p.add_argument("--train-mb", type=float, default=20.0)
    p.add_argument("--out", type=str, default="experiments/nbb_bigram/results/sweep.json")
    args = p.parse_args()

    print("loading data...")
    train_bytes = TRAIN.read_bytes()[: int(args.train_mb * 1_000_000)]
    val_bytes = VALID.read_bytes()
    train_arr = np.frombuffer(train_bytes, dtype=np.uint8)
    val_arr = np.frombuffer(val_bytes, dtype=np.uint8)
    print(f"  train: {train_arr.size:,} bytes   val: {val_arr.size:,} bytes")
    print("  unigram floor:    0.1885")
    print("  bigram-table cap: 0.2894")
    print()

    # Per the analysis at top: E[ΔW/W] per presentation = p·η − λ.
    # Sweep eta from λ/4 (dissipate) through 4λ (blow up).
    lam = 0.005
    etas = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050]
    results = []
    for eta in etas:
        print(f"\n=== lam={lam}  eta={eta}  η/λ={eta/lam:.1f} ===")
        res = run_one(
            train_arr, val_arr,
            n_hidden=args.n_hidden, lam=lam, eta=eta, n_ticks=args.ticks,
            max_seconds=args.max_seconds, eval_chars=args.eval_chars, seed=0,
        )
        results.append(res)
        traj_summary = " → ".join(f"{x['acc']:.3f}" for x in res["trajectory"])
        print(f"  presentations: {res['presentations']:,}")
        print(f"  overflow/NaN:  {res['overflow']}  W_ho max={res['w_ho_max_final']:.2e}")
        print(f"  acc trajectory: {traj_summary or '(no checkpoints)'}")
        print(f"  final acc (8K val): {res['final_acc']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsweep results → {out}")

    print("\n--- summary ---")
    print(f"{'eta':>8s} {'η/λ':>5s} {'final_acc':>9s} {'overflow':>9s} {'W_ho_max':>10s}")
    for r in results:
        print(f"{r['eta']:>8.4f} {r['eta']/r['lam']:>5.1f} "
              f"{r['final_acc']:>9.4f} {str(r['overflow']):>9s} "
              f"{r['w_ho_max_final']:>10.2e}")


if __name__ == "__main__":
    main()
