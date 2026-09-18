"""
Plot a training run's `log.json` -- the standard diagnostic figure.

Two panels, because they answer different questions:

  losses  train and val on the same axes. Both falling together means the model
          is still learning. Train falling while val flattens is OVERFITTING --
          more steps will not help, more data or augmentation will. Both stuck
          high is UNDERFITTING: the model is too small, the learning rate is
          wrong, or something is broken.
  APD     the TAP-Vid-3D metric, which is what a paper would report. It can
          keep improving after val_loss flattens, and it is the number to pick
          a checkpoint on.

Run:  python scripts/plot_log.py local/outputs/run/log.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no display on a compute node
import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path, help="log.json from a training run")
    p.add_argument("--out", type=Path, default=None, help="default: alongside the log")
    args = p.parse_args()

    entries = json.loads(args.log.read_text())
    if not entries:
        raise SystemExit(f"{args.log} is empty -- no validation has run yet")

    steps = [e["step"] for e in entries]
    pick = lambda k: [e[k] for e in entries if k in e]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    train, val = pick("train_loss"), pick("val_loss")
    if train:
        ax1.plot(steps[-len(train):], train, label="train", marker="o", ms=3)
    ax1.plot(steps[-len(val):], val, label="val", marker="o", ms=3)
    ax1.set_xlabel("step"); ax1.set_ylabel("flow matching loss")
    ax1.set_title("loss -- a widening gap is overfitting")
    ax1.legend(); ax1.grid(alpha=0.3)

    apd, base = pick("average_pts_within_thresh"), pick("apd_baseline")
    if apd:
        ax2.plot(steps[-len(apd):], apd, marker="o", ms=3, color="tab:green",
                 label="model")
    if base:
        # The same metric on the mean trajectory: the model has to clear this
        # line to have learnt anything at all.
        ax2.plot(steps[-len(base):], base, ls="--", color="tab:grey",
                 label="mean-trajectory baseline")
        ax2.legend()
    ax2.set_xlabel("step"); ax2.set_ylabel("APD")
    ax2.set_title("APD (TAP-Vid-3D) -- higher is better")
    ax2.set_ylim(0, 1); ax2.grid(alpha=0.3)

    out = args.out or args.log.with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    # Also say it in words, since the figure needs copying off the cluster.
    last = entries[-1]
    print(f"\nlast validation, step {last['step']}:")
    for k in ("train_loss", "val_loss", "average_pts_within_thresh", "apd_baseline"):
        if k in last:
            print(f"  {k:<26} {last[k]:.4f}")
    if len(entries) >= 2 and "train_loss" in last:
        prev = entries[-2]
        d_train = last["train_loss"] - prev["train_loss"]
        d_val = last["val_loss"] - prev["val_loss"]
        if d_train < 0 and d_val >= 0:
            print("\n  train improving while val is not -> OVERFITTING;"
                  " more steps will not help")
        elif d_train >= 0 and d_val >= 0:
            print("\n  neither improving -> check the learning rate before"
                  " assuming the model is too small")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
