"""
Overlay several runs' `log.json` on shared axes.

`plot_log.py` draws a single run. A run is only interpretable against the one
it changed, so comparisons need them on the same axes.

Runs are labelled by position, so the legend stays readable as names grow;
the table in README.md maps each number to what changed.

Run:  python runs/plot.py runs/*.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no display on a compute node
import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/compare.png"))
    args = p.parse_args()

    runs = [(f"run {i}", json.loads(f.read_text()))
            for i, f in enumerate(args.logs, 1)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colours = plt.cm.viridis([i / max(len(runs) - 1, 1) * 0.75 for i in range(len(runs))])

    for (name, e), c in zip(runs, colours):
        s = [x["step"] for x in e]
        ax[0].plot(s, [x["average_pts_within_thresh"] for x in e],
                   marker="o", ms=3, color=c, label=name)

        # Solid train, dashed val: the gap between a pair is the overfitting.
        ax[1].plot(s, [x["train_loss"] for x in e], color=c, label=f"{name} train")
        ax[1].plot(s, [x["val_loss"] for x in e], color=c, ls="--", label=f"{name} val")

    ax[0].set_ylabel(r"APD (average percent of points within $\delta$ error)")
    ax[1].set_ylabel("loss")
    ax[1].set_yscale("log")
    for a in ax:
        a.set_xlabel("step")
        a.grid(alpha=.3)
        a.legend(fontsize=8)

    # TAPVid-3D's Static Baseline: a flat line every run has to clear. Measured
    # once by scripts/static_baseline.py -- it is a property of the data, not of
    # training -- and read from a run's own log if it recorded one.
    static = next((e[0]["apd_static"] for _, e in runs if "apd_static" in e[0]), None)
    if static is None:
        f = args.out.parent / "static_baseline.json"
        if f.exists():
            static = json.loads(f.read_text())["average_pts_within_thresh"]
    if static is not None:
        ax[0].axhline(static, ls=":", c="grey", lw=1.2, label="static baseline")
        ax[0].legend(fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    for (name, e), f in zip(runs, args.logs):
        best = max(e, key=lambda x: x["average_pts_within_thresh"])
        print(f"  {name}  {f.stem:22} best APD "
              f"{best['average_pts_within_thresh']:.4f} at step {best['step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
