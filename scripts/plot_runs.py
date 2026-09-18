"""
Plot one or more runs' `log.json` on shared axes.

`plot_log.py` draws a single run. This one overlays several, which is what a
comparison needs: a run is only interpretable against the one it changed.

Run:  python scripts/plot_runs.py runs/run3493.json runs/run3493_apdloss.json
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

    runs = [(f.stem, json.loads(f.read_text())) for f in args.logs]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    colours = plt.cm.viridis([i / max(len(runs) - 1, 1) * 0.8 for i in range(len(runs))])

    for (name, e), c in zip(runs, colours):
        s = [x["step"] for x in e]
        ax[0].plot(s, [x["average_pts_within_thresh"] for x in e],
                   marker="o", ms=3, color=c, label=name)
        ax[1].plot(s, [x["pts_within_1"] for x in e], marker="o", ms=3, color=c, label=name)
        # Train and val on one axis: the gap between them is the overfitting.
        ax[2].plot(s, [x["train_loss"] for x in e], color=c, label=f"{name} train")
        ax[2].plot(s, [x["val_loss"] for x in e], color=c, ls="--", label=f"{name} val")

    ax[0].set_ylabel("APD"); ax[0].set_title("APD (TAP-Vid-3D) — higher is better")
    ax[1].set_ylabel("pts_within_1"); ax[1].set_yscale("log")
    ax[1].set_title("precision at 1px — log scale")
    ax[2].set_ylabel("loss"); ax[2].set_yscale("log")
    ax[2].set_title("loss — solid train, dashed val")
    for a in ax:
        a.set_xlabel("step"); a.grid(alpha=.3); a.legend(fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    for name, e in runs:
        best = max(e, key=lambda x: x["average_pts_within_thresh"])
        print(f"  {name:22} best APD {best['average_pts_within_thresh']:.4f}"
              f" at step {best['step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
