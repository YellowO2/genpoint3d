"""Symlink a slice of one train/val split into its own directory.

`evaluate.py` scores every clip in a cache directory, which is right for a test
cache but gives no way to ask "how does it do on clips it was TRAINED on". That
question is the bias/variance check: a model that cannot score well on data it
has already seen is underfitting, and no amount of extra data will help it.

The split is the run's own (`split_clips`, with the val_frac and seed
in its run.json under `--run`, the defaults every run uses otherwise), so the
clips here are the ones that run really did train or validate on.

  python scripts/split_dir.py --cache ~/scratch/cache/kubric \
      --split train --n 50 --run outputs/RUN --out /tmp/fit_train
"""

import argparse
import json
from pathlib import Path

from genpoint3d.train.data.split_clips import SEED, VAL_FRAC, split_clips


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--n", type=int, default=50, help="0 means every clip")
    p.add_argument("--run", default=None, help="the run's directory, whose run.json gives its split")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    val_frac, seed = VAL_FRAC, SEED
    if args.run:
        a = json.loads((Path(args.run) / "run.json").read_text())["args"]
        val_frac, seed = a["val_frac"], a["seed"]
    split = split_clips([args.cache], val_frac, seed)
    chosen = split.train if args.split == "train" else split.val
    if args.n:
        chosen = chosen[: args.n]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.pt"):
        old.unlink()
    for c in chosen:
        (out / c.name).symlink_to(c.resolve())

    print(f"{len(chosen)} {args.split} clips -> {out}"
          f" (of {len(split.train) + len(split.val)} cached, val_frac {val_frac}, seed {seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
