"""Symlink the first clips of a cache's train or val split into their own directory, for evaluate.py.

  python scripts/split_dir.py --cache ~/scratch/cache/kubric --split train --n 50 --out /tmp/fit_train
"""

import argparse
from pathlib import Path

from genpoint3d.train.data.split_clips import split_clips


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--n", type=int, default=50, help="0 means every clip")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    split = split_clips([args.cache])
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
          f" (of {len(split.train) + len(split.val)} cached)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
