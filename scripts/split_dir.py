"""Symlink a slice of one train/val split into its own directory.

`evaluate.py` scores every clip in a cache directory, which is right for a test
cache but gives no way to ask "how does it do on clips it was TRAINED on". That
question is the bias/variance check: a model that cannot score well on data it
has already seen is underfitting, and no amount of extra data will help it.

The split is reproduced exactly as `train.py:split` does it -- same generator,
same seed, same val_frac -- so the clips here are the ones that run really did
train or validate on.

  python scripts/split_dir.py --cache ~/scratch/cache/kubric \
      --split train --n 50 --out /tmp/fit_train
"""

import argparse
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--n", type=int, default=50, help="0 means every clip")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    clips = sorted(Path(args.cache).glob("*.pt"))
    if not clips:
        raise SystemExit(f"no cached clips in {args.cache}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * args.val_frac))
    chosen = [clips[i] for i in (perm[n_val:] if args.split == "train" else perm[:n_val])]
    if args.n:
        chosen = chosen[: args.n]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.pt"):
        old.unlink()
    for c in chosen:
        (out / c.name).symlink_to(c.resolve())

    print(f"{len(chosen)} {args.split} clips -> {out}"
          f" (of {len(clips)} cached, val_frac {args.val_frac}, seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
