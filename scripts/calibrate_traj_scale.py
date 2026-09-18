"""
Measure `TRAJ_SCALE` -- the dataset-level constant in
`genpoint3d/data/transform.py`.

Flow matching mixes the denoising target with `x0 ~ N(0, I)`, so the target
must have roughly unit variance -- otherwise the interpolant is nearly pure
noise and the model learns only to output `-x0`.

Scene normalisation makes geometry O(1) but leaves trajectories well below
that. This measures the remaining spread, pooled over the whole training set.
One global number, not per-clip, so no individual sample's own motion leaks
into its normalisation -- and so a fast clip stays genuinely faster than a slow
one.

Reads the CACHE, which already holds metres and every scale statistic, so this
is arithmetic over small tensors rather than a second pass over the images.
The answer depends on `--norm-mode`, because a different scale divisor leaves a
different spread behind.

Run:  python scripts/calibrate_traj_scale.py --cache local/cache/kubric
Then paste the printed value into `TRAJ_SCALE`.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from genpoint3d.data.cache import CachedClip
from genpoint3d.data.transform import TRAJ_SCALE


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="cache DIRECTORY from scripts/preprocess.py")
    p.add_argument("--norm-mode", default="median",
                   choices=["median", "mean", "centroid_max"])
    # A displacement target is much smaller than an absolute one -- the scene's
    # position cancels and only motion is left -- so it needs its own constant.
    # Reusing the absolute one would leave the target far below unit variance,
    # which is the failure check_transform.py's scale sanity test exists to catch.
    p.add_argument("--target", default="absolute",
                   choices=["absolute", "displacement"],
                   help="displacement measures traj - traj[0]")
    # A standard deviation over a few hundred clips is already ~1M visible
    # coordinates. Reading all 3500 buys no precision and costs minutes.
    p.add_argument("--sample", type=int, default=400,
                   help="clips to sample; 0 uses every clip")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    files = sorted(Path(args.cache).glob("*.pt"))
    if not files:
        raise SystemExit(f"no cached clips in {args.cache}")
    if args.sample and args.sample < len(files):
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(files), generator=g)[: args.sample].tolist()
        files = [files[i] for i in sorted(idx)]   # random, not the first N

    # Accumulate sum and sum-of-squares rather than keeping every point, so the
    # memory does not grow with the dataset.
    total = sq_total = n = 0.0
    per_clip = []
    t0 = time.time()
    for i, f in enumerate(files, 1):
        # mmap: a cached clip is ~7.3 MB and almost all of it is DINOv3
        # features we never touch here. Without this every clip ships its
        # whole context tensor across the network for four numbers.
        clip = CachedClip.from_dict(torch.load(f, weights_only=False, mmap=True))
        # traj_scale=1.0 leaves the trajectory in raw scene-normalised units,
        # which is exactly the spread we are trying to measure.
        traj, _ = clip.normalised(args.norm_mode, traj_scale=1.0)
        if args.target == "displacement":
            traj = traj - traj[:1]
        v = traj[clip.visibility]
        total += v.sum().item()
        sq_total += v.pow(2).sum().item()
        n += v.numel()
        per_clip.append(v.std().item())
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {(time.time() - t0) / i * 1000:.0f} ms/clip",
                  flush=True)

    mean = total / n
    scale = (sq_total / n - mean ** 2) ** 0.5
    per_clip = torch.tensor(per_clip)

    print(f"pooled over {len(files)} clips, {int(n):,} visible coordinates"
          f"  (--norm-mode {args.norm_mode}, --target {args.target})")
    print(f"  per-clip std: median {per_clip.median():.4f}"
          f"  min {per_clip.min():.4f}  max {per_clip.max():.4f}")
    print(f"\n  TRAJ_SCALE = {scale:.4f}      (current: {TRAJ_SCALE})")
    print("\nPaste that into TRAJ_SCALE in genpoint3d/data/transform.py.")
    if len(files) < 50:
        print("NOTE: too few clips for a real constant -- this is a sanity "
              "check, not a calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
