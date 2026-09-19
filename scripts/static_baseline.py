"""
Measure TAPVid-3D's Static Baseline on a cache.

The baseline takes each query point's 3D position and assumes it never moves.
It is the floor a tracker has to clear: a model scoring below it would be
beaten by predicting no motion at all. TAPVid-3D reports 9.4 for it (table 3),
above TAPIR-3D's 5.9 and not far below SpatialTracker's 15.5, so it is a real
bar rather than a formality.

No model and no sampling, so this is fast -- the cost is reading clips.

Measured on the SAME val split training uses, so the number is directly
comparable to a run's logged APD rather than to a different set of clips.

Run:  python scripts/static_baseline.py --cache ~/scratch/cache/kubric
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from genpoint3d.eval.metrics import tapvid3d_metrics
from train import ClipDataset, split, to_device, to_metres


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--sample", type=int, default=100,
                   help="val clips to score; 0 uses the whole val split")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--points", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--norm-mode", default="median")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/static_baseline.json")
    args = p.parse_args()

    _, val_clips, _, _ = split(args.cache, args.val_frac, args.seed)
    if args.sample and args.sample < len(val_clips):
        val_clips = val_clips[: args.sample]
    print(f"{len(val_clips)} val clips", flush=True)

    # resample=False so the number does not change between invocations.
    loader = DataLoader(
        ClipDataset(val_clips, args.points, resample=False, norm_mode=args.norm_mode),
        batch_size=args.batch, shuffle=False, num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sums, n = {}, 0
    t0 = time.time()
    for batch in loader:
        b = to_device(batch, device)
        gt = to_metres(b["traj"], b)
        # Every frame equals frame 0: the point never moved.
        m = tapvid3d_metrics(gt[:, :1].expand_as(gt), gt, b["visibility"],
                             b["intrinsics"], b["extrinsics"])
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1

    m = {k: v / max(n, 1) for k, v in sums.items()}
    print(f"scored in {time.time() - t0:.0f}s\n")
    print(f"  APD (static baseline)  {m['average_pts_within_thresh']:.4f}")
    for t in (1, 2, 4, 8, 16):
        print(f"  pts_within_{t:<12} {m[f'pts_within_{t}']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"cache": args.cache, "clips": len(val_clips),
         "val_frac": args.val_frac, "seed": args.seed, **m}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
