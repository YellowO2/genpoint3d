"""Does optical flow alone follow points that turn? No model, nothing trained.

Each query point is carried from frame 0 by the cached flow
(`scripts/cache_flow.py`): its pixel in frame t+1 is its pixel in frame t
plus the forward flow read there, and its 3D position is that pixel at the
depth the frames cache has for it. Nothing ever looks back at frame 0, so this
is the frame-to-frame half on its own, with all its drift.

Scored as `scripts/evaluate.py --per-frame` scores a model: error against
distance moved, split by how far the point turned since frame 0, and moving
points' APD by frame. On the same clips and points as a fit_check of a run
(`--cache` its val directory, `--points` its points), the tables line up
with that run's.

  python scripts/flow_chain.py --cache <fit_check val dir> --frames <frames cache> \\
      --flow <flow cache> [--points 128] [--truth]

`--truth` carries the points by their true pixels instead of the flow: what is
left is the error of reading the depth at a cell, the floor this test has.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.tools.batch import to_metres
from genpoint3d.train.tools.camera import image_projection, project, unproject
from genpoint3d.train.tools.metrics import (appearance_change, change_table, clip_mean, error_and_motion,
                                            motion_px, motion_table, tapvid3d_metrics)

FLOW = ".flow.npy"


def at_cells(grid: torch.Tensor, uv: torch.Tensor, mode: str) -> torch.Tensor:
    """grid (B, C, h, w) on the CNN's cells, uv (B, N, 2) in [0, 1] -> (B, N, C).
    Cell j is centred at `(j + 1/8) / w` across (`scripts/cache_frames.py`)."""
    h, w = grid.shape[-2:]
    cell = uv * uv.new_tensor([w, h]) - 0.125
    g = 2 * cell / cell.new_tensor([w - 1, h - 1]) - 1
    return F.grid_sample(grid.float(), g[:, :, None], mode=mode, align_corners=True)[..., 0].transpose(1, 2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--flow", required=True)
    p.add_argument("--points", type=int, default=128)
    p.add_argument("--truth", action="store_true")
    args = p.parse_args()

    clips = sorted(Path(args.cache).glob("*.pt"))
    ds = ClipDataset(clips, args.points, resample=False, norm_mode="median", target="displacement",
                     fix_idcard=True, frames=Path(args.frames), features=False, depth=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    errs, disps, turns, vals = [], [], [], None
    k = 0
    for b in loader:
        B = b["traj"].shape[0]
        names = [c.stem for c in clips[k:k + B]]
        k += B
        flow = torch.from_numpy(np.stack([np.load(Path(args.flow) / f"{n}{FLOW}")[0] for n in names]))
        gt = to_metres(b["traj"].float(), b)
        proj = image_projection(b["intrinsics"], b["extrinsics"], b["hw"], b["norm_scale"], b["norm_mean"])
        scene = (gt - b["norm_mean"][:, None, None]) / b["norm_scale"][:, None, None, None]
        uv_true, _ = project(scene, proj)                                      # (B, T, N, 2)
        T = gt.shape[1]
        uv = [uv_true[:, 0]]
        for t in range(T - 1):
            nxt = uv_true[:, t + 1] if args.truth else uv[-1] + at_cells(flow[:, t], uv[-1], "bilinear")
            uv.append(nxt)
        uv = torch.stack(uv, dim=1)
        depth = torch.stack([at_cells(b["depth"][:, t, None], uv[:, t], "nearest")[..., 0]
                             for t in range(T)], dim=1)                        # (B, T, N)
        pr = unproject(uv, depth, proj.float())
        pr_m = pr * b["norm_scale"][:, None, None, None] + b["norm_mean"][:, None, None]
        vis = b["visibility"].bool()
        e, d = error_and_motion(pr_m, gt, vis, b["intrinsics_256"], b["extrinsics"])
        errs.append(e); disps.append(d)
        a, _ = appearance_change(gt, b["extrinsics"])
        keep = vis.clone()
        keep[:, 0] = False
        turns.append(a[keep])
        moving = motion_px(gt, vis, b["intrinsics_256"], b["extrinsics"]) > 4
        seen = vis & moving[:, None]
        if vals is None:
            vals = [[] for _ in range(T)]
        for t in range(T):
            m = tapvid3d_metrics(pr_m[:, t:t + 1], gt[:, t:t + 1], seen[:, t:t + 1],
                                 b["intrinsics_256"][:, t:t + 1], b["extrinsics"][:, t:t + 1],
                                 scaling="none", per_clip=True)
            vals[t].append(m["average_pts_within_thresh"])

    err, disp, turn = torch.cat(errs), torch.cat(disps), torch.cat(turns)
    print(f"{len(clips)} clips | {'true pixels (the depth floor)' if args.truth else 'chained SEA-RAFT flow'}")
    print("\n    every seen point after frame 0: error against how far it has moved")
    print(motion_table(err, disp))
    known = ~turn.isnan()
    print("\n    error / moved by how far the point turned since frame 0, degrees")
    print(change_table(err[known], disp[known], turn[known], (0, 5, 15, 45, 180), "turned"))
    print("\n    moving points' APD by frame")
    for t, v in enumerate(vals):
        print(f"    frame {t:2d}  {clip_mean(v):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
