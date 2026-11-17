"""What the adding rule gives on real clips, with no model: the points a clip
gets, the points on each object, and the seen surface left with no point within
its spacing, at a few gaps and limits. The points are picked as training picks
them (`ClipDataset._spawned`).

  python scripts/count_points.py --cache CACHE --frames FRAMES --gaps 0.025 0.035 0.05 --limits 1000 4000
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from genpoint3d.CachedClip import CachedClip
from genpoint3d.tools.traj_scale import REACH
from genpoint3d.train.data.ClipDataset import DEPTH, _near, _query_frames, _spawned
from genpoint3d.train.tools.farthest_first import spacing

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--frames", required=True, help="the cache's frames and depth (scripts/cache_frames.py)")
p.add_argument("--gaps", type=float, nargs="+", default=[0.025, 0.035, 0.05])
p.add_argument("--limits", type=int, nargs="+", default=[1000, 4000], help="the most points a clip holds")
p.add_argument("--asked", type=int, default=256)
p.add_argument("--clips", type=int, default=60)
args = p.parse_args()

torch.manual_seed(0)
files = sorted(Path(args.cache).expanduser().glob("*.pt"))
files = files[:: max(len(files) // args.clips, 1)][: args.clips]
print(f"{len(files)} clips of {args.cache}, {args.asked} asked")
print("gap    limit  points a clip  at the limit  first frame  an object's points (median, under 4)  surface left empty (median, worst)")
for gap in args.gaps:
    for limit in args.limits:
        count, first, full, per_object, empty = [], [], [], [], []
        for f in files:
            clip = _near(CachedClip.from_dict(torch.load(f, weights_only=False)))
            T, N = clip.visibility.shape
            got, _ = _spawned(clip, limit, args.asked, gap, torch.ones(T, dtype=torch.bool), torch.randperm(N))
            q, vis = _query_frames(got), got.visibility.bool()
            count.append(len(q)); first.append(int((q == 0).sum())); full.append(len(q) >= limit)
            # Each object the clip has points on (the whole clip's, not only the picked ones).
            per_object += [int((got.segment == k).sum()) for k in clip.segment.unique().tolist() if k > 0]
            P = torch.cat([got.traj.float(), torch.ones(T, len(q), 1)], -1)
            pix = torch.einsum("tij,tnj->tni", got.intrinsics.float(), torch.einsum("tij,tnj->tni", got.extrinsics.float()[:, :3, :], P))
            uv = pix[..., :2] / pix[..., 2:].clamp_min(1e-6)
            depth = torch.from_numpy(np.load(Path(args.frames).expanduser() / (got.seq_id + DEPTH))).float()
            h, w = depth.shape[-2:]
            H, W = got.hw
            v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            cell = torch.stack([(u.flatten() + 0.125) / w * W, (v.flatten() + 0.125) / h * H], -1).float()
            bad = tot = 0
            for t in range(T):
                z = depth[t].flatten()
                m = z < REACH
                on = vis[t] & (q <= t)
                far = spacing(z[m], float(got.intrinsics[t, 0, 0]), W, gap)
                d = torch.cdist(cell[m], uv[t, on]).min(1).values if on.any() else torch.full_like(far, 1e9)
                bad += int((d > far).sum()); tot += int(m.sum())
            empty.append(bad / max(tot, 1))
        po = np.array(per_object)
        print(f"{gap:<6} {limit:<6} {int(np.median(count)):<14} {np.mean(full):<13.0%} {int(np.median(first)):<12} "
              f"{int(np.median(po))}, {(po < 4).mean():.0%}{'':<24} {np.median(empty):.1%}, {max(empty):.1%}", flush=True)
