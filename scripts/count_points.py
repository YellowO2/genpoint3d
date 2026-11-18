"""What the adding rule gives on real clips, with no model: the points a clip
gets, how many are on the background, the points on each object, and the seen
surface left with no point within its spacing and off its body's flat patch,
at a few gaps, flat-patch distances and limits. The points are picked as
training picks them (`ClipDataset._spawned`).

  python scripts/count_points.py --cache CACHE --gaps 0.025 --flats 0 0.005 0.01 0.02 --limits 4000
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from genpoint3d.CachedClip import CachedClip
from genpoint3d.train.data.ClipDataset import _near, _query_frames, _spawned
from genpoint3d.train.tools.farthest_first import on_patch, spacing

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--gaps", type=float, nargs="+", default=[0.025])
p.add_argument("--flats", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02],
               help="metres a body's points explain surface within; 0 for the spacing alone")
p.add_argument("--limits", type=int, nargs="+", default=[4000], help="the most points a clip holds")
p.add_argument("--asked", type=int, default=256)
p.add_argument("--clips", type=int, default=60)
args = p.parse_args()

torch.manual_seed(0)
files = sorted(Path(args.cache).expanduser().glob("*.pt"))
files = files[:: max(len(files) // args.clips, 1)][: args.clips]
print(f"{len(files)} clips of {args.cache}, {args.asked} asked")
print("gap    flat   limit  points a clip  at the limit  on the background  an object's points (median, under 4)  surface left out (median, worst)", flush=True)
for gap in args.gaps:
    for flat in args.flats:
        for limit in args.limits:
            count, full, back, per_object, left = [], [], [], [], []
            for f in files:
                clip = _near(CachedClip.from_dict(torch.load(f, weights_only=False)))
                T, N = clip.visibility.shape
                got, _ = _spawned(clip, limit, args.asked, gap, torch.ones(T, dtype=torch.bool), torch.randperm(N), flat)
                q = _query_frames(got)
                count.append(len(q)); full.append(len(q) >= limit); back.append(float((got.segment == 0).float().mean()))
                # Each object the clip has points on (the whole clip's, not only the picked ones).
                per_object += [int((got.segment == k).sum()) for k in clip.segment.unique().tolist() if k > 0]

                def pixels(c):
                    P = torch.cat([c.traj.float(), torch.ones(*c.traj.shape[:2], 1)], -1)
                    pix = torch.einsum("tij,tnj->tni", c.intrinsics.float(), torch.einsum("tij,tnj->tni", c.extrinsics.float()[:, :3, :], P))
                    return pix[..., :2] / pix[..., 2:].clamp_min(1e-6), pix[..., 2]

                (uv, z), (uv_got, _) = pixels(clip), pixels(got)
                bad = tot = 0
                for t in range(T):
                    m, on = clip.visibility[t].bool(), got.visibility[t].bool() & (q <= t)
                    if not m.any():
                        continue
                    far = spacing(z[t, m], float(clip.intrinsics[t, 0, 0]), clip.hw[1], gap)
                    out = (torch.cdist(uv[t, m], uv_got[t, on]).min(1).values > far) if on.any() else torch.ones_like(far, dtype=torch.bool)
                    if flat:
                        out &= ~on_patch(clip.traj[t, m].float(), clip.segment[m], got.traj[t, on].float(), got.segment[on], flat)
                    bad += int(out.sum()); tot += int(m.sum())
                left.append(bad / max(tot, 1))
            po = np.array(per_object)
            print(f"{gap:<6} {flat:<6} {limit:<6} {int(np.median(count)):<14} {np.mean(full):<13.0%} {np.median(back):<18.0%} "
                  f"{int(np.median(po))}, {(po < 4).mean():.0%}{'':<24} {np.median(left):.1%}, {max(left):.1%}", flush=True)
