"""On the spinning-ball clips: is comparing whole neighbouring frames enough to
follow a turning surface? No model, nothing trained.

SEA-RAFT compares frame t-1 with frame t over the whole frame. Every cell of
the ball visible in both frames gives a correspondence in 3D (its pixel at the
true depth, then where the flow sends it, at the true depth there). Each step,
the one rotation and shift that best carry the ball's cells from t-1 to t
(Kabsch, after dropping the worst-fitting quarter), composed onto the ball's
motion so far. Each tracked point of the ball is carried by that motion from
frame 0. Which cells are the ball's comes from its true centre: it stands in
for the grouping a model would learn.

Scored like `scripts/ball_test.pbs`: median error in the metric's pixels by
how far the point has moved, for the ball's points.

  python scripts/ball_flow_fit.py --clips DIR --repo references/searaft/repo \\
      --weights <SEA-RAFT snapshot>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from cache_flow import load_raft  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--clips", required=True)
p.add_argument("--repo", required=True)
p.add_argument("--weights", required=True)
args = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, margs = load_raft(Path(args.repo), Path(args.weights), "spring-M.json")
model = model.to(dev).eval()
R_BALL = 0.6


def kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = torch.linalg.svd((a - ca).T @ (b - cb))
    d = torch.sign(torch.det(Vt.T @ U.T))
    R = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, d], device=a.device)) @ U.T
    return R, cb - R @ ca


edges = (16, 32, 64, 128, float("inf"))
by_motion = {}
for clip in sorted(Path(args.clips).glob("*.npz")):
    d = np.load(clip)
    frames = torch.from_numpy(d["images"]).to(dev)
    depth = torch.from_numpy(np.load(clip.with_suffix(".depth.npy")).astype(np.float32)).to(dev)
    f, _, cx, cy = d["fx_fy_cx_cy"]
    centres = torch.from_numpy(d["centres"]).to(dev)
    T, H, W = depth.shape
    x = frames.permute(0, 3, 1, 2).float()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        flow = model(x[:-1], x[1:], iters=margs.iters, test_mode=True)["flow"][-1].float()   # (T-1, 2, H, W) px
    js = torch.arange(0, H, 4, device=dev)
    v, u = torch.meshgrid(js, js, indexing="ij")
    u, v = u.flatten(), v.flatten()

    def lift(t, uu, vv):
        z = depth[t, vv.long().clamp(0, H - 1), uu.long().clamp(0, W - 1)]
        return torch.stack([(uu + 0.5 - cx) / f * z, (vv + 0.5 - cy) / f * z, z], -1)

    R_tot, s_tot = torch.eye(3, device=dev), torch.zeros(3, device=dev)
    poses = [(R_tot, s_tot)]
    for t in range(1, T):
        a = lift(t - 1, u.float(), v.float())
        un = u.float() + flow[t - 1, 0, v, u]
        vn = v.float() + flow[t - 1, 1, v, u]
        b = lift(t, un.round(), vn.round())
        on = ((a - centres[t - 1]).norm(dim=-1) < R_BALL + 0.02) & ((b - centres[t]).norm(dim=-1) < R_BALL + 0.02)
        on &= (un >= 0) & (un < W) & (vn >= 0) & (vn < H)
        if on.sum() >= 3:
            Rs, ss = kabsch(a[on], b[on])
            res = (a[on] @ Rs.T + ss - b[on]).norm(dim=-1)
            keep = res <= res.quantile(0.75)
            Rs, ss = kabsch(a[on][keep], b[on][keep])
        else:
            Rs, ss = poses[-1][0] @ poses[-2][0].T if len(poses) > 1 else torch.eye(3, device=dev), torch.zeros(3, device=dev)
        R_tot, s_tot = Rs @ R_tot, Rs @ s_tot + ss
        poses.append((R_tot, s_tot))
    n = int(d["ball_points"])
    gt = torch.from_numpy(d["tracks_XYZ"][:, :n]).to(dev)
    vis = torch.from_numpy(d["visibility"][:, :n]).to(dev)
    pred = torch.stack([gt[0] @ R.T + s for R, s in poses])
    mult = gt[..., 2].abs() / (f * 256 / W)                    # metres per pixel of the metric
    err = (pred - gt).norm(dim=-1) / mult
    moved = (gt - gt[:1]).norm(dim=-1) / mult
    keep = vis.clone()
    keep[0] = False
    motion = clip.stem.rsplit("_", 1)[0]
    by_motion.setdefault(motion, [[], []])
    by_motion[motion][0].append(err[keep].cpu())
    by_motion[motion][1].append(moved[keep].cpu())

print("ball points: median error in metric px by distance moved, whole-frame flow + rigid fit")
print(f"  {'motion':>14} " + " ".join(f"{f'{lo:g}-{hi:g} px':>11}" for lo, hi in zip(edges[:-1], edges[1:])))
for m, (e, mv) in by_motion.items():
    e, mv = torch.cat(e), torch.cat(mv)
    cells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (mv >= lo) & (mv < hi)
        cells.append(f"{e[k].median():>11.2f}" if k.sum() >= 20 else f"{'-':>11}")
    print(f"  {m:>14} " + " ".join(cells))
