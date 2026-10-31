"""Score the step tracker chained over whole clips, with and without the rigid
step of each point's piece (`StepTracker.track`).

Median error in the metric's pixels by how far the point has moved, as
`scripts/ball_test.pbs` and `scripts/ball_flow_fit.py` print it; on clips
with cameras, also split by how far the point turned.

  python scripts/eval_step.py --ckpt outputs/step_a/step.pt --cache DIR --frames-cache FRAMES
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.model.StepTracker import StepTracker
from genpoint3d.train.tools.batch import _cell_xyz, to_device, to_metres
from genpoint3d.train.tools.camera import image_projection, project
from genpoint3d.train.tools.metrics import appearance_change, change_table, error_and_motion, motion_table

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--frames-cache", required=True)
p.add_argument("--points", type=int, default=1000)
args = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model = StepTracker(ck["feat_dim"]).to(dev).eval()
model.load_state_dict(ck["model"])
clips = sorted(Path(args.cache).glob("*.pt"))
ds = ClipDataset(clips, args.points, resample=False, norm_mode="median", target="displacement",
                 fix_idcard=True, frames=Path(args.frames_cache), features=True, depth=True)
res = {k: ([], [], []) for k in ("rigid", "own step")}
with torch.no_grad():
    for batch in DataLoader(ds, batch_size=1, num_workers=2):
        b = to_device(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            fmap = model.finder.encode(b["frames"])
        fmap = fmap.float()
        proj = image_projection(b["intrinsics"], b["extrinsics"], b["hw"], b["norm_scale"], b["norm_mean"]).float()
        xyz = _cell_xyz(b["depth"].float(), proj)
        uv0, _ = project(b["anchor"].float()[:, None], proj[:, :1])
        emb = model.groups(b["id_card"], fmap[:, 0], uv0[:, 0], b["anchor"])
        gt = to_metres(b["traj"].float(), b)
        vis = b["visibility"].bool()
        turned, _ = appearance_change(gt, b["extrinsics"])
        keep = vis.clone()
        keep[:, 0] = False
        for name, rigid in (("rigid", True), ("own step", False)):
            pos = model.track(fmap, xyz, proj, b["anchor"], emb, rigid=rigid)
            pr = pos * b["norm_scale"][:, None, None, None] + b["norm_mean"][:, None, None]
            e, d = error_and_motion(pr, gt, vis, b["intrinsics_256"], b["extrinsics"])
            res[name][0].append(e); res[name][1].append(d); res[name][2].append(turned[keep].cpu())
print(f"{len(clips)} clips | {args.ckpt}")
for name, (e, d, a) in res.items():
    e, d, a = torch.cat(e), torch.cat(d), torch.cat(a)
    print(f"\n  {name}: error against how far the point has moved")
    print(motion_table(e, d))
    known = ~a.isnan()
    if known.sum() > 100:
        print(f"  {name}: error / moved by how far the point turned since frame 0, degrees")
        print(change_table(e[known], d[known], a[known], (0, 5, 15, 45, 180), "turned"))
