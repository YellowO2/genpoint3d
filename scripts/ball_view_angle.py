"""On the spinning-ball clips: is a point tracked well for as long as it is
seen, and lost only near the edge and after it turns away?

Each point of the ball and frame after 0 is put in one class:

    facing     seen, its surface turned less than 45 deg from the camera
    oblique    seen, 45-70 deg
    edge       seen, 70-90 deg: about to turn away, little left to match
    returned   seen again after being hidden at some earlier frame
    hidden     turned away (or covered)

the angle being between the surface's outward direction at the point (from
the ball's centre, `scripts/render_ball.py`) and the line back to the camera.
For each class, the median error in the metric's pixels (metres over depth /
focal, as `metrics.error_and_motion`) and the share within 4 px.

  python scripts/ball_view_angle.py --ckpt CKPT --cache DIR --frames-cache FRAMES --raw RAW
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import to_metres

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--frames-cache", required=True)
p.add_argument("--raw", required=True, help="the rendered clips, with the ball's centre")
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
targs = ckpt["args"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clips = sorted(Path(args.cache).glob("*.pt"))
ds = ClipDataset(clips, 1000, resample=False, norm_mode=targs.get("norm_mode", "median"),
                 target=targs.get("target", "displacement"), fix_idcard=bool(targs.get("fix_idcard", 0)),
                 frames=Path(args.frames_cache), features=True, depth=True)
first = torch.load(clips[0], weights_only=False, mmap=True)
model = build_model(Settings(**targs), first["feat_dim"], first["context"] is not None).to(dev)
model.load_state_dict(ckpt["model"])
predictor = Predictor(model, ScoreSettings(
    device=dev, steps=50, amp=dev.type == "cuda", anchor_frame0=bool(targs.get("anchor_frame0", 0)),
    loss_type=targs.get("loss_type", "l21"), method=targs.get("method", "regress"),
    refine_iters=targs.get("refine_iters", 4), param=targs.get("param", "position"),
    carry=bool(targs.get("carry", 0)), window=targs.get("window", 0), overlap=targs.get("overlap", 0),
    reach=targs.get("reach", "hold")))
names = ("facing", "oblique", "edge", "returned", "hidden")
errs = {n: [] for n in names}
with torch.no_grad():
    for clip, batch in zip(clips, DataLoader(ds, batch_size=1, num_workers=2)):
        raw = np.load(Path(args.raw) / f"{clip.stem.split('_q')[0]}.npz")
        n = int(raw["ball_points"])
        centre = torch.from_numpy(raw["centres"]).to(dev).float()            # (T, 3) camera, metres
        pred = predictor(batch)
        b = pred.batch
        gt, pr = to_metres(b["traj"].float(), b)[0, :, :n], to_metres(pred.tracks.float(), b)[0, :, :n]
        vis = b["visibility"][0, :, :n].bool()
        K = b["intrinsics_256"][0].float()
        focal = (K[..., 0, 0] * K[..., 1, 1]).sqrt()                          # (T,) or ()
        focal = focal[:, None] if focal.dim() else focal
        err = (pr - gt).norm(dim=-1) / (gt[..., 2].abs() / focal).clamp_min(1e-12)   # (T, n) px
        normal = gt - centre[:, None]
        normal = normal / normal.norm(dim=-1, keepdim=True)
        back = -gt / gt.norm(dim=-1, keepdim=True)                            # towards the camera
        angle = torch.rad2deg(torch.acos((normal * back).sum(-1).clamp(-1, 1)))
        was_hidden = (~vis).int().cummax(dim=0).values.bool()                # hidden at some frame <= t
        was_hidden = torch.cat([torch.zeros_like(was_hidden[:1]), was_hidden[:-1]], 0)
        cls = {"hidden": ~vis, "returned": vis & was_hidden}
        fresh = vis & ~was_hidden
        cls["facing"] = fresh & (angle < 45)
        cls["oblique"] = fresh & (angle >= 45) & (angle < 70)
        cls["edge"] = fresh & (angle >= 70)
        for name, m in cls.items():
            m = m.clone()
            m[0] = False
            errs[name].append(err[m].cpu())
print(f"{Path(args.cache).name} | {args.ckpt}")
print(f"  {'class':>9} {'point-frames':>13} {'median px':>10} {'within 4 px':>12}")
for name in names:
    e = torch.cat(errs[name]) if errs[name] else torch.zeros(0)
    if len(e):
        print(f"  {name:>9} {len(e):>13} {e.median():>10.2f} {(e < 4).float().mean():>12.3f}")
