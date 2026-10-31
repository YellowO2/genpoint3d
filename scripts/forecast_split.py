"""Where forecasting fails: points at rest, in smooth motion, or hitting something.

The run forecasts frames --cutoff on with no image, as its validation does.
Each point is put in one of three kinds by its true track over the frames
from the last one with an image to the end:

  at rest   moves less than 1 cm in all of them
  smooth    its acceleration never jumps (free flight, steady rolling or
            sliding): every frame-to-frame change of acceleration under --jerk
  impact    it does jump somewhere in them: a bounce, a hit, a stop

For each kind, how many points, and how far off the forecast is, in the
metric's pixels at the point's depth, over the forecast frames: the median
error and the share within 1 / 4 / 16 px. Against the same split for holding
the last seen position still and for keeping its last speed.

  python scripts/forecast_split.py --ckpt outputs/fyT_dense_reference/best.pt \\
      --cache <val dir> --frames-cache <frames> [--points 500] [--cutoff 12]
"""

import argparse
import dataclasses
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import to_metres
from genpoint3d.train.tools.metrics import _to_camera_t

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--frames-cache", required=True)
p.add_argument("--points", type=int, default=500)
p.add_argument("--cutoff", type=int, default=12)
p.add_argument("--jerk", type=float, default=0.02, help="metres per frame^3")
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
targs = ckpt["args"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp = dev.type == "cuda" and torch.cuda.is_bf16_supported()
clips = sorted(Path(args.cache).glob("*.pt"))
ds = ClipDataset(clips, args.points, resample=False, norm_mode=targs.get("norm_mode", "median"),
                 target=targs.get("target", "displacement"), fix_idcard=bool(targs.get("fix_idcard", 0)),
                 frames=Path(args.frames_cache), features=bool(targs.get("dino", 1)),
                 depth=bool(targs.get("fine_xyz", 0)))
first = torch.load(clips[0], weights_only=False, mmap=True)
model = build_model(Settings(**targs), first["feat_dim"], first["context"] is not None).to(dev)
model.load_state_dict(ckpt["model"])
judging = ScoreSettings(device=dev, steps=50, amp=amp, anchor_frame0=bool(targs.get("anchor_frame0", 0)),
                        loss_type=targs.get("loss_type", "l21"), method=targs.get("method", "regress"),
                        refine_iters=targs.get("refine_iters", 4), param=targs.get("param", "position"),
                        carry=bool(targs.get("carry", 0)), window=targs.get("window", 0),
                        overlap=targs.get("overlap", 0), cutoff=args.cutoff)
predictor = Predictor(model, judging)

C = args.cutoff
kinds = ("at rest", "smooth", "impact")
errs = {(k, w): [] for k in kinds for w in ("model", "stop", "last speed")}
with torch.no_grad():
    for batch in DataLoader(ds, batch_size=1, num_workers=2):
        pred = predictor(batch)
        b = pred.batch
        gt = to_metres(b["traj"].float(), b)                                  # (1, T, N, 3)
        pr = to_metres(pred.tracks.float(), b)
        stop = gt[:, C - 1:C].expand_as(gt)
        speed = gt[:, C - 1:C] - gt[:, C - 2:C - 1]
        steps = torch.arange(gt.shape[1], device=dev)[None, :, None, None] - (C - 1)
        last_speed = gt[:, C - 1:C] + speed * steps
        win = gt[:, C - 1:]
        moved = (win - win[:, :1]).norm(dim=-1).amax(1)                       # (1, N)
        jerk = win.diff(dim=1).diff(dim=1).diff(dim=1).norm(dim=-1).amax(1)
        kind = torch.where(moved < 0.01, 0, torch.where(jerk < args.jerk, 1, 2))
        cam = _to_camera_t(gt, b["extrinsics"])
        focal = (b["intrinsics_256"][..., 0, 0] * b["intrinsics_256"][..., 1, 1]).sqrt()
        mult = (cam[..., 2].abs() / focal[..., None]).clamp_min(1e-6)        # metres per px
        for name, guess in (("model", pr), ("stop", stop), ("last speed", last_speed)):
            e = ((guess - gt).norm(dim=-1) / mult)[:, C:]                     # (1, T - C, N)
            for i, k in enumerate(kinds):
                m = (kind == i)[:, None].expand_as(e)
                if m.any():
                    errs[(k, name)].append(e[m].cpu())

print(f"{len(clips)} clips | {args.ckpt} | frames {C} on forecast | impact: acceleration jumps "
      f"more than {args.jerk} m/frame^3")
print(f"  {'kind':>8} {'guess':>11} {'point-frames':>13} {'median px':>10} {'<1px':>6} {'<4px':>6} {'<16px':>6}")
for k in kinds:
    for name in ("model", "stop", "last speed"):
        if errs[(k, name)]:
            e = torch.cat(errs[(k, name)])
            print(f"  {k:>8} {name:>11} {len(e):>13} {e.median():>10.2f} {(e < 1).float().mean():>6.3f}"
                  f" {(e < 4).float().mean():>6.3f} {(e < 16).float().mean():>6.3f}")
