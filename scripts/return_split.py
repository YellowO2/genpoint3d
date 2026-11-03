"""Is a moving point tracked well for as long as it stays in view, and lost
only once it has been hidden?

Each moving point (more than 4 px from its start at some frame) and frame
after 0, every frame given its image, is put in one class:

    always seen   seen, and seen in every frame before
    returned      seen again after being hidden at some earlier frame
    hidden        hidden (occluded or out of the frame)

split by frame (1-7, 8-15, 16+). For each, the median error in the metric's
pixels (metres over depth / focal, as `metrics.error_and_motion`) and the
share within 4 px. The Kubric counterpart of `scripts/ball_view_angle.py`,
without the angle, which needs the surface's direction.

  python scripts/return_split.py --ckpt CKPT --cache DIR --frames-cache FRAMES [--points 256]
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import to_metres
from genpoint3d.train.tools.metrics import _to_camera_t, motion_px

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--frames-cache", required=True)
p.add_argument("--points", type=int, default=256)
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
targs = ckpt["args"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clips = sorted(Path(args.cache).glob("*.pt"))
ds = ClipDataset(clips, args.points, resample=False, norm_mode=targs.get("norm_mode", "median"),
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
    reach=targs.get("reach", "hold"), back_iters=targs.get("back_iters", 0)))
names = ("always seen", "returned", "hidden")
spans = ((1, 8), (8, 16), (16, 10_000))
errs = {(n, s): [] for n in names for s in spans}
with torch.no_grad():
    for batch in DataLoader(ds, batch_size=1, num_workers=2):
        pred = predictor(batch)
        b = pred.batch
        gt, pr = to_metres(b["traj"].float(), b), to_metres(pred.tracks.float(), b)
        vis = b["visibility"].bool()
        K = b["intrinsics_256"].float()
        focal = (K[..., 0, 0] * K[..., 1, 1]).sqrt()
        focal = focal[..., None] if focal.dim() == 2 else focal[:, None, None]
        cam = _to_camera_t(gt, b["extrinsics"])
        err = ((pr - gt).norm(dim=-1) / (cam[..., 2].abs() / focal).clamp_min(1e-12))[0]   # (T, N) px
        moving = (motion_px(gt, vis, b["intrinsics_256"], b["extrinsics"]) > 4)[0]          # (N,)
        v = vis[0]
        was_hidden = (~v).int().cummax(dim=0).values.bool()
        was_hidden = torch.cat([torch.zeros_like(was_hidden[:1]), was_hidden[:-1]], 0)
        cls = {"always seen": v & ~was_hidden, "returned": v & was_hidden, "hidden": ~v}
        T = v.shape[0]
        for name, m in cls.items():
            for lo, hi in spans:
                sel = m.clone()
                sel[:lo] = False
                sel[min(hi, T):] = False
                sel &= moving[None]
                errs[(name, (lo, hi))].append(err[sel].cpu())
print(f"{len(clips)} clips | {args.ckpt} | moving points, every frame given its image")
print(f"  {'class':>12} {'frames':>7} {'point-frames':>13} {'median px':>10} {'within 4 px':>12}")
for name in names:
    for lo, hi in spans:
        e = torch.cat(errs[(name, (lo, hi))])
        if len(e):
            span = f"{lo}-{hi - 1}" if hi < 10_000 else f"{lo}+"
            print(f"  {name:>12} {span:>7} {len(e):>13} {e.median():>10.2f} {(e < 4).float().mean():>12.3f}")
