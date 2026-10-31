"""How much of a run's error is where the point is drawn, and how much its depth.

For every seen point after frame 0: the 3D error in the metric's pixels (as
the scores count it); the 2D error, between the drawn pixels of the guess and
the truth in the metric's 256 px frame; and the depth error, the gap along
the camera axis in the same metric pixels. Medians, for points that moved
more than 16 px.

  python scripts/error_split.py --ckpt CKPT --cache DIR --frames-cache FRAMES [--points 1000]
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
from genpoint3d.train.tools.metrics import _project, _to_camera_t

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--frames-cache", required=True)
p.add_argument("--points", type=int, default=1000)
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
predictor = Predictor(model, ScoreSettings(
    device=dev, steps=50, amp=amp, anchor_frame0=bool(targs.get("anchor_frame0", 0)),
    loss_type=targs.get("loss_type", "l21"), method=targs.get("method", "regress"),
    refine_iters=targs.get("refine_iters", 4), param=targs.get("param", "position"),
    carry=bool(targs.get("carry", 0)), window=targs.get("window", 0), overlap=targs.get("overlap", 0)))
e3, e2, ez = [], [], []
with torch.no_grad():
    for batch in DataLoader(ds, batch_size=1, num_workers=2):
        pred = predictor(batch)
        b = pred.batch
        gt, pr = to_metres(b["traj"].float(), b), to_metres(pred.tracks.float(), b)
        gc, pc = _to_camera_t(gt, b["extrinsics"]), _to_camera_t(pr, b["extrinsics"])
        K = b["intrinsics_256"]
        focal = (K[..., 0, 0] * K[..., 1, 1]).sqrt()
        mult = (gc[..., 2].abs() / focal[..., None]).clamp_min(1e-6)
        moved = (gt - gt[:, :1]).norm(dim=-1) / mult
        keep = b["visibility"].bool() & (moved > 16)
        keep[:, 0] = False
        e3.append(((pr - gt).norm(dim=-1) / mult)[keep].cpu())
        e2.append((_project(pc, K) - _project(gc, K)).norm(dim=-1)[keep].cpu())
        ez.append(((pc[..., 2] - gc[..., 2]).abs() / mult)[keep].cpu())
e3, e2, ez = torch.cat(e3), torch.cat(e2), torch.cat(ez)
print(f"  {Path(args.cache).name:>14}  {len(e3):>6} point-frames moved > 16 px | median px:"
      f" 3D {e3.median():.2f}  2D {e2.median():.2f}  depth {ez.median():.2f}")
