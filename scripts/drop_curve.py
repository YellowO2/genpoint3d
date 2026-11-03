"""Does a run forecast free fall at the right speed? On the Kubric drop test
(`scripts/kubric_movi_tracks.py --drop_test`, cached as usual), each frame's
height of the dropped object's centre above where it started, true and
forecast, the forecast given frames before --cutoff only. Up is the clip's own
(`CachedClip.up`), so the heights are along gravity.

  python scripts/drop_curve.py --ckpt CKPT --cache DIR --frames-cache FRAMES [--cutoff 2]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from genpoint3d.CachedClip import CachedClip
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
p.add_argument("--cutoff", type=int, default=2)
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
targs = ckpt["args"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clips = sorted(Path(args.cache).glob("*.pt"))
level = bool(targs.get("level", 0))
ds = ClipDataset(clips, 3000, resample=False, norm_mode=targs.get("norm_mode", "median"),
                 target=targs.get("target", "displacement"), fix_idcard=bool(targs.get("fix_idcard", 0)),
                 frames=Path(args.frames_cache), features=True, depth=True, level=level)
first = torch.load(clips[0], weights_only=False, mmap=True)
model = build_model(Settings(**targs), first["feat_dim"], first["context"] is not None).to(dev)
model.load_state_dict(ckpt["model"])
predictor = Predictor(model, ScoreSettings(
    device=dev, amp=dev.type == "cuda", anchor_frame0=bool(targs.get("anchor_frame0", 0)),
    loss_type=targs.get("loss_type", "l21"), method=targs.get("method", "regress"),
    refine_iters=targs.get("refine_iters", 4), param=targs.get("param", "position"),
    window=targs.get("window", 0), overlap=targs.get("overlap", 0),
    schedule=targs.get("schedule", ""), cutoff=args.cutoff))
up_clip = CachedClip.from_dict(first).up.float()
up = torch.tensor([0.0, -1.0, 0.0]) if level else up_clip / up_clip.norm()
with torch.no_grad():
    batch = next(iter(DataLoader(ds, batch_size=1)))
    pred = predictor(batch)
    b = pred.batch
    gt, pr = to_metres(b["traj"].float(), b)[0].cpu(), to_metres(pred.tracks.float(), b)[0].cpu()
# The dropped object's points, handed to the model at frame 0 and seen there.
moving = ((gt - gt[:1]).norm(dim=-1).amax(0) > 0.1) & (b["query_t"][0].cpu() == 0) & b["visibility"][0, 0].cpu().bool()
h_gt = ((gt[:, moving] - gt[:1, moving]) @ up).mean(1).numpy()
h_pr = ((pr[:, moving] - gt[:1, moving]) @ up).mean(1).numpy()
fall = 0.5 * 9.81 * (np.arange(len(h_gt)) / 12.0) ** 2
print(f"{args.ckpt} | {int(moving.sum())} points on the object | images up to frame {args.cutoff - 1}")
print(f"  {'frame':>5} {'true height m':>14} {'forecast':>9} {'free fall':>10}")
for t in range(len(h_gt)):
    mark = "" if t >= args.cutoff else "  (seen)"
    print(f"  {t:>5} {h_gt[t]:>14.3f} {h_pr[t]:>9.3f} {-fall[t]:>10.3f}{mark}")
