"""On the spinning-ball clips: when a point turns to the ball's hidden back,
does the run send it round there, or keep it on the side it can see?

For every point of the ball and frame where the truth is hidden (turned away),
whether the prediction is behind the ball's centre (on the back, as it truly
is) or in front of it (still on the visible face), along the camera's line of
sight; and the same for frames where the point is seen. The ball's centre
comes from the rendered clip (`scripts/render_ball.py`).

  python scripts/ball_backside.py --ckpt CKPT --cache DIR --frames-cache FRAMES --raw RAW
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
    carry=bool(targs.get("carry", 0)), window=targs.get("window", 0), overlap=targs.get("overlap", 0)))
counts = {"hidden": [0, 0], "seen": [0, 0]}
with torch.no_grad():
    for clip, batch in zip(clips, DataLoader(ds, batch_size=1, num_workers=2)):
        raw = np.load(Path(args.raw) / f"{clip.stem.split('_q')[0]}.npz")
        n = int(raw["ball_points"])
        centre = torch.from_numpy(raw["centres"]).to(dev)                    # (T, 3) camera, metres
        pred = predictor(batch)
        b = pred.batch
        gt, pr = to_metres(b["traj"].float(), b)[0, :, :n], to_metres(pred.tracks.float(), b)[0, :, :n]
        vis = b["visibility"][0, :, :n].bool()
        sight = centre / centre.norm(dim=-1, keepdim=True)                   # (T, 3)
        behind = ((pr - centre[:, None]) * sight[:, None]).sum(-1) > 0     # farther than the centre
        for name, m in (("hidden", ~vis), ("seen", vis)):
            m = m.clone()
            m[0] = False
            counts[name][0] += int((behind & m).sum())
            counts[name][1] += int(m.sum())
print(f"{Path(args.cache).name}: points of the ball, the prediction behind the ball's centre")
for name, (k, total) in counts.items():
    print(f"  truly {name:>6}: {k / max(total, 1):.1%} of {total} point-frames")
