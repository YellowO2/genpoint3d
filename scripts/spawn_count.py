"""Where test-time spawning adds points on a clip: per frame, how many start
there and how many of them land on the ball (near its surface, its centre
from the raw render), against the ball's own points seen there.

Run:  python scripts/spawn_count.py --ckpt CKPT --cache DIR --frames-cache FRAMES --raw RAW
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import genpoint3d.train.score.Predictor as predictor_module
from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.Run import Run
from genpoint3d.train.score.spawn_points import spawn_points
from genpoint3d.train.tools.batch import to_metres
from genpoint3d.train.tools.metrics import _to_camera_t

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--frames-cache", required=True)
p.add_argument("--raw", required=True)
p.add_argument("--spawn", type=int, default=1000, help="points each clip is filled up to")
args = p.parse_args()

dev = torch.device("cuda")
run = Run.load(args.ckpt, dev)
ds = run.dataset(args.cache, args.frames_cache, points=1000)
added = {}


def kept(*a, **k):
    """`spawn_points`, its clip kept to count."""
    added["b"] = spawn_points(*a, **k)
    return added["b"]


predictor_module.spawn_points = kept
predictor = Predictor(run.model, run.judging(dev, spawn=args.spawn))
with torch.no_grad():
    for clip, batch in zip(ds.clips, DataLoader(ds, batch_size=1)):
        raw = np.load(Path(args.raw) / f"{clip.stem.split('_q')[0]}.npz")
        n = int(raw["ball_points"])
        centre = torch.from_numpy(raw["centres"]).to(dev).float()            # (T, 3) camera, metres
        predictor(batch)
        b = added["b"]
        N0 = batch["traj"].shape[2]
        q = b["query_t"][0, N0:]
        at = to_metres(torch.zeros_like(b["traj"][:, :1, N0:]), {**b, "norm_offset": b["norm_offset"][:, :, N0:]})[0, 0]
        cam = _to_camera_t(at[None, None].expand(1, len(centre), -1, -1).float(), b["extrinsics"].float())[0]   # (T, M, 3)
        # On the ball: within a fifth more than its radius, measured on its own points at frame 0.
        r = (to_metres(b["traj"][:, :1, :n], {**b, "norm_offset": b["norm_offset"][:, :, :n]})[0, 0] - centre[0]).norm(dim=-1).median()
        on = (cam[q, torch.arange(len(q), device=dev)] - centre[q]).norm(dim=-1) < 1.2 * r
        seen = b["visibility"][0, :, :n].sum(1).tolist()
        print(f"{clip.stem}: {len(q)} added")
        print("  frame:        " + " ".join(f"{t:>3}" for t in range(len(seen))))
        print("  ball seen:    " + " ".join(f"{s:>3}" for s in seen))
        print("  added:        " + " ".join(f"{int((q == t).sum()):>3}" for t in range(len(seen))))
        print("  added on ball:" + " ".join(f"{int(((q == t) & on).sum()):>3}" for t in range(len(seen))))
