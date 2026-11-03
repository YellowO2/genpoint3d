"""What a checkpoint predicts on a few clips, saved, or checked against a save:
the guard that a cleanup of the code changes nothing it predicts.

  python scripts/snapshot.py --ckpt CKPT --cache DIR --frames-cache FRAMES --clips 6 --cutoff 12 --out a.pt
  python scripts/snapshot.py ... --against a.pt     # prints the largest difference
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

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--frames-cache", required=True)
p.add_argument("--clips", type=int, default=6)
p.add_argument("--cutoff", type=int, default=0)
p.add_argument("--out", default=None)
p.add_argument("--against", default=None)
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
targs = ckpt["args"]
dev = torch.device("cuda")
clips = sorted(Path(args.cache).glob("*.pt"))[: args.clips]
ds = ClipDataset(clips, targs["points"], resample=False, norm_mode=targs.get("norm_mode", "median"),
                 target=targs.get("target", "displacement"), fix_idcard=bool(targs.get("fix_idcard", 0)),
                 frames=Path(args.frames_cache), features=True, depth=True,
                 level=bool(targs.get("level", 0)), query_before=args.cutoff)
first = torch.load(clips[0], weights_only=False, mmap=True)
model = build_model(Settings(**targs), first["feat_dim"], first["context"] is not None).to(dev)
model.load_state_dict(ckpt["model"])
predictor = Predictor(model, ScoreSettings(
    device=dev, amp=True, anchor_frame0=bool(targs.get("anchor_frame0", 0)),
    loss_type=targs.get("loss_type", "l21"), method="regress", refine_iters=targs.get("refine_iters", 4),
    param=targs.get("param", "velocity"), window=targs.get("window", 0), overlap=targs.get("overlap", 0),
    schedule=targs.get("schedule", ""), cutoff=args.cutoff))
out = []
torch.manual_seed(0)
with torch.no_grad():
    for batch in DataLoader(ds, batch_size=1):
        pred = predictor(batch)
        out.append({"tracks": pred.tracks.float().cpu(), "loss": pred.loss,
                    "visibility": None if pred.visibility is None else pred.visibility.float().cpu()})
if args.out:
    torch.save(out, args.out)
    print(f"saved {len(out)} clips to {args.out}")
if args.against:
    ref = torch.load(args.against, weights_only=False)
    worst = max((a["tracks"] - b["tracks"]).abs().max().item() for a, b in zip(out, ref))
    loss = max(abs(a["loss"] - b["loss"]) for a, b in zip(out, ref))
    print(f"{args.ckpt}: largest track difference {worst:.2e}, loss difference {loss:.2e}"
          f" over {len(out)} clips -> {'SAME' if worst < 1e-3 else 'CHANGED'}")
