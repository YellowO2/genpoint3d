"""What a checkpoint predicts on a few clips, saved or checked against a save,
and one training step's loss and gradients checked against a save: the guard
that a cleanup of the code changes nothing.

  python scripts/snapshot.py --ckpt CKPT --cache DIR --frames-cache FRAMES --clips 6 --cutoff 12 --out a.pt
  python scripts/snapshot.py ... --against a.pt        # the largest track difference
  python scripts/snapshot.py ... --train-against t.pt  # one step on the first clip
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.learn import learn
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
p.add_argument("--train-against", default=None)
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
targs = ckpt["args"]
settings = Settings(**{"schedule": "", **targs})
dev = torch.device("cuda")
clips = sorted(Path(args.cache).glob("*.pt"))[: args.clips]
level = bool(targs.get("level", 0))
first = torch.load(clips[0], weights_only=False, mmap=True)
model = build_model(settings, first["feat_dim"]).to(dev)
model.load_state_dict(ckpt["model"])

if args.train_against:
    ds = ClipDataset(clips[:1], targs["points"], resample=False, frames=Path(args.frames_cache), level=level)
    model.train()
    torch.manual_seed(0)
    losses = learn(model, next(iter(DataLoader(ds, batch_size=1))), settings, Hardware(dev, True))
    ref = torch.load(args.train_against, weights_only=False)
    grads = {n: q.grad.float().cpu() for n, q in model.named_parameters() if q.grad is not None}
    # Each parameter's largest change, over its largest gradient.
    rel = max(((grads[n] - g).abs().max() / g.abs().max().clamp_min(1e-12)).item() for n, g in ref["grads"].items())
    print(f"{args.ckpt}: training loss {losses.method:.6f} vs {ref['method']:.6f}, visibility "
          f"{losses.visibility:.6f} vs {ref['visibility']:.6f}; {len(grads)} vs {len(ref['grads'])} "
          f"params with gradients, largest relative gradient change {rel:.2e}"
          f" -> {'SAME' if rel < 1e-3 and set(grads) == set(ref['grads']) else 'CHANGED'}")
    raise SystemExit

ds = ClipDataset(clips, targs["points"], resample=False, frames=Path(args.frames_cache), level=level,
                 query_before=args.cutoff)
predictor = Predictor(model, ScoreSettings(
    device=dev, amp=True, refine_iters=targs.get("refine_iters", 4), window=targs.get("window", 0),
    overlap=targs.get("overlap", 0), schedule=targs.get("schedule", ""), cutoff=args.cutoff))
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
    vis = max(((a["visibility"] - b["visibility"]).abs().max().item() if a["visibility"] is not None else 0.0)
              for a, b in zip(out, ref))
    print(f"{args.ckpt}: largest track difference {worst:.2e}, visibility {vis:.2e}, loss {loss:.2e}"
          f" over {len(out)} clips -> {'SAME' if worst < 1e-3 else 'CHANGED'}")
