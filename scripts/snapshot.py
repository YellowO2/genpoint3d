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
from genpoint3d.train.learn import learn
from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.Run import Run

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

dev = torch.device("cuda")
run = Run.load(args.ckpt, dev)
model, settings = run.model, run.settings
clips = sorted(Path(args.cache).glob("*.pt"))[: args.clips]

if args.train_against:
    # The batch before the seed: a loader's iterator draws from the generator too.
    batch = next(iter(DataLoader(run.dataset(clips[:1], args.frames_cache), batch_size=1)))
    model.train()
    steps = []
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        losses = learn(model, batch, settings, Hardware(dev, True))
        steps.append((losses, {n: q.grad.float().cpu() for n, q in model.named_parameters() if q.grad is not None}))
    ref = torch.load(args.train_against, weights_only=False)
    (losses, grads), (_, again) = steps
    # The largest gradient change and the whole gradient's relative change,
    # against the reference and against the same code run twice (the GPU's
    # backward is not deterministic).
    gap = lambda a, b: (max((a[n] - b[n]).abs().max().item() for n in b),
                        sum((a[n] - b[n]).pow(2).sum() for n in b).sqrt().item()
                        / sum(b[n].pow(2).sum() for n in b).sqrt().item())
    (d_abs, d_rel), (n_abs, n_rel) = gap(grads, ref["grads"]), gap(grads, again)
    same = (losses.method == ref["method"] and set(grads) == set(ref["grads"])
            and d_abs <= 3 * max(n_abs, 1e-12) and d_rel <= 3 * max(n_rel, 1e-12))
    print(f"{args.ckpt}: training loss {losses.method:.6f} vs {ref['method']:.6f}, visibility "
          f"{losses.visibility:.6f} vs {ref['visibility']:.6f}; gradients against the reference: largest "
          f"change {d_abs:.2e}, relative {d_rel:.2e}; run twice: {n_abs:.2e}, {n_rel:.2e}"
          f" -> {'SAME' if same else 'CHANGED'}")
    raise SystemExit

ds = run.dataset(clips, args.frames_cache, query_before=args.cutoff)
predictor = Predictor(model, run.judging(dev, cutoff=args.cutoff))
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
