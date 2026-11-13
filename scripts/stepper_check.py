"""Whether stepping every point one frame at a time, as GNS does, forecasts
better than what we have: a stepper (`model/Stepper.py`) trained on the
training clips' true 3D tracks alone, no images, then rolled out on the own
validation clips from frame `--cutoff` on, from the true positions before it.
Scored as validation scores the forecast (moving points, no rescale), beside
the baselines from the same true past (held still, constant speed); plainly,
and with each true piece kept rigid every step (shape matching). Each point is
told which of its neighbours share its piece, by the truth here (the cache's
objects, or where it has none the points whose distances stay put; the model's
groups, 96% right, would tell it in the model). Our tracker's forecast, from
its own past: 0.353 (stop 0.334, constant speed 0.332).

Run:  python scripts/stepper_check.py --cache A B C --frames-cache FA FB FC --mix 0.45 0.35 0.2 --out outputs/stepper_check
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from genpoint3d.tools.traj_scale import GRAVITY_REF
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.model.Stepper import Stepper, shape_match
from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.score.track_scores import track_scores
from genpoint3d.train.tools.batch import components, rigid_groups
from genpoint3d.train.tools.metrics import clip_mean

KEEP = ("traj", "visibility", "query_t", "segment", "intrinsics_256", "extrinsics", "norm_mean", "norm_scale",
        "norm_traj_scale", "norm_offset", "gravity", "down", "down_known")


class _Lean(Dataset):
    """A ClipDataset's items without the images and features."""

    def __init__(self, ds: ClipDataset) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        b = self.ds[i]
        return {k: b[k] for k in KEEP}


def _pieces(b: dict, device) -> torch.Tensor:
    """(1, N) each point's piece where the cache does not say: the points
    whose distances to each other stay put over the clip (`rigid_groups`'
    motion rule), joined through each other."""
    return components(rigid_groups({k: v.to(device) for k, v in b.items()}, by_object=False)).cpu()


def _clips(ds: ClipDataset, workers: int, device) -> list[dict]:
    """Each clip: the batch (B = 1) less images and features, its positions
    `pos` (1, T, N, 3) in the clip's unit, its gravity `g` (1, 4) and its
    pieces in `segment`, by the motion rule where the cache does not say."""
    out = []
    for b in DataLoader(_Lean(ds), batch_size=1, num_workers=workers):
        # Copied out of the workers' shared memory: kept, thousands would hold a file each.
        b = {k: v.clone() for k, v in b.items()}
        if (b["segment"] < 0).all():
            b["segment"] = _pieces(b, device)
        b["pos"] = b["traj"].float() * b["norm_traj_scale"][:, None, None, None] + b["norm_offset"].float()
        known = b["down_known"].float()[:, None]
        b["g"] = torch.cat([b["gravity"].float()[:, None] * GRAVITY_REF * b["down"].float() * known, known], -1)
        out.append(b)
    return out


@torch.no_grad()
def _score(stepper: Stepper, val: list[dict], cutoff: int, device) -> dict:
    """Mean over clips of the forecast scores, plain and kept rigid."""
    j = ScoreSettings(device=device, cutoff=cutoff)
    rows: dict[str, list] = {}
    H = stepper.history
    for b in val:
        b = {k: v.to(device) for k, v in b.items()}
        pos, piece = b["pos"], b["segment"]
        for name, rigid in (("stepped", False), ("stepped, rigid", True)):
            x = pos.clone()
            for t in range(cutoff, x.shape[1]):
                nxt = stepper.step(x[:, t - H - 1:t], b["g"], piece)
                x[:, t] = shape_match(nxt, pos[:, cutoff - 1], piece) if rigid else nxt
            tracks = (x - b["norm_offset"]) / b["norm_traj_scale"][:, None, None, None]
            s = track_scores(Prediction(batch=b, visual_mask=None, tracks=tracks, loss=0.0), j)
            rows.setdefault(name, []).append(s["apd_moving_fut"].cpu())
            if not rigid:
                rows.setdefault("stop", []).append(s["apd_moving_stop_fut"].cpu())
                rows.setdefault("constant speed", []).append(s["apd_moving_cv_fut"].cpu())
    return {k: clip_mean(v) for k, v in rows.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", nargs="+", required=True)
    p.add_argument("--frames-cache", nargs="+", required=True)
    p.add_argument("--mix", nargs="+", type=float)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--points", type=int, default=1000, help="points per training clip")
    p.add_argument("--samples", type=int, default=16, help="steps drawn from each training clip")
    p.add_argument("--clips", type=int, default=4, help="training clips per optimiser step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--noise", type=float, default=0.003, help="random walk on the last input step, clip units")
    p.add_argument("--moving", type=float, default=1e-3, help="a point moves if a step of its exceeds this")
    p.add_argument("--cutoff", type=int, default=12)
    p.add_argument("--val-points", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    clips = split_clips(args.cache, 0.1, 0, args.mix)
    frames = {Path(c): Path(f) for c, f in zip(args.cache, args.frames_cache)}
    t0 = time.time()
    # As validation reads them: its first points, queried before the cutoff.
    val = _clips(ClipDataset(clips.val, args.val_points, resample=False, query_before=args.cutoff,
                             frames=frames), args.workers, device)
    train = _clips(ClipDataset(clips.train, 10 ** 6, resample=False, frames=frames), args.workers, device)
    weights = torch.tensor(clips.weights)
    print(f"{len(train)} training clips, {len(val)} validation clips, read in {time.time() - t0:.0f} s", flush=True)

    stepper = Stepper().to(device)
    H = stepper.history
    vel = torch.cat([c["pos"][0].diff(dim=0).flatten(0, 1) for c in train[:500]])
    acc = torch.cat([c["pos"][0].diff(dim=0).diff(dim=0).flatten(0, 1) for c in train[:500]])
    some = [c["pos"][0, 0] for c in train[:200]]
    reach = torch.stack([torch.cdist(x, x).topk(stepper.k, largest=False).values[:, -1].median() for x in some]).median()
    stepper.set_stats(vel, acc, reach.item(), args.noise)
    print(f"velocity spread {vel.std(0).tolist()}, acceleration spread {acc.std(0).tolist()}, "
          f"k-th nearest {reach.item():.4f}, noise {args.noise} (clip units)", flush=True)
    opt = torch.optim.Adam(stepper.parameters(), lr=args.lr)
    # GNS's decay to a hundredth plus a floor of 1e-6, over this run.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (1e-6 + (args.lr - 1e-6) * 0.01 ** (s / args.steps)) / args.lr)

    print(f"untrained: {_score(stepper, val, args.cutoff, device)}", flush=True)
    t0, run = time.time(), 0.0
    for step in range(1, args.steps + 1):
        for _ in range(args.clips):
            c = train[torch.multinomial(weights, 1).item()]
            pos = c["pos"][0]
            T, N = pos.shape[:2]
            keep = torch.randperm(N)[: args.points]
            t = torch.randint(H + 1, T, (args.samples,))
            span = torch.stack([pos[s - H - 1:s + 1, keep] for s in t.tolist()]).to(device)  # (S, H+2, n, 3)
            moving = span.diff(dim=1).norm(dim=-1).amax(1) > args.moving
            g = c["g"].to(device).expand(args.samples, -1)
            piece = c["segment"][0, keep].to(device).expand(args.samples, -1)
            loss = stepper.loss(span[:, :-1], span[:, -1], g, piece, args.noise, moving) / args.clips
            loss.backward()
            run += loss.item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 100 == 0:
            print(f"step {step}  loss {run / 100:.4f}  {(time.time() - t0) / step:.3f} s a step", flush=True)
            run = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            print(f"step {step}: {_score(stepper, val, args.cutoff, device)}", flush=True)
            Path(args.out).mkdir(parents=True, exist_ok=True)
            torch.save({"stepper": stepper.state_dict(), "args": vars(args), "step": step}, Path(args.out) / "stepper.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
