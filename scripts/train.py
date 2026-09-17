"""
Real training run: many clips, held-out validation, checkpoints.

`overfit.py` loads the whole dataset as one batch to prove the machinery works.
This is the opposite question -- does the model learn anything that transfers
to clips it has never seen?

The metric that matters is `ratio` on the VAL split: RMSE of sampled
trajectories divided by the RMSE of the mean trajectory. 1.0 means nothing was
learnt. Below 1.0 means real signal.

With a feature cache this trains pure TRACKING: every frame keeps its image, so
`visual_mask` is all-True and the null embedding is never used. Masking is
deliberately left off for the first learning run -- forecasting is far harder
than tracking, and training both at once splits the signal between them. If
tracking does not work, forecasting never would. Turn it on with
`--random-cutoff` once tracking is learning.

Without a feature cache this falls back to the step-2 model, which sees no
images at all -- a pure motion prior rather than tracking.

Run:  python scripts/preprocess.py --root DATA --out local/cache/kubric
      python scripts/train.py --cache local/cache/kubric --steps 20000 --batch 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset

from genpoint3d.data.cache import CachedClip
from genpoint3d.models.flow import flow_matching_loss, sample
from genpoint3d.models.model import PointDiT


class ClipDataset(Dataset):
    """Serves cached clips, subsampling `num_points` of them per draw.

    Reads from `scripts/preprocess.py`'s output rather than decoding PNGs.
    Resampling which points are used each epoch is free augmentation and makes
    the model robust to the point count, which the paper does deliberately.

    Clips are held as PATHS and loaded in `__getitem__`, not up front. At 7.3 MB
    per cached clip an eager load is 25 GB for 3500 clips -- and every dataloader
    worker gets its own copy, so it OOMs the moment `--workers` is non-zero.
    Lazily, only the clips in flight are resident. This is what makes workers
    affordable, which is what keeps the GPU fed.
    """

    def __init__(self, clips: list[Path | dict], num_points: int, resample: bool = True):
        self.clips, self.num_points, self.resample = clips, num_points, resample

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        c = self.clips[i]
        if isinstance(c, Path):
            c = torch.load(c, weights_only=False)
        clip = c if isinstance(c, CachedClip) else CachedClip.from_dict(c)

        traj, anchor, vis = clip.traj, clip.anchor, clip.visibility
        id_card = clip.id_card
        n = traj.shape[1]
        if self.num_points < n:
            idx = (torch.randperm(n)[: self.num_points] if self.resample
                   else torch.arange(self.num_points))
            traj, anchor, vis = traj[:, idx], anchor[idx], vis[:, idx]
            if id_card is not None:
                id_card = id_card[idx]

        # Empty tensors rather than None so the default collate still works.
        ctx = clip.context
        return (
            traj, anchor, vis,
            ctx.float() if ctx is not None else torch.zeros(0),
            id_card.float() if id_card is not None else torch.zeros(0),
        )


def split(cache: str, val_frac: float, seed: int):
    """Deterministic train/val split over whole clips -- never over points.

    Splitting by point would put the same scene in both halves and the val
    number would be meaningless.

    Returns paths, not loaded clips -- see `ClipDataset`. Also returns the
    cache's feature width and whether it has features, both read from a single
    probe clip, since the encoder chooses its own width and it need not equal
    the model's.
    """
    path = Path(cache)
    if path.is_dir():
        # one .pt per clip -- written incrementally so preprocessing resumes
        clips: list[Path | dict] = sorted(path.glob("*.pt"))
        if not clips:
            raise SystemExit(f"no cached clips in {path} -- run scripts/preprocess.py first")
        probe = CachedClip.from_dict(torch.load(clips[0], weights_only=False))
    else:
        clips = torch.load(path, weights_only=False)["clips"]  # legacy single file
        probe = CachedClip.from_dict(clips[0])

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * val_frac))
    if probe.norm is None:
        print("WARNING: cache has no `norm` -- metrics cannot be reported in "
              "metres. Re-run scripts/preprocess.py to fix.", flush=True)
    return ([clips[i] for i in perm[n_val:]],
            [clips[i] for i in perm[:n_val]],
            probe.feat_dim,
            probe.context is not None)


def to_device(batch, device):
    """Move a batch and normalise the optional feature tensors to None."""
    traj, anchor, vis, ctx, idc = (t.to(device) for t in batch)
    return (
        traj, anchor, vis,
        ctx if ctx.numel() else None,
        idc if idc.numel() else None,
    )


# Distance thresholds for delta_avg, in model units. TAP-Vid uses pixel
# thresholds; ours are metric-ish, so these are indicative only until we
# calibrate them against a published protocol.
DELTA_THRESHOLDS = (0.05, 0.1, 0.2, 0.4, 0.8)


@torch.no_grad()
def evaluate(model, loader, device, steps: int = 50) -> dict:
    """Three numbers, deliberately distinct -- see docs/misc.md for the naming.

    val_loss  the SAME flow-matching objective as training, on held-out clips.
              Plot it against training loss: the gap is overfitting. This is
              the standard figure and the only one comparable to our own
              training curve.
    delta_avg fraction of predicted points within a distance threshold of the
              truth. The TAP-Vid-3D style metric; the closest thing we have to
              something other papers report.
    ratio     sampled RMSE / mean-trajectory RMSE. Homemade sanity check --
              1.0 means nothing was learnt. NOT comparable to any paper.
    """
    model.eval()
    err_sq = base_sq = n = 0.0
    loss_sum = loss_n = 0.0
    hits = {t: 0.0 for t in DELTA_THRESHOLDS}

    for batch in loader:
        traj, anchor, vis, ctx, idc = to_device(batch, device)
        vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=device) if ctx is not None else None

        l, _ = flow_matching_loss(model, traj, anchor, mask=vis,
                                  context=ctx, visual_mask=vm, id_card=idc)
        loss_sum += l.item(); loss_n += 1

        pred = sample(model, anchor, num_frames=traj.shape[1], steps=steps,
                      context=ctx, visual_mask=vm, id_card=idc)
        mean_traj = traj.mean(dim=(1, 2), keepdim=True)
        err_sq += (pred - traj).pow(2).mean(-1)[vis].sum().item()
        base_sq += (mean_traj - traj).pow(2).mean(-1)[vis].sum().item()
        n += vis.sum().item()

        dist = (pred - traj).norm(dim=-1)[vis]          # model units
        for t in DELTA_THRESHOLDS:
            hits[t] += (dist < t).sum().item()

    model.train()
    rmse, base = (err_sq / max(n, 1)) ** 0.5, (base_sq / max(n, 1)) ** 0.5
    deltas = {f"d{t}": hits[t] / max(n, 1) for t in DELTA_THRESHOLDS}
    return {"val_loss": loss_sum / max(loss_n, 1),
            "delta_avg": sum(deltas.values()) / len(deltas),
            **deltas,
            "rmse": rmse, "baseline": base, "ratio": rmse / max(base, 1e-9)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="cache DIRECTORY from scripts/preprocess.py")
    p.add_argument("--out", default="local/outputs/run")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--points", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=2000)
    # Non-zero by default: clips are loaded lazily now, so with 0 workers the
    # GPU sits idle while the main process reads 7.3 MB/clip off Lustre and
    # converts the fp16 features to fp32. 4 is enough to stay ahead at batch 16.
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_clips, val_clips, feat_dim, has_feats = split(args.cache, args.val_frac, args.seed)
    print(f"device {device} | {len(train_clips)} train clips, {len(val_clips)} val clips"
          f" | {'TRACKING (with images)' if has_feats else 'no images (step 2)'}", flush=True)

    train_loader = DataLoader(
        ClipDataset(train_clips, args.points),
        batch_size=args.batch, shuffle=True, num_workers=args.workers,
        drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        ClipDataset(val_clips, args.points, resample=False),
        batch_size=args.batch, shuffle=False, num_workers=args.workers,
    )

    model = PointDiT(dim=args.dim, depth=args.depth, num_heads=args.heads,
                     cross_attn=has_feats, feat_dim=feat_dim).to(device)
    print(f"model {model.num_parameters() / 1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, args.warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps - args.warmup)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [args.warmup])

    log, step, t0, running = [], 0, time.time(), 0.0
    best = float('inf')
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            traj, anchor, vis, ctx, idc = to_device(batch, device)
            # All-True: pure tracking. Every frame keeps its image.
            vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=device) if ctx is not None else None
            loss, _ = flow_matching_loss(model, traj, anchor, mask=vis,
                                         context=ctx, visual_mask=vm, id_card=idc)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            running += loss.item()
            step += 1

            if step % 100 == 0:
                print(f"  step {step:>6}  loss {running / 100:.5f}"
                      f"  lr {sched.get_last_lr()[0]:.2e}"
                      f"  {(time.time() - t0) / step:.3f}s/it", flush=True)
                running = 0.0

            if step % args.val_every == 0 or step == args.steps:
                tv = time.time()
                m = evaluate(model, val_loader, device)
                print(f"  VAL step {step}  val_loss {m['val_loss']:.4f}"
                      f"  delta_avg {m['delta_avg']:.3f}"
                      f"  ratio {m['ratio']:.3f}"
                      f"  ({time.time() - tv:.0f}s)", flush=True)
                log.append({"step": step, **m})
                (out / "log.json").write_text(json.dumps(log, indent=2))
                ckpt = {"model": model.state_dict(), "step": step,
                        "args": vars(args), "val": m}
                torch.save(ckpt, out / "ckpt.pt")
                # Keep the best separately: val typically bottoms out and then
                # drifts up as the model overfits, so the LAST checkpoint is
                # not the one you want.
                if m["ratio"] < best:
                    best = m["ratio"]
                    torch.save(ckpt, out / "best.pt")
                    print(f"       new best ratio {best:.3f}", flush=True)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
