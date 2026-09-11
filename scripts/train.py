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

Run:  python scripts/preprocess.py --root DATA --out cache/kubric.pt
      python scripts/train.py --cache cache/kubric.pt --steps 20000 --batch 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset

from genpoint3d.models.flow import flow_matching_loss, sample
from genpoint3d.models.model import PointDiT


class ClipDataset(Dataset):
    """Serves cached clips, subsampling `num_points` of them per draw.

    Reads from `scripts/preprocess.py`'s output rather than decoding PNGs.
    Resampling which points are used each epoch is free augmentation and makes
    the model robust to the point count, which the paper does deliberately.
    """

    def __init__(self, clips: list[dict], num_points: int, resample: bool = True):
        self.clips, self.num_points, self.resample = clips, num_points, resample

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        c = self.clips[i]
        traj, anchor, vis = c["traj"], c["anchor"], c["visibility"]
        id_card = c.get("id_card")
        n = traj.shape[1]
        if self.num_points < n:
            idx = (torch.randperm(n)[: self.num_points] if self.resample
                   else torch.arange(self.num_points))
            traj, anchor, vis = traj[:, idx], anchor[idx], vis[:, idx]
            if id_card is not None:
                id_card = id_card[idx]

        # Empty tensors rather than None so the default collate still works.
        ctx = c.get("context")
        return (
            traj, anchor, vis,
            ctx.float() if ctx is not None else torch.zeros(0),
            id_card.float() if id_card is not None else torch.zeros(0),
        )


def split(cache: str, val_frac: float, seed: int) -> tuple[list[dict], list[dict], int | None]:
    """Deterministic train/val split over whole clips -- never over points.

    Splitting by point would put the same scene in both halves and the val
    number would be meaningless.

    Also returns the cache's feature width, which the model needs: the encoder
    chooses its own width and it need not equal the model's.
    """
    blob = torch.load(cache, weights_only=False)
    clips = blob["clips"]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * val_frac))
    return [clips[i] for i in perm[n_val:]], [clips[i] for i in perm[:n_val]], blob.get("feat_dim")


def to_device(batch, device):
    """Move a batch and normalise the optional feature tensors to None."""
    traj, anchor, vis, ctx, idc = (t.to(device) for t in batch)
    return (
        traj, anchor, vis,
        ctx if ctx.numel() else None,
        idc if idc.numel() else None,
    )


@torch.no_grad()
def evaluate(model, loader, device, steps: int = 50) -> dict:
    """Sample from pure noise and compare against ground truth.

    `ratio` is the headline: sampled RMSE over the RMSE of predicting the mean
    trajectory. A model that learnt nothing scores ~1.0.
    """
    model.eval()
    err_sq = base_sq = n = 0.0
    for batch in loader:
        traj, anchor, vis, ctx, idc = to_device(batch, device)
        vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=device) if ctx is not None else None
        pred = sample(model, anchor, num_frames=traj.shape[1], steps=steps,
                      context=ctx, visual_mask=vm, id_card=idc)
        mean_traj = traj.mean(dim=(1, 2), keepdim=True)
        err_sq += (pred - traj).pow(2).mean(-1)[vis].sum().item()
        base_sq += (mean_traj - traj).pow(2).mean(-1)[vis].sum().item()
        n += vis.sum().item()
    model.train()
    rmse, base = (err_sq / max(n, 1)) ** 0.5, (base_sq / max(n, 1)) ** 0.5
    return {"rmse": rmse, "baseline": base, "ratio": rmse / max(base, 1e-9)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="output of scripts/preprocess.py")
    p.add_argument("--out", default="outputs/run")
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
    p.add_argument("--workers", type=int, default=0)
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

    train_clips, val_clips, feat_dim = split(args.cache, args.val_frac, args.seed)
    has_feats = "context" in train_clips[0]
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
                m = evaluate(model, val_loader, device)
                print(f"  VAL step {step}  rmse {m['rmse']:.4f}"
                      f"  baseline {m['baseline']:.4f}"
                      f"  ratio {m['ratio']:.3f}   <- want << 1", flush=True)
                log.append({"step": step, **m})
                (out / "log.json").write_text(json.dumps(log, indent=2))
                torch.save(
                    {"model": model.state_dict(), "step": step, "args": vars(args)},
                    out / "ckpt.pt",
                )

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
