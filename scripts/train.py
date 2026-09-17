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
from genpoint3d.eval.metrics import tapvid3d_metrics
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

    def __init__(self, clips: list[Path | dict], num_points: int, resample: bool = True,
                 norm_mode: str = "median"):
        self.clips, self.num_points, self.resample = clips, num_points, resample
        # The cache holds metres; normalisation happens here, so switching
        # scheme is a flag rather than hours of reprocessing.
        self.norm_mode = norm_mode

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        c = self.clips[i]
        if isinstance(c, Path):
            c = torch.load(c, weights_only=False)
        clip = c if isinstance(c, CachedClip) else CachedClip.from_dict(c)

        traj, anchor = clip.normalised(self.norm_mode)
        vis = clip.visibility
        id_card = clip.id_card
        n = traj.shape[1]
        if self.num_points < n:
            idx = (torch.randperm(n)[: self.num_points] if self.resample
                   else torch.arange(self.num_points))
            traj, anchor, vis = traj[:, idx], anchor[idx], vis[:, idx]
            if id_card is not None:
                id_card = id_card[idx]

        # A dict rather than a tuple: evaluation needs the geometry that scoring
        # in metres depends on, and a positional tuple of eight was already
        # hard to read. Empty tensors rather than None so default collate works.
        #
        # Features stay in the fp16 they were cached in. Upcasting here cost
        # 452 MB per batch of 16 -- the single largest tensor in the step --
        # only for autocast to cast it straight back down. `to_device` restores
        # fp32 when autocast is off.
        ctx = clip.context
        norm = clip.norm(self.norm_mode)
        return {
            "traj": traj, "anchor": anchor, "visibility": vis,
            "context": ctx if ctx is not None else torch.zeros(0),
            "id_card": id_card if id_card is not None else torch.zeros(0),
            # metrics only -- the model never sees these
            "intrinsics": clip.intrinsics,
            "extrinsics": clip.extrinsics,
            "norm_mean": norm.mean,
            "norm_scale": norm.scale,
            "norm_traj_scale": norm.traj_scale,
        }


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
        probe = CachedClip.from_dict(torch.load(clips[0], weights_only=False, mmap=True))
    else:
        clips = torch.load(path, weights_only=False)["clips"]  # legacy single file
        probe = CachedClip.from_dict(clips[0])
    if probe.stats is None:
        raise SystemExit(f"cache in {path} has no scale statistics -- "
                         "run scripts/patch_cache.py")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * val_frac))
    return ([clips[i] for i in perm[n_val:]],
            [clips[i] for i in perm[:n_val]],
            probe.feat_dim,
            probe.context is not None)


def to_device(batch: dict, device, amp: bool = False) -> dict:
    """Move a batch, and turn the empty placeholder tensors back into None.

    Cached features are fp16. Under autocast that is what the matmuls want
    anyway; without it they have to be widened, since an fp16 input to an fp32
    Linear raises.
    """
    b = {k: v.to(device) for k, v in batch.items()}
    for k in ("context", "id_card"):
        if b[k].numel() == 0:
            b[k] = None
        elif not amp:
            b[k] = b[k].float()
    return b


def to_metres(pts: torch.Tensor, b: dict) -> torch.Tensor:
    """(B, T, N, 3) model units -> metres, using each clip's own normalisation.

    The inverse of `NormStats.invert_traj`, batched: every clip in the batch has
    its own scale, which is exactly why a threshold in model units meant a
    different physical distance per clip.
    """
    scale = (b["norm_scale"] * b["norm_traj_scale"])[:, None, None, None]
    return pts * scale + b["norm_mean"][:, None, None, :]


@torch.no_grad()
def evaluate(model, loader, device, steps: int = 50, amp: bool = False) -> dict:
    """Three numbers, deliberately distinct -- see docs/misc.md for the naming.

    val_loss  the SAME flow-matching objective as training, on held-out clips.
              Plot it against training loss: the gap is overfitting. This is
              the standard figure and the only one comparable to our own
              training curve.
    APD       `average_pts_within_thresh` from the TAP-Vid-3D benchmark, scored
              in METRES -- see genpoint3d/eval/metrics.py. The number other
              papers report. AJ and OA need a visibility prediction, so they
              appear only once the visibility head exists.
    ratio     sampled RMSE / mean-trajectory RMSE. Homemade sanity check --
              1.0 means nothing was learnt. NOT comparable to any paper.
    """
    model.eval()
    err_sq = base_sq = n = 0.0
    loss_sum = loss_n = 0.0
    metric_sums, metric_n = {}, 0

    for batch in loader:
        b = to_device(batch, device, amp)
        traj, anchor, vis, ctx, idc = (b["traj"], b["anchor"], b["visibility"],
                                       b["context"], b["id_card"])
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

        # Scored in metres, per clip -- a threshold in model units would mean a
        # different physical distance in every clip.
        m = tapvid3d_metrics(
            to_metres(pred, b), to_metres(traj, b), vis,
            b["intrinsics"], b["extrinsics"],
        )
        for k, v in m.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + v
        metric_n += 1

    model.train()
    rmse, base = (err_sq / max(n, 1)) ** 0.5, (base_sq / max(n, 1)) ** 0.5
    return {"val_loss": loss_sum / max(loss_n, 1),
            **{k: v / max(metric_n, 1) for k, v in metric_sums.items()},
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
    # 5e-4 is what both references use on Kubric -- CoTracker3's
    # launch_training_kubric_offline.sh and genpt's train_tracker_tapvid_kubric.
    p.add_argument("--lr", type=float, default=5e-4)
    # A FRACTION, not a step count. 500 fixed steps was 20% of a 2500-step run
    # and 2.5% of a 20000-step one -- the same flag meaning two different
    # things. CoTracker3 uses OneCycleLR with pct_start=0.05.
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--wdecay", type=float, default=1e-4,
                   help="CoTracker3 uses 5e-4 on Kubric; ours was 0.01, 20x more")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=2000)
    # Non-zero by default: clips are loaded lazily now, so with 0 workers the
    # GPU sits idle while the main process reads 7.3 MB/clip off Lustre and
    # converts the fp16 features to fp32. 4 is enough to stay ahead at batch 16.
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="bf16 autocast on CUDA; roughly halves memory and time")
    p.add_argument("--norm-mode", default="median",
                   choices=["median", "mean", "centroid_max"],
                   help="scene normalisation; see genpoint3d/data/cache.py")
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
        ClipDataset(train_clips, args.points, norm_mode=args.norm_mode),
        batch_size=args.batch, shuffle=True, num_workers=args.workers,
        drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        ClipDataset(val_clips, args.points, resample=False, norm_mode=args.norm_mode),
        batch_size=args.batch, shuffle=False, num_workers=args.workers,
    )

    model = PointDiT(dim=args.dim, depth=args.depth, num_heads=args.heads,
                     cross_attn=has_feats, feat_dim=feat_dim).to(device)
    print(f"model {model.num_parameters() / 1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    warmup = max(1, int(args.steps * args.warmup_frac))
    warm = torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps - warmup)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [warmup])

    # bf16 rather than fp16: it has the same exponent range as fp32, so no loss
    # scaler is needed and nothing silently underflows. A100 and newer only.
    amp = args.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"lr {args.lr:.1e} | warmup {warmup} steps ({args.warmup_frac:.0%})"
          f" | wdecay {args.wdecay:g} | amp {'bf16' if amp else 'off'}", flush=True)

    # `running` is the 100-step print window; `since_val` spans a whole
    # validation interval so the log holds train and val loss on the same
    # steps -- the gap between them IS the overfitting measurement.
    log, step, t0, running = [], 0, time.time(), 0.0
    since_val, since_val_n = 0.0, 0
    best = float('inf')
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            b = to_device(batch, device, amp)
            traj, anchor, vis, ctx, idc = (b["traj"], b["anchor"], b["visibility"],
                                           b["context"], b["id_card"])
            # All-True: pure tracking. Every frame keeps its image.
            vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=device) if ctx is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss, _ = flow_matching_loss(model, traj, anchor, mask=vis,
                                             context=ctx, visual_mask=vm, id_card=idc)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            running += loss.item()
            since_val += loss.item(); since_val_n += 1
            step += 1

            if step % 100 == 0:
                print(f"  step {step:>6}  loss {running / 100:.5f}"
                      f"  lr {sched.get_last_lr()[0]:.2e}"
                      f"  {(time.time() - t0) / step:.3f}s/it", flush=True)
                running = 0.0

            if step % args.val_every == 0 or step == args.steps:
                tv = time.time()
                m = evaluate(model, val_loader, device, amp=amp)
                print(f"  VAL step {step}  train_loss {since_val / max(since_val_n, 1):.4f}"
                      f"  val_loss {m['val_loss']:.4f}"
                      f"  APD {m['average_pts_within_thresh']:.3f}"
                      f"  ratio {m['ratio']:.3f}"
                      f"  ({time.time() - tv:.0f}s)", flush=True)
                log.append({"step": step,
                            "train_loss": since_val / max(since_val_n, 1), **m})
                since_val, since_val_n = 0.0, 0
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
