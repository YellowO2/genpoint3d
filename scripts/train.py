"""
Real training run: many clips, held-out validation, checkpoints.

`overfit.py` loads the whole dataset as one batch to prove the machinery works.
This is the opposite question -- does the model learn anything that transfers
to clips it has never seen?

The metric that matters is APD on the VAL split -- `average_pts_within_thresh`
from the TAP-Vid-3D benchmark, scored in metres. `apd_static` is the
benchmark's own Static Baseline -- the query point assumed never to move --
which is the bar a tracker has to clear.

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
from genpoint3d.data.transform import TRAJ_SCALE, TRAJ_SCALE_DISP
from genpoint3d.eval.metrics import _to_camera_t, tapvid3d_metrics
from genpoint3d.models.flow import flow_matching_loss, sample
from genpoint3d.models.model import PointDiT

# TAPIP3D clamps depth the same way before dividing, so a point behind the
# camera or at zero depth cannot produce an enormous weight.
DEPTH_MIN = 0.1


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
                 norm_mode: str = "median", target: str = "absolute"):
        self.clips, self.num_points, self.resample = clips, num_points, resample
        # The cache holds metres; normalisation happens here, so switching
        # scheme is a flag rather than hours of reprocessing.
        self.norm_mode = norm_mode
        # "absolute" denoises where each point IS. "displacement" denoises how
        # far each point has moved from ITS OWN frame-0 position.
        #
        # Measured per clip: absolute std 0.59, one-shared-anchor 0.49,
        # per-point 0.08. Only per-point removes the static layout, which is
        # 86% of an absolute target's magnitude and which the model was already
        # told via `anchor`. MolmoMotion subtracts one shared anchor instead,
        # but it emits coordinates as text in a single frame and needs them
        # mutually comparable; we do not. TAPIP3D and Tesfaldet's model both
        # use per-point displacement from the query.
        #
        # The layout is not lost: `anchor` stays absolute and is what spatial
        # RoPE reads, so where points sit relative to each other is unaffected.
        self.target = target
        self.traj_scale = TRAJ_SCALE_DISP if target == "displacement" else TRAJ_SCALE

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        c = self.clips[i]
        if isinstance(c, Path):
            c = torch.load(c, weights_only=False)
        clip = c if isinstance(c, CachedClip) else CachedClip.from_dict(c)

        # traj_scale=1.0 leaves both in scene-normalised units, so the offset
        # below is subtracted before the target is scaled, not after.
        n = clip.norm(self.norm_mode, traj_scale=1.0)
        traj, anchor = n.apply(clip.traj), n.apply(clip.anchor)
        if self.target == "displacement":
            offset = traj[:1].clone()          # (1, N, 3), each point's own start
            traj = traj - offset
        else:
            offset = torch.zeros_like(traj[:1])
        traj = traj / self.traj_scale
        vis = clip.visibility
        id_card = clip.id_card
        n = traj.shape[1]
        if self.num_points < n:
            idx = (torch.randperm(n)[: self.num_points] if self.resample
                   else torch.arange(self.num_points))
            traj, anchor, vis = traj[:, idx], anchor[idx], vis[:, idx]
            offset = offset[:, idx]
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
        norm = clip.norm(self.norm_mode, traj_scale=self.traj_scale)
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
            # (1, N, 3) scene-normalised per-point origin; zeros when the
            # target is absolute, so `to_metres` inverts both with one
            # expression.
            "norm_offset": offset,
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

    Cached features are fp16 and are widened here. Leaving them narrow looks
    like free memory under autocast, but autocast does not reach every module:
    AdaRMSNorm returns `.to(x.dtype)`, so a half tensor survives into a Linear
    holding fp32 weights and raises "mat1 and mat2 have different dtype".
    Revisit only with a measurement, not by reasoning about autocast.
    """
    b = {k: v.to(device) for k, v in batch.items()}
    for k in ("context", "id_card"):
        b[k] = None if b[k].numel() == 0 else b[k].float()
    return b


def git_commit() -> str:
    """The commit this run is from, so a number can always be traced to code.

    train.pbs prints this to the live log, but that log lives on scratch and is
    purged. A result whose code cannot be identified is not reproducible, so it
    goes in the log and the checkpoint too.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent.parent)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5,
                               cwd=Path(__file__).resolve().parent.parent)
        if out.returncode:
            return "unknown"
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        # Compute nodes have no git on PATH; an unknown commit is not worth
        # killing a run over.
        return "unknown"


class EMA:
    """A slowly-following copy of the weights, used for evaluation.

    SGD leaves the weights jittering around a minimum rather than sitting in
    it, and an average over recent steps lands closer to the middle than any
    single step does. Diffusion models gain more from this than most, and the
    paper lists EMA among its training ingredients (section 4.1).

    `decay` warms up: averaging over 1000 steps is meaningless at step 10, when
    the initialisation would still dominate.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.n = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v.detach().float(), 1 - d)

    def state_dict(self, model) -> dict:
        """The averaged weights, in the model's own dtypes."""
        sd = model.state_dict()
        return {k: (self.shadow[k].to(v.dtype) if k in self.shadow else v.clone())
                for k, v in sd.items()}


def known_frame0(b: dict) -> torch.Tensor:
    """(B, N, 3) frame 0 in model units.

    `anchor` is the same point in scene-normalised units, so this is available
    at inference exactly as it is in training -- no ground truth is used that a
    deployed model would not have.
    """
    # With a per-point displacement target frame 0 is exactly zero, which this
    # expression produces without special-casing: anchor IS each point's origin.
    return (b["anchor"] - b["norm_offset"][:, 0]) / b["norm_traj_scale"][:, None, None]


def apd_weight(b: dict, traj: torch.Tensor) -> torch.Tensor:
    """(B, T, N) weight that puts the loss in units of the APD threshold.

    The metric allows a point an error of `thresh * depth / focal`, so a point
    20 m away is given ten times the slack of one 2 m away. An unweighted loss
    does not know that: it spends the same effort on a distant point that was
    going to pass anyway as on a near point that was always going to fail.

    Dividing the error by `depth / focal` measures it in threshold units
    instead, which is what TAPIP3D's `scale_loss_by_depth` does
    (`training/criterion.py:165`, "Normalize to match with the APD metric").

    Normalised to mean 1 over the batch so the loss keeps its magnitude and the
    learning rate stays comparable across runs -- only the RELATIVE weighting
    of points is the point here.
    """
    gt_cam = _to_camera_t(to_metres(traj, b), b["extrinsics"])
    depth = gt_cam[..., 2].abs().clamp(min=DEPTH_MIN)                  # (B, T, N)
    focal = (b["intrinsics"][..., 0, 0] * b["intrinsics"][..., 1, 1]).sqrt()
    w = focal[..., None] / depth
    return w / w.mean().clamp(min=1e-8)


def to_metres(pts: torch.Tensor, b: dict) -> torch.Tensor:
    """(B, T, N, 3) model units -> metres, using each clip's own normalisation.

    The inverse of `NormStats.invert_traj`, batched: every clip in the batch has
    its own scale, which is exactly why a threshold in model units meant a
    different physical distance per clip.
    """
    ts = b["norm_traj_scale"][:, None, None, None]
    scale = b["norm_scale"][:, None, None, None]
    scene = pts * ts + b["norm_offset"]                    # undo the target offset
    return scene * scale + b["norm_mean"][:, None, None, :]


@torch.no_grad()
def evaluate(model, loader, device, steps: int = 50, amp: bool = False,
             anchor_frame0: bool = False, loss_type: str = "l2") -> dict:
    """Two numbers, both standard -- no homemade units.

    val_loss  the SAME flow-matching objective as training, on held-out clips.
              Plot it against training loss: the gap is overfitting.
    APD       `average_pts_within_thresh` from the TAP-Vid-3D benchmark, scored
              in METRES -- see genpoint3d/eval/metrics.py. The number other
              papers report. AJ and OA need a visibility prediction, so they
              appear only once the visibility head exists.

    `apd_static` is the benchmark's own Static Baseline (TAPVid-3D, table 3):
    take the query point's 3D position and assume it never moves. That is a
    real bar rather than a formality -- it scores 9.4 there, above TAPIR-3D's
    5.9 and not far below SpatialTracker's 15.5.
    """
    model.eval()
    loss_sum = loss_n = 0.0
    sums, n_batches = {}, 0

    for batch in loader:
        b = to_device(batch, device, amp)
        traj, anchor, vis, ctx, idc = (b["traj"], b["anchor"], b["visibility"],
                                       b["context"], b["id_card"])
        vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=device) if ctx is not None else None

        kx0 = known_frame0(b) if anchor_frame0 else None
        l, _ = flow_matching_loss(model, traj, anchor, mask=vis, known_x0=kx0,
                                  loss_type=loss_type,
                                  context=ctx, visual_mask=vm, id_card=idc)
        loss_sum += l.item(); loss_n += 1

        pred = sample(model, anchor, num_frames=traj.shape[1], steps=steps,
                      known_x0=kx0, context=ctx, visual_mask=vm, id_card=idc)

        # Scored in metres, per clip -- a threshold in model units would mean a
        # different physical distance in every clip.
        gt_m = to_metres(traj, b)
        score = lambda p: tapvid3d_metrics(p, gt_m, vis, b["intrinsics"], b["extrinsics"])
        m = score(to_metres(pred, b))
        # Static Baseline: each point stays where it started. Free -- no
        # sampling -- and the number a reader will ask for first.
        m["apd_static"] = score(
            gt_m[:, :1].expand_as(gt_m)
        )["average_pts_within_thresh"]

        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v
        n_batches += 1

    model.train()
    return {"val_loss": loss_sum / max(loss_n, 1),
            **{k: v / max(n_batches, 1) for k, v in sums.items()}}


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
    # On by default: an unweighted loss optimises something the metric does not
    # measure. 0 reproduces every run before 2026-09-18.
    p.add_argument("--depth-scaled-loss", type=int, default=1, choices=[0, 1])
    # Frame 0 is handed to the model as `anchor` and was still its worst-scoring
    # frame in absolute terms (APD 0.028 on run3493), so it is pinned rather
    # than denoised. 0 reproduces the earlier behaviour.
    p.add_argument("--anchor-frame0", type=int, default=1, choices=[0, 1])
    # Evaluate and checkpoint the averaged weights, not the jittering ones.
    # 0 disables. The paper lists EMA among its training ingredients.
    p.add_argument("--ema", type=float, default=0.999)
    # l21 is the Euclidean distance, which is what the metric counts. l2 squares
    # it, so a few catastrophic points dominate -- visible in run3493 as rmse
    # sitting flat at 0.11 while APD tripled.
    p.add_argument("--loss-type", default="l21", choices=["l2", "l21"])
    p.add_argument("--target", default="absolute",
                   choices=["absolute", "displacement"],
                   help="what the model denoises towards; see ClipDataset")
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

    commit = git_commit()
    train_clips, val_clips, feat_dim, has_feats = split(args.cache, args.val_frac, args.seed)
    print(f"device {device} | {len(train_clips)} train clips, {len(val_clips)} val clips"
          f" | {'TRACKING (with images)' if has_feats else 'no images (step 2)'}", flush=True)

    train_loader = DataLoader(
        ClipDataset(train_clips, args.points, norm_mode=args.norm_mode,
                    target=args.target),
        batch_size=args.batch, shuffle=True, num_workers=args.workers,
        drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        ClipDataset(val_clips, args.points, resample=False,
                    norm_mode=args.norm_mode, target=args.target),
        batch_size=args.batch, shuffle=False, num_workers=args.workers,
    )

    # The denoising target must sit near unit variance: flow matching mixes it
    # with x0 ~ N(0, I), and a target far below 1 teaches the model to output
    # -x0 while the loss still falls. That failure is silent, so check it once
    # against a real batch rather than trusting the constant.
    probe = next(iter(train_loader))
    tstd = probe["traj"][probe["visibility"]].std().item()
    print(f"target std {tstd:.3f} ({args.target}, TRAJ_SCALE"
          f" {train_loader.dataset.traj_scale})", flush=True)
    if not 0.3 < tstd < 3.0:
        raise SystemExit(
            f"target std {tstd:.3f} is outside [0.3, 3.0] -- recalibrate with\n"
            f"  python scripts/calibrate_traj_scale.py --cache {args.cache}"
            f" --target {args.target}")
    del probe

    model = PointDiT(dim=args.dim, depth=args.depth, num_heads=args.heads,
                     cross_attn=has_feats, feat_dim=feat_dim).to(device)
    print(f"model {model.num_parameters() / 1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    ema = EMA(model, args.ema) if args.ema > 0 else None
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
    best = -float('inf')
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            b = to_device(batch, device, amp)
            traj, anchor, vis, ctx, idc = (b["traj"], b["anchor"], b["visibility"],
                                           b["context"], b["id_card"])
            # All-True: pure tracking. Every frame keeps its image.
            vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=device) if ctx is not None else None
            w = apd_weight(b, traj) if args.depth_scaled_loss else None
            kx0 = known_frame0(b) if args.anchor_frame0 else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss, _ = flow_matching_loss(model, traj, anchor, mask=vis, weight=w,
                                             known_x0=kx0, loss_type=args.loss_type,
                                             context=ctx, visual_mask=vm, id_card=idc)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(model)
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
                # Score the averaged weights, which are also what gets saved --
                # evaluating the live weights and shipping the EMA ones would
                # report a number for a checkpoint nobody has.
                live = None
                if ema is not None:
                    live = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(ema.state_dict(model))
                m = evaluate(model, val_loader, device, amp=amp,
                             anchor_frame0=args.anchor_frame0,
                             loss_type=args.loss_type)
                print(f"  VAL step {step}  train_loss {since_val / max(since_val_n, 1):.4f}"
                      f"  val_loss {m['val_loss']:.4f}"
                      f"  APD {m['average_pts_within_thresh']:.3f}"
                      f"  (static {m['apd_static']:.3f})"
                      f"  ({time.time() - tv:.0f}s)", flush=True)
                log.append({"step": step,
                            "train_loss": since_val / max(since_val_n, 1), **m})
                since_val, since_val_n = 0.0, 0
                (out / "log.json").write_text(json.dumps(log, indent=2))
                # Config and commit alongside the curve, in their own file so
                # log.json stays a plain list that plot_log.py can read.
                (out / "run.json").write_text(json.dumps(
                    {"commit": commit, "args": vars(args),
                     "clips": {"train": len(train_clips), "val": len(val_clips)}},
                    indent=2))
                ckpt = {"model": model.state_dict(), "step": step,
                        "args": vars(args), "val": m, "commit": commit}
                torch.save(ckpt, out / "ckpt.pt")
                # Keep the best separately: val typically bottoms out and then
                # drifts up as the model overfits, so the LAST checkpoint is
                # not the one you want.
                # Maximise APD, as genpt's model_checkpoint does
                # (monitor: val/d_all_avg/avg, mode: max).
                if m["average_pts_within_thresh"] > best:
                    best = m["average_pts_within_thresh"]
                    torch.save(ckpt, out / "best.pt")
                    print(f"       new best APD {best:.3f}", flush=True)

                if live is not None:
                    model.load_state_dict(live)   # training continues on the live weights

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
