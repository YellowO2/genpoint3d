"""Score a trained checkpoint on every clip of a cache, with no training and no
checkpoint selection: the number to report for a held-out test set.

Run:  python scripts/evaluate.py --ckpt outputs/RUN/best.pt --cache DIR --frames-cache FRAMES
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.score.Run import Run
from genpoint3d.train.score.main import score
from genpoint3d.train.tools.batch import fine_inputs, to_device, to_metres, visual_mask
from genpoint3d.train.tools.metrics import HIDDEN_KEYS, clip_mean, error_and_motion, motion_px, motion_table, tapvid3d_metrics
from genpoint3d.train.tools.regress import refine


@torch.no_grad()
def per_frame_apd(run: Run, loader, device) -> list[float]:
    """APD by frame over the visible points that move more than 4 px from
    their query position, every frame given its image, with no rescale; and
    printed, every seen point's error against how far it has moved."""
    model, s = run.model, run.settings
    amp = device.type == "cuda"
    vals, errs, disps = None, [], []
    for batch in loader:
        b = to_device(batch, device)
        traj, vis = b["traj"], b["visibility"]
        cond = dict(context=b["context"], visual_mask=visual_mask(b, None), id_card=b["id_card"],
                    patch_xyz=b["patch_xyz"], **fine_inputs(model, b, amp))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred, _ = refine(model, b["anchor"], traj.shape[1], window=s.window, overlap=s.overlap, schedule=s.schedule, **cond)
        pred_m, gt_m = to_metres(pred.float(), b), to_metres(traj, b)
        e, d = error_and_motion(pred_m, gt_m, vis, b["intrinsics_256"], b["extrinsics"], b["query_t"])
        errs.append(e)
        disps.append(d)
        seen = vis.bool() & (motion_px(gt_m, vis, b["intrinsics_256"], b["extrinsics"], b["query_t"]) > 4)[:, None]
        vals = vals or [[] for _ in range(traj.shape[1])]
        for t in range(traj.shape[1]):
            m = tapvid3d_metrics(pred_m[:, t:t + 1], gt_m[:, t:t + 1], seen[:, t:t + 1],
                                 b["intrinsics_256"][:, t:t + 1], b["extrinsics"][:, t:t + 1],
                                 scaling="none", per_clip=True)
            vals[t].append(m["average_pts_within_thresh"].cpu())
    print("\n    every seen point but at its query frame: error against how far it has moved")
    print(motion_table(torch.cat(errs), torch.cat(disps)))
    return [clip_mean(v) for v in vals]


# Each ablation destroys one input; a score that does not move names a pathway
# the model ignores.
ABLATIONS = ("swap-features", "noise-features", "shuffle-pos", "swap-id", "swap-frames")


class Ablated:
    """A loader whose CPU batches come out with one input corrupted."""

    def __init__(self, loader, kind: str) -> None:
        self.loader, self.kind = loader, kind

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for b in self.loader:
            b = dict(b)
            ctx = b["context"]
            if self.kind == "swap-features":
                b["context"] = ctx.roll(1, 0)          # needs batch > 1
            elif self.kind == "noise-features":
                b["context"] = (torch.randn_like(ctx.float()) * ctx.float().std() + ctx.float().mean()).to(ctx.dtype)
            elif self.kind == "shuffle-pos":
                b["patch_xyz"] = b["patch_xyz"][:, :, torch.randperm(b["patch_xyz"].shape[2])]
            elif self.kind == "swap-id":
                b["id_card"] = b["id_card"].roll(1, 0)
            elif self.kind == "swap-frames":
                b["frames"], b["depth"] = b["frames"].roll(1, 0), b["depth"].roll(1, 0)
            yield b


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="best.pt or ckpt.pt from a run")
    p.add_argument("--cache", required=True, help="cache directory to score, every clip")
    p.add_argument("--frames-cache", required=True, help="RGB and depth of --cache's clips")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--points", type=int, default=None, help="default: the run's own")
    p.add_argument("--out", default=None, help="write the metrics as JSON here")
    p.add_argument("--ablate", nargs="*", default=None, choices=ABLATIONS,
                   help="also score with one input corrupted; no argument means all")
    p.add_argument("--space", default="3d", choices=["3d", "2d"],
                   help="2d scores the projections at TAP-Vid's pixel thresholds, blind to depth")
    p.add_argument("--scaling", default="median", choices=["median", "none"],
                   help="median: TAPVid-3D's one global rescale of the prediction")
    p.add_argument("--min-motion-px", type=float, default=0.0,
                   help="score only points that move more than this many pixels")
    p.add_argument("--oracle-axis", default="", choices=["", "x", "y", "z"],
                   help="that channel of the prediction set to the truth before scoring")
    p.add_argument("--eval-cutoff", type=int, default=0, metavar="C",
                   help="frames C on get no image, observed (_obs) and forecast (_fut) scored apart")
    p.add_argument("--per-frame", action="store_true", help="also APD by frame, every frame seen")
    p.add_argument("--spawn", type=int, default=0, help="at most this many points added where new surface has none")
    p.add_argument("--spawn-gap", type=float, default=0.05, help="a gap that wide (a share of the frame's width) gets one")
    args = p.parse_args()
    if args.eval_cutoff == 1 or args.eval_cutoff < 0:
        p.error("--eval-cutoff is 0 (off) or at least 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run.load(args.ckpt, device)
    ds = run.dataset(args.cache, args.frames_cache, points=args.points, query_before=args.eval_cutoff)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)
    print(f"{len(ds)} clips | {device} | points {ds.num_points} | step {run.step} of {args.ckpt}", flush=True)

    t0 = time.time()
    per_frame = per_frame_apd(run, loader, device) if args.per_frame else None
    judging = run.judging(device, cutoff=args.eval_cutoff, space=args.space, oracle_axis=args.oracle_axis,
                          scaling=args.scaling, min_motion_px=args.min_motion_px,
                          spawn=args.spawn, spawn_gap=args.spawn_gap)
    m = score(run.model, loader, judging, split="test")
    print(f"\nscored in {(time.time() - t0) / 60:.1f} min\n", flush=True)

    for k in ("average_pts_within_thresh", "apd_static", "apd_moving", "apd_moving_static", "val_loss",
              "moving_frac", "disp_ratio", "disp_cos"):
        if k in m:
            print(f"  {k:<26} {m[k]:.4f}")
    if "moving_clips" in m:
        print(f"  {'moving_clips':<26} {m['moving_clips']} of {len(ds)}")
    groups = (("group vectors against the true pieces (right: of those put with a point, truly on its "
               "piece; found: of those on it, put with it):", [k for k in ("group_right", "group_found") if k in m]),
              ("visibility head, and AJ and OA on the observed frames:",
               [k for k in ("vis_acc", "vis_acc_hidden", "vis_acc_visible", "average_jaccard",
                            "occlusion_accuracy") if k in m]),
              ("hidden on the observed frames, beside the last-seen and constant-speed baselines:",
               [k for k in HIDDEN_KEYS if k in m]),
              (f"forecast, frames {args.eval_cutoff}+ without images:",
               [k for k in m if k.endswith("_obs") or "_fut" in k]))
    for title, keys in groups:
        if keys:
            print(f"\n  {title}")
            for k in keys:
                print(f"  {k:<30} {m[k]:.4f}")
    print()
    for t in (1, 2, 4, 8, 16):
        print(f"  pts_within_{t:<15} {m[f'pts_within_{t}']:.4f}")

    if per_frame is not None:
        print("\n  APD by frame, moving seen points, every frame with its image:")
        top = max((v for v in per_frame if v == v), default=0.0)
        for t, v in enumerate(per_frame):
            bar = "" if v != v else "#" * int(round(v / max(top, 1e-9) * 40))
            print(f"    frame {t:>2}  {v:.4f}  {bar}")

    if args.ablate is not None:
        print("\n  Ablations -- APD with one input destroyed:\n")
        print(f"    {'condition':<16} {'APD':>7} {'vs intact':>10}")
        print(f"    {'intact':<16} {m['average_pts_within_thresh']:>7.4f} {'--':>10}")
        for kind in args.ablate or ABLATIONS:
            apd = score(run.model, Ablated(loader, kind), judging, split=kind)["average_pts_within_thresh"]
            print(f"    {kind:<16} {apd:>7.4f} {apd / max(m['average_pts_within_thresh'], 1e-9):>9.2f}x", flush=True)
            m[f"ablate_{kind}"] = apd

    if args.out:
        Path(args.out).write_text(json.dumps({"ckpt": args.ckpt, "cache": args.cache, "step": run.step,
                                              "clips": len(ds), "eval_cutoff": args.eval_cutoff, **m}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
