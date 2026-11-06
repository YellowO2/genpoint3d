"""APD by frame after each round of the refinement, every frame given its image:
where along the clip a schedule loses its tracks, and whether later rounds win
them back.

With a schedule (fbfb), round k is sweep k, so stage k is the clip after sweep
k. Without one (four rounds a window), stage k is each window's guess after
its round k, so only the last stage is a state the clip is ever in.

Run:  python scripts/sweep_curve.py --ckpt outputs/RUN/best.pt --cache DIR --frames-cache FRAMES
"""

import argparse

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.score.Run import Run
from genpoint3d.train.tools.batch import fine_inputs, to_device, to_metres, visual_mask
from genpoint3d.train.tools.metrics import clip_mean, tapvid3d_metrics
from genpoint3d.train.tools.regress import Sweeps


def apd(pred, gt, mask, b, t=None, key="average_pts_within_thresh", scaling="none"):
    """(B,) per clip, over the entries in `mask`, at frame `t` or the whole clip."""
    s = slice(None) if t is None else slice(t, t + 1)
    return tapvid3d_metrics(pred[:, s], gt[:, s], mask[:, s], b["intrinsics_256"][:, s], b["extrinsics"][:, s],
                            scaling=scaling, per_clip=True)[key].cpu()


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--frames-cache", required=True)
    p.add_argument("--points", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run.load(args.ckpt, device)
    model, s = run.model, run.settings
    ds = run.dataset(args.cache, args.frames_cache, points=args.points)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    amp = device.type == "cuda"
    print(f"{len(ds)} clips | step {run.step} of {args.ckpt} | schedule {s.schedule!r} "
          f"rounds {s.refine_iters} window {s.window} overlap {s.overlap}", flush=True)

    rows = {}       # (stage, name, frame or None) -> per-clip values
    for batch in loader:
        b = to_device(batch, device)
        traj, vis = b["traj"], b["visibility"].bool()
        T = traj.shape[1]
        cond = dict(context=b["context"], visual_mask=visual_mask(b, None), id_card=b["id_card"],
                    patch_xyz=b["patch_xyz"], **fine_inputs(model, b, amp))
        sweeps = Sweeps(model, b["anchor"], T, s.refine_iters, s.schedule, s.window, s.overlap, cond)
        stages = {}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            for r in sweeps:
                stage = stages.setdefault(r.i, torch.zeros_like(traj))
                stage[:, r.start:r.end] = r.g.float()
        gt = to_metres(traj, b)
        for k, guess in stages.items():
            pred = to_metres(guess, b)
            vals = {("seen", None): apd(pred, gt, vis, b, scaling="median"),
                    ("hidden", None): apd(pred, gt, ~vis, b, scaling="median")}
            for t in range(T):
                vals["seen", t] = apd(pred, gt, vis, b, t)
                vals["hidden", t] = apd(pred, gt, ~vis, b, t)
                vals["kept", t] = apd(pred, gt, vis, b, t, key="pts_within_16")
            for (name, t), v in vals.items():
                rows.setdefault((k, name, t), []).append(v)
        if "spans" not in locals():
            spans = sweeps.spans

    stages = sorted({k for k, _, _ in rows})
    mean = {key: clip_mean(v) for key, v in rows.items()}
    print(f"\nwindows {spans}")
    print("\nwhole clip (TAPVid-3D's median rescale), by stage:")
    for k in stages:
        print(f"  stage {k}  seen {mean[k, 'seen', None]:.4f}  hidden {mean[k, 'hidden', None]:.4f}")
    for name, what in (("seen", "APD of the seen points"), ("kept", "share of seen points within the loosest "
                                                                   "threshold (16 px at their depth)"),
                       ("hidden", "APD of the hidden points")):
        print(f"\n{what}, by frame (no rescale):")
        print("  frame " + " ".join(f"{f'stage {k}':>8}" for k in stages))
        for t in range(T):
            print(f"  {t:>5} " + " ".join(f"{mean[k, name, t]:>8.4f}" for k in stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
