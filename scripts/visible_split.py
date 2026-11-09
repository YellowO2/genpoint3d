"""Visible-point accuracy split by whether a point is ever hidden: APD over the
frames each point is seen, for the points seen in every frame of the clip and
for the rest (hidden for some frames, seen in others), all and moving (more
than 4 px from the query position). Every frame keeps its image.

Run:  python scripts/visible_split.py --ckpt outputs/RUN/best.pt --cache DIR --frames-cache FRAMES
"""

import argparse

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.score.Run import Run
from genpoint3d.train.tools.batch import fine_inputs, to_device, to_metres, visual_mask
from genpoint3d.train.tools.metrics import clip_mean, motion_px, tapvid3d_metrics
from genpoint3d.train.tools.regress import refine


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
    amp = device.type == "cuda"
    rows, shares = {}, []
    for batch in DataLoader(ds, batch_size=1, shuffle=False, num_workers=4):
        b = to_device(batch, device)
        vis = b["visibility"].bool()
        cond = dict(context=b["context"], visual_mask=visual_mask(b, None), id_card=b["id_card"],
                    patch_xyz=b["patch_xyz"], **fine_inputs(model, b, amp))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred, _ = refine(model, b["anchor"], vis.shape[1], iters=s.refine_iters, window=s.window,
                             overlap=s.overlap, schedule=s.schedule, **cond)
        pred_m, gt_m = to_metres(pred.float(), b), to_metres(b["traj"], b)
        always = vis.all(1)                                                  # (B, N)
        moving = motion_px(gt_m, vis, b["intrinsics_256"], b["extrinsics"], b["query_t"]) > 4
        shares.append(always.float().mean().item())
        for name, pts in (("always seen", always), ("sometimes hidden", ~always)):
            for which, sel, scaling in (("all", pts, "median"), ("moving", pts & moving, "none")):
                m = vis & sel[:, None]
                rows.setdefault((name, which), []).append(tapvid3d_metrics(
                    pred_m, gt_m, m, b["intrinsics_256"], b["extrinsics"], scaling=scaling,
                    per_clip=True)["average_pts_within_thresh"].cpu())
    print(f"{len(ds)} clips | {args.ckpt} | points seen in every frame: {sum(shares) / len(shares):.3f}")
    print(f"  {'points':<18} {'all (median rescale)':>21} {'moving (no rescale)':>20}")
    for name in ("always seen", "sometimes hidden"):
        print(f"  {name:<18} {clip_mean(rows[name, 'all']):>21.4f} {clip_mean(rows[name, 'moving']):>20.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
