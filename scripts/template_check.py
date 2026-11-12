"""What a fresher matcher template is worth on a run's weights, without
training: the tracks with each window's template the point's look in its query
frame; its latest clear look the window before settled, read where the model
guessed it (`--template latest`); and the same look read where it truly is, in
a frame where it truly is visible. If the true one helps and the guessed one
does not, a fresh look is right and only placing it is wrong; if neither
helps, the query look is not what limits the matcher. Every frame keeps its
image.

Run:  python scripts/template_check.py --ckpt outputs/RUN/best.pt --cache DIR --frames-cache FRAMES
"""

import argparse

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.score.Run import Run
from genpoint3d.train.tools.batch import fine_inputs, to_device, to_metres, visual_mask
from genpoint3d.train.tools.metrics import clip_mean, motion_px, tapvid3d_metrics
from genpoint3d.train.tools.regress import Sweeps

_truth = {}


def _true_template(self, start: int):
    """`Sweeps._template` with the look read where the point truly is, in the
    latest frame the window before settled where it truly is visible."""
    q = self.cond["query_t"]
    if start == 0:
        return q, self.anchor
    prev_end = max(e for s, e in self.spans if s < start)
    T = self.whole.shape[1]
    t = torch.arange(T, device=q.device)[None, :, None]
    clear = (t >= q[:, None]) & (t < prev_end) & _truth["vis"]
    f = torch.where(clear, t, torch.full_like(t, -1)).amax(1)
    has = f >= 0
    pos = self.model.lookup_pos(_truth["x"], self.anchor)
    at = pos.gather(1, f.clamp_min(0)[:, None, :, None].expand(-1, 1, -1, 3))[:, 0]
    return torch.where(has, f, q), torch.where(has[..., None], at, self.anchor.float())


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
    print(f"{len(ds)} clips | step {run.step} of {args.ckpt}", flush=True)
    guessed = Sweeps._template
    variants = (("query look", "query", guessed), ("latest, guessed", "latest", guessed),
                ("latest, true", "latest", _true_template))

    rows = {}
    for batch in DataLoader(ds, batch_size=1, shuffle=False, num_workers=4):
        b = to_device(batch, device)
        traj, vis, q = b["traj"], b["visibility"].bool(), b["query_t"]
        T = traj.shape[1]
        _truth.update(x=traj.float(), vis=vis)
        cond = dict(context=b["context"], visual_mask=visual_mask(b, None), id_card=b["id_card"],
                    patch_xyz=b["patch_xyz"], **fine_inputs(model, b, amp))
        gt = to_metres(traj, b)
        moving = (motion_px(gt, vis, b["intrinsics_256"], b["extrinsics"], q) > 4)[:, None]
        late = (torch.arange(T, device=device) >= 12)[None, :, None]
        for name, template, pick in variants:
            model.template, Sweeps._template = template, pick
            sweeps = Sweeps(model, b["anchor"], T, s.refine_iters, s.schedule, s.window, s.overlap, cond)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in sweeps:
                    pass
            pred = to_metres(sweeps.whole.float(), b)
            for which, m, scaling in (("all", vis, "median"), ("moving", vis & moving, "none"),
                                      ("moving, frames 12-23", vis & moving & late, "none")):
                apd = tapvid3d_metrics(pred, gt, m, b["intrinsics_256"], b["extrinsics"], scaling=scaling,
                                       per_clip=True)["average_pts_within_thresh"].cpu()
                rows.setdefault((name, which), []).append(apd)
        Sweeps._template = guessed

    print("\nAPD on seen point-frames (all: median rescale; moving: none):")
    print(f"  {'template':<16} {'all':>7} {'moving':>8} {'moving, frames 12-23':>21}")
    for name, _, _ in variants:
        print(f"  {name:<16} {clip_mean(rows[name, 'all']):>7.4f} {clip_mean(rows[name, 'moving']):>8.4f} "
              f"{clip_mean(rows[name, 'moving, frames 12-23']):>21.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
