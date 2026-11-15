"""Is the fit the model is handed any good? `Sweeps` hands each round the
object fit of the guess that round starts from. For every round of every
window, on the clip's own moving point-frames that are truly hidden (and, for
reference, seen): the error of the guess the round starts from, the error of
the fit as handed (guess + offset), and how often the fit is the nearer of
the two. If the handed fit is often worse than the guess, the model is right
not to follow it. Runs as the design does: filled to `--fill` points.

    python scripts/handed_fit.py --ckpt outputs/RUN/best.pt --cache DIR --frames-cache DIR --points 256 --fill 1000
"""
import argparse

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.score.Run import Run
from genpoint3d.train.score.spawn_points import spawn_points
from genpoint3d.train.tools.batch import fine_inputs, to_device, to_metres, visual_mask
from genpoint3d.train.tools.metrics import MOVING_PX, motion_px
from genpoint3d.train.tools.regress import Sweeps


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--frames-cache", required=True)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--fill", type=int, default=1000)
    p.add_argument("--clips", type=int, default=40)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run.load(args.ckpt, device)
    model, s = run.model, run.settings
    loader = DataLoader(run.dataset(args.cache, args.frames_cache, points=args.points), batch_size=1, shuffle=False, num_workers=4)
    amp = device.type == "cuda"

    handed = []
    plain = Sweeps.proposal

    def recorded(self, g, groups, start, end):
        out = plain(self, g, groups, start, end)
        handed.append((start, end, g.float(), out.float()))
        return out

    rows = {}
    for n_clip, batch in enumerate(loader):
        if n_clip >= args.clips:
            break
        b = to_device(batch, device)
        vis, N, T = b["visibility"].bool(), b["traj"].shape[2], b["traj"].shape[1]
        fine, view = fine_inputs(model, b, amp), visual_mask(b, None)

        def track(bb):
            cond = dict(context=bb["context"], visual_mask=view, id_card=bb["id_card"], patch_xyz=bb["patch_xyz"],
                        **{**fine, "query_t": bb["query_t"]})
            sweeps = Sweeps(model, bb["anchor"], T, s.schedule, s.window, s.overlap, cond)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in sweeps:
                    pass
            return sweeps

        every = spawn_points(model, b, fine, view, args.fill, run.spawn_gap, run.spawn_share,
                             lambda bb: (lambda sw: (sw.whole, sw.vis))(track(bb))) if args.fill else b
        handed.clear()
        Sweeps.proposal = recorded
        final = track(every).whole.float()
        Sweeps.proposal = plain
        gt = to_metres(b["traj"].float(), b)
        moving = (motion_px(gt, vis, b["intrinsics_256"], b["extrinsics"], b["query_t"]) > MOVING_PX)[:, None]
        for k, (start, end, g, fit) in enumerate(handed):
            e_g = (to_metres(g[:, :, :N], b) - gt[:, start:end]).norm(dim=-1)
            e_f = (to_metres(g[:, :, :N] + fit[:, :, :N, :3] / model.traj_scale, b) - gt[:, start:end]).norm(dim=-1)
            e_end = (to_metres(final[:, start:end, :N], b) - gt[:, start:end]).norm(dim=-1)
            backed = fit[:, :, :N, 3] > 0
            for kind, m in (("hidden", ~vis[:, start:end] & moving), ("seen", vis[:, start:end] & moving)):
                r = rows.setdefault((k, start, end, kind), dict(g=[], f=[], end=[], near=[], backed=[]))
                r["g"].append(e_g[m].cpu()); r["f"].append(e_f[m].cpu()); r["end"].append(e_end[m].cpu())
                r["near"].append((e_f < e_g)[m & backed].float().cpu()); r["backed"].append(backed[m].float().cpu())

    print(f"{args.ckpt} step {run.step} | {args.cache} | moving point-frames, error in cm (median)")
    print(f"  {'round':>5} {'frames':>7} {'kind':<7}{'count':>9}{'has a fit':>10}{'start guess':>12}{'fit handed':>11}{'fit nearer':>11}{'final':>8}")
    med = lambda xs: 100 * torch.cat(xs).median().item() if sum(len(x) for x in xs) else float("nan")
    mean = lambda xs: torch.cat(xs).mean().item() if sum(len(x) for x in xs) else float("nan")
    for (k, start, end, kind), r in sorted(rows.items()):
        print(f"  {k:>5} {f'{start}-{end - 1}':>7} {kind:<7}{sum(len(x) for x in r['g']):>9}{mean(r['backed']):>10.2f}"
              f"{med(r['g']):>12.1f}{med(r['f']):>11.1f}{mean(r['near']):>11.2f}{med(r['end']):>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
