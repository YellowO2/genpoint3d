"""One clip's rigid fit, written out to look at (docs: the fit-view page): per
frame, each point's true place and visibility (the clip's own points), the
model's place and belief that it is seen, and where the fit carried from the
nearest seen frame puts it; for each of the clip's hidden points, the mates
its fit rests on there and their weights. Points added at test time are in
too, with no truth. Places in metres, the first frame's camera.

Run:  python scripts/fit_view.py --ckpt CKPT --cache DIR --frames-cache FRAMES --clip 0 --out view.json
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader, Subset

from genpoint3d.train.score.Run import Run
from genpoint3d.train.score.spawn_points import spawn_all
from genpoint3d.train.tools.batch import fine_inputs, rigid_groups, to_device, to_metres, visual_mask
from genpoint3d.train.tools.regress import Sweeps, _sq_dist, last_seen, rigid_proposal


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--frames-cache", required=True)
    p.add_argument("--clip", type=int, default=0)
    p.add_argument("--points", type=int, default=1000)
    p.add_argument("--spawn", type=int, default=150)
    p.add_argument("--mates", type=int, default=40, help="mates written per hidden point and frame, the heaviest")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda")
    run = Run.load(args.ckpt, device)
    model, s = run.model, run.settings
    ds = run.dataset(args.cache, args.frames_cache, points=args.points)
    b = to_device(next(iter(DataLoader(Subset(ds, [args.clip]), batch_size=1))), device)
    N, T = b["traj"].shape[2], b["traj"].shape[1]
    fine, view = fine_inputs(model, b, True), visual_mask(b, None)

    def track(bb):
        cond = dict(context=bb["context"], visual_mask=view, id_card=bb["id_card"], patch_xyz=bb["patch_xyz"],
                    **{**fine, "query_t": bb["query_t"]})
        sweeps = Sweeps(model, bb["anchor"], T, s.schedule, s.window, s.overlap, cond)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in sweeps:
                pass
        return sweeps

    sweeps = track(b)
    every = spawn_all(model, b, fine, view, args.spawn, 0.05, lambda bb: (lambda sw: (sw.whole, sw.vis))(track(bb))) if args.spawn else b
    sweeps = track(every)
    anchor, eq = every["anchor"].float(), every["query_t"]
    guess = sweeps.whole.float()
    pos = model.lookup_pos(guess, anchor)
    belief, member = torch.sigmoid(sweeps.vis.float()), sweeps.seen & sweeps.known
    ref = last_seen(belief, eq)
    fit = guess + rigid_proposal(pos, eq, sweeps.groups, member, belief, 0, T, robust=3, ref=ref)[..., :3] / model.traj_scale
    metres = lambda x: to_metres(x, every)[0]
    # The fit's weights before its refits, as `_rigid_fit` builds them.
    a = 2 * torch.sigmoid(-_sq_dist(sweeps.groups.float())) * (member[:, :, None] & member[:, None, :])
    a = (a * (a > 0.5)).masked_fill(torch.eye(a.shape[1], dtype=torch.bool, device=device), 0.0)
    w = a[:, None] * belief[:, :, None, :] * belief[torch.arange(1, device=device)[:, None, None], ref]   # (1, T, M, M)
    vis = b["visibility"][0].bool()
    true = to_metres(b["traj"].float(), b)[0]
    piece = rigid_groups(b)[0].float().argmax(-1)                     # each point's piece: its first point
    hidden = []
    for t in range(T):
        for i in torch.nonzero(~vis[t])[:, 0].tolist():
            top = w[0, t, i].topk(args.mates)
            keep = top.values > 1e-3
            hidden.append({"t": t, "i": i, "ref": int(ref[0, t, i]),
                           "mates": top.indices[keep].tolist(), "w": [round(v, 3) for v in top.values[keep].tolist()]})
    r = lambda x: [[round(v, 4) for v in row] for row in x.tolist()]
    out = {"clip": ds.clips[args.clip].stem, "T": T, "given": N, "total": pos.shape[2],
           "true": [r(true[t]) for t in range(T)], "vis": vis.int().tolist(),
           "pred": [r(metres(guess)[t]) for t in range(T)], "belief": [[round(v, 2) for v in row] for row in belief[0].tolist()],
           "fit": [r(metres(fit)[t]) for t in range(T)], "piece": piece.tolist(),
           "query_t": eq[0].tolist(), "hidden": hidden}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"{out['clip']}: {N} given, {out['total'] - N} added, {len(hidden)} hidden point-frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
