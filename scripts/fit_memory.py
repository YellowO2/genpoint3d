"""Why the object-pose fit fills the GPU in training: on a few training clips
as training draws them (`--points`, `--spawn`), one round of tracking for the
group vectors, then how many objects `objects` makes and how big, and the
peak memory of one `object_proposal` with its gradient, as the fit loss runs
it.

Run:  python scripts/fit_memory.py --ckpt CKPT --cache DIR --frames-cache FRAMES --points 1000 --spawn 500
"""

import argparse

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.score.Run import Run
from genpoint3d.train.tools.batch import fine_inputs, to_device, visual_mask
from genpoint3d.train.tools.regress import Sweeps, object_proposal, objects


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--frames-cache", required=True)
    p.add_argument("--points", type=int, default=1000)
    p.add_argument("--spawn", type=int, default=500)
    p.add_argument("--clips", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda")
    run = Run.load(args.ckpt, device)
    model, s = run.model, run.settings
    ds = run.dataset(args.cache, args.frames_cache, points=args.points, spawn=args.spawn)
    for k, batch in enumerate(DataLoader(ds, batch_size=1, shuffle=True)):
        if k == args.clips:
            break
        b = to_device(batch, device)
        T = b["traj"].shape[1]
        fine = fine_inputs(model, b, True)
        cond = dict(context=b["context"], visual_mask=visual_mask(b, None), id_card=b["id_card"],
                    patch_xyz=b["patch_xyz"], **fine)
        sweeps = Sweeps(model, b["anchor"], T, s.schedule, s.window, s.overlap, cond)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in sweeps:
                pass
        member = sweeps.seen & sweeps.known
        weight, which = objects(sweeps.groups.float(), member)
        sizes = torch.bincount(which[0][which[0] >= 0]).tolist()
        left_out = int((member[0] & (which[0] < 0)).sum())
        # The fit loss's call: one window's frames, the gradient on.
        groups = sweeps.groups.float().detach().requires_grad_(True)
        pos = model.lookup_pos(sweeps.whole.float(), b["anchor"].float())
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = object_proposal(pos, b["query_t"], groups, member, torch.sigmoid(sweeps.vis.float()), 0, min(16, T),
                              robust=model.fit_robust, fresh=model.fit_fresh)
        out[..., :3].sum().backward()
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
        print(f"clip {k}: {T} frames, {b['traj'].shape[2]} points, {int(member.sum())} with a group; "
              f"{len(sizes)} objects of 3+ points, sizes {sorted(sizes, reverse=True)[:12]}{' ...' if len(sizes) > 12 else ''}; "
              f"{left_out} points in groups of under 3; one fit with its gradient: {peak:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
