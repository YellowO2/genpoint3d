"""How good the rigid proposal is: for every point and frame, where moving
with its piece would put it, built as the model would get it in one more
round (from the final guesses, group vectors and visibility), against the truth,
beside the model's own final guess. Hidden and seen point-frames apart, all
points and the ones that move more than 4 px. A proposal far better than the
guess on hidden points, which the model does not follow, is a proposal the
model has not learned to use; one no better is one it is right to ignore.
Beside it, the proposal built from the true groups (the training labels'
pieces) instead of the model's: good with them and poor with the model's
says the groups are what is wrong; poor with both, the fit. Beside each, the
fit refit 3 times with mates far from it counting less (`robust`): good says
a few wrong mates were what dragged it. The hidden moving
point-frames are then split by how many points of the point's true piece are
seen in that frame: with none, the fit has only guesses to go on; with many,
it should be about as good as the seen points.

Run:  python scripts/rigid_check.py --ckpt outputs/RUN/best.pt --cache DIR --frames-cache FRAMES
"""

import argparse

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.score.Run import Run
from genpoint3d.train.tools.batch import fine_inputs, rigid_groups, to_device, to_metres, visual_mask
from genpoint3d.train.tools.metrics import clip_mean, motion_px, tapvid3d_metrics
from genpoint3d.train.tools.regress import Sweeps, rigid_proposal


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
    if not model.rigid_proposal:
        raise SystemExit("the run has no rigid proposal")
    ds = run.dataset(args.cache, args.frames_cache, points=args.points)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    amp = device.type == "cuda"
    print(f"{len(ds)} clips | step {run.step} of {args.ckpt}", flush=True)

    rows, backing = {}, {"hidden": [], "seen": []}
    bins = (("none", 0, 1), ("1-9", 1, 10), ("10+", 10, 1 << 30))
    counts = {label: 0 for label, _, _ in bins}
    for batch in loader:
        b = to_device(batch, device)
        traj, vis, q = b["traj"], b["visibility"].bool(), b["query_t"]
        T = traj.shape[1]
        cond = dict(context=b["context"], visual_mask=visual_mask(b, None), id_card=b["id_card"],
                    patch_xyz=b["patch_xyz"], **fine_inputs(model, b, amp))
        sweeps = Sweeps(model, b["anchor"], T, s.refine_iters, s.schedule, s.window, s.overlap, cond)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            for _ in sweeps:
                pass
        guess = sweeps.whole.float()
        pos = model.lookup_pos(guess, b["anchor"].float())
        prop = rigid_proposal(pos, q, b["anchor"].float(), sweeps.groups, sweeps.seen & sweeps.known,
                              torch.sigmoid(sweeps.vis.float()), 0, T)
        proposal = guess + prop[..., :3] / model.traj_scale
        robust = guess + rigid_proposal(pos, q, b["anchor"].float(), sweeps.groups, sweeps.seen & sweeps.known,
                                        torch.sigmoid(sweeps.vis.float()), 0, T, robust=3)[..., :3] / model.traj_scale
        # The true pieces as group vectors: one axis each, far apart.
        piece = rigid_groups(b).float().argmax(-1)                       # (B, N) the piece's first point
        truth = torch.nn.functional.one_hot(piece, piece.shape[1]).float() * 10
        true_prop = rigid_proposal(pos, q, b["anchor"].float(), truth, sweeps.seen,
                                   torch.sigmoid(sweeps.vis.float()), 0, T)
        true_groups = guess + true_prop[..., :3] / model.traj_scale
        true_robust = guess + rigid_proposal(pos, q, b["anchor"].float(), truth, sweeps.seen,
                                             torch.sigmoid(sweeps.vis.float()), 0, T, robust=3)[..., :3] / model.traj_scale
        gt = to_metres(traj, b)
        moving = (motion_px(gt, vis, b["intrinsics_256"], b["extrinsics"], q) > 4)[:, None]
        # Each point-frame's seen points on its true piece, itself left out.
        mates = rigid_groups(b) & ~torch.eye(traj.shape[2], dtype=torch.bool, device=traj.device)
        seen_mates = torch.einsum("btj,bij->bti", vis.float(), mates.float())             # (B, T, N)
        for label, lo, hi in bins:
            counts[label] += int((~vis & moving & (seen_mates >= lo) & (seen_mates < hi)).sum())
        for name, pts in (("guess", guess), ("proposal", proposal), ("robust", robust), ("true groups", true_groups),
                          ("true robust", true_robust)):
            m = to_metres(pts, b)
            for kind, mask in (("hidden", ~vis), ("seen", vis)):
                for which, sel in (("all", mask), ("moving", mask & moving)):
                    apd = tapvid3d_metrics(m, gt, sel, b["intrinsics_256"], b["extrinsics"], scaling="none",
                                           per_clip=True)["average_pts_within_thresh"].cpu()
                    rows.setdefault((kind, which, name), []).append(apd)
            for label, lo, hi in bins:
                sel = ~vis & moving & (seen_mates >= lo) & (seen_mates < hi)
                apd = tapvid3d_metrics(m, gt, sel, b["intrinsics_256"], b["extrinsics"], scaling="none",
                                       per_clip=True)["average_pts_within_thresh"].cpu()
                rows.setdefault(("mates", label, name), []).append(apd)
        for kind, mask in (("hidden", ~vis), ("seen", vis)):
            backing[kind].append(prop[..., 3][mask].float().cpu())

    names = ("guess", "proposal", "robust", "true groups", "true robust")
    print("\nAPD (no rescale), the model's final guess against where its piece would put it (robust: refit 3 times):")
    print(f"  {'point-frames':<16}" + "".join(f"{n:>13}" for n in names))
    for kind in ("hidden", "seen"):
        for which in ("all", "moving"):
            print(f"  {kind + ' ' + which:<16}" + "".join(f"{clip_mean(rows[kind, which, n]):>13.4f}" for n in names))
    total = max(sum(counts.values()), 1)
    print("\nHidden moving point-frames by how many points of their true piece are seen there:")
    print(f"  {'seen mates':<10} {'share':>6}" + "".join(f"{n:>13}" for n in names))
    for label, _, _ in bins:
        print(f"  {label:<10} {counts[label] / total:>6.3f}" + "".join(f"{clip_mean(rows['mates', label, n]):>13.4f}" for n in names))
    for kind in ("hidden", "seen"):
        v = torch.cat(backing[kind])
        print(f"  how much backs the proposal, {kind}: median {v.median():.3f}, "
              f"share above 0.5 {(v > 0.5).float().mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
