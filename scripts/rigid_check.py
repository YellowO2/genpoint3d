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
a few wrong mates were what dragged it. Then, with the true pieces, the true
visibility deciding which mates count (`true seen`), and the true places of
the mates too (`all true`): what each of the fit's inputs costs it. The hidden moving
point-frames are then split by how many points of the point's true piece are
seen in that frame: with none, the fit has only guesses to go on; with many,
it should be about as good as the seen points. With `--spawn`, points are
first added on new surface as at test time; they are mates but not scored,
and have no truth, so the true-group columns are left out.

Run:  python scripts/rigid_check.py --ckpt outputs/RUN/best.pt --cache DIR --frames-cache FRAMES
"""

import argparse

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.score.Run import Run
from genpoint3d.train.score.spawn_points import spawned
from genpoint3d.train.tools.batch import fine_inputs, rigid_groups, to_device, to_metres, visual_mask
from genpoint3d.train.tools.metrics import clip_mean, motion_px, tapvid3d_metrics
from genpoint3d.train.tools.regress import Sweeps, last_seen, rigid_motion, rigid_proposal


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--frames-cache", required=True)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--spawn", type=int, default=0, help="at most this many points added on new surface first")
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
    names = ("guess", "proposal", "robust", "last seen") + (() if args.spawn else ("true groups", "true robust", "true seen", "all true"))
    for batch in loader:
        b = to_device(batch, device)
        vis, q, N = b["visibility"].bool(), b["query_t"], b["traj"].shape[2]
        T = b["traj"].shape[1]
        fine, view = fine_inputs(model, b, amp), visual_mask(b, None)

        def track(bb):
            cond = dict(context=bb["context"], visual_mask=view, id_card=bb["id_card"], patch_xyz=bb["patch_xyz"],
                        **{**fine, "query_t": bb["query_t"]})
            sweeps = Sweeps(model, bb["anchor"], T, s.schedule, s.window, s.overlap, cond)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in sweeps:
                    pass
            return sweeps

        sweeps, every = track(b), b
        if args.spawn:
            every = spawned(model, b, fine, sweeps.whole, sweeps.vis, view, args.spawn, 0.05)
            sweeps = track(every)
        # Fitted over every point, the added ones too; only the clip's own are scored.
        anchor, eq = every["anchor"].float(), every["query_t"]
        guess = sweeps.whole.float()
        pos = model.lookup_pos(guess, anchor)
        belief, member = torch.sigmoid(sweeps.vis.float()), sweeps.seen & sweeps.known
        prop = rigid_proposal(pos, eq, sweeps.groups, member, belief, 0, T)
        fits = {"guess": guess, "proposal": guess + prop[..., :3] / model.traj_scale,
                "robust": guess + rigid_proposal(pos, eq, sweeps.groups, member, belief, 0, T,
                                                 robust=3)[..., :3] / model.traj_scale,
                # Each point carried from the nearest frame it is believed seen in, not its query frame.
                "last seen": guess + rigid_proposal(pos, eq, sweeps.groups, member, belief, 0, T, robust=3,
                                                    ref=last_seen(belief, eq))[..., :3] / model.traj_scale}
        if not args.spawn:
            # The true pieces as group vectors: one axis each, far apart.
            piece = rigid_groups(b).float().argmax(-1)                       # (B, N) the piece's first point
            truth = torch.nn.functional.one_hot(piece, piece.shape[1]).float() * 10
            fits["true groups"] = guess + rigid_proposal(pos, q, truth, sweeps.seen, belief, 0, T)[..., :3] / model.traj_scale
            fits["true robust"] = guess + rigid_proposal(pos, q, truth, sweeps.seen, belief, 0, T,
                                                         robust=3)[..., :3] / model.traj_scale
            # Which mates count, then where they are too, from the truth: what the fit's inputs cost it.
            seen_true = b["visibility"].float()
            fits["true seen"] = guess + rigid_proposal(pos, q, truth, sweeps.seen, seen_true, 0, T,
                                                       robust=3)[..., :3] / model.traj_scale
            fits["all true"] = b["traj"].float() + rigid_proposal(model.lookup_pos(b["traj"].float(), anchor), q,
                                                                  truth, sweeps.seen, seen_true, 0, T, robust=3)[..., :3] / model.traj_scale
        fits = {k: v[:, :, :N] for k, v in fits.items()}
        traj = b["traj"]
        gt = to_metres(traj, b)
        moving = (motion_px(gt, vis, b["intrinsics_256"], b["extrinsics"], q) > 4)[:, None]
        # Each point-frame's seen points on its true piece, itself left out.
        mates = rigid_groups(b) & ~torch.eye(traj.shape[2], dtype=torch.bool, device=traj.device)
        seen_mates = torch.einsum("btj,bij->bti", vis.float(), mates.float())             # (B, T, N)
        for label, lo, hi in bins:
            counts[label] += int((~vis & moving & (seen_mates >= lo) & (seen_mates < hi)).sum())
        for name, pts in fits.items():
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
            backing[kind].append(prop[:, :, :N, 3][mask].float().cpu())
        if not args.spawn:
            # For the hidden moving point-frames with 10+ seen mates: how good the seen mates
            # the fit rests on are in those frames, and its turn from the predicted places
            # against the turn from the true ones (both with the true pieces and visibility).
            used = ~vis & moving & (seen_mates >= 10)                                       # (B, T, N)
            on_used = vis & (torch.einsum("bti,bij->btj", used.float(), mates.float()) > 0)
            apd = tapvid3d_metrics(to_metres(guess[:, :, :N], b), gt, on_used, b["intrinsics_256"], b["extrinsics"],
                                   scaling="none", per_clip=True)["average_pts_within_thresh"].cpu()
            rows.setdefault(("used mates", "apd"), []).append(apd)
            turn = lambda places: rigid_motion(places, q, truth, sweeps.seen, seen_true, robust=3)[0][:, :, :N]
            R_pred, R_true = turn(pos), turn(model.lookup_pos(b["traj"].float(), anchor))
            angle = lambda R: torch.acos(((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)).rad2deg()
            if used.any():
                rows.setdefault(("turn", "true"), []).append(angle(R_true)[used].cpu())
                rows.setdefault(("turn", "pred"), []).append(angle(R_pred)[used].cpu())
                rows.setdefault(("turn", "apart"), []).append(angle(R_pred.transpose(-1, -2) @ R_true)[used].cpu())

    print("\nAPD (no rescale), the model's final guess against where its piece would put it (robust: refit 3 times):")
    print(f"  {'point-frames':<16}" + "".join(f"{n:>14}" for n in names))
    for kind in ("hidden", "seen"):
        for which in ("all", "moving"):
            print(f"  {kind + ' ' + which:<16}" + "".join(f"{clip_mean(rows[kind, which, n]):>14.4f}" for n in names))
    total = max(sum(counts.values()), 1)
    print("\nHidden moving point-frames by how many points of their true piece are seen there:")
    print(f"  {'seen mates':<10} {'share':>6}" + "".join(f"{n:>14}" for n in names))
    for label, _, _ in bins:
        print(f"  {label:<10} {counts[label] / total:>6.3f}" + "".join(f"{clip_mean(rows['mates', label, n]):>14.4f}" for n in names))
    if ("turn", "true") in rows:
        med = lambda k: torch.cat(rows["turn", k]).median().item()
        print(f"\nHidden moving point-frames with 10+ seen mates: their seen mates' APD there "
              f"{clip_mean(rows['used mates', 'apd']):.4f} (all seen: {clip_mean(rows['seen', 'all', 'guess']):.4f}); "
              f"the piece's turn since the query frame, median degrees: true {med('true'):.1f}, "
              f"from the predicted places {med('pred'):.1f}, the two {med('apart'):.1f} apart")
    for kind in ("hidden", "seen"):
        v = torch.cat(backing[kind])
        print(f"  how much backs the proposal, {kind}: median {v.median():.3f}, "
              f"share above 0.5 {(v > 0.5).float().mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
