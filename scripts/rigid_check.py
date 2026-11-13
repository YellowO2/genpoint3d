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
from genpoint3d.train.score.spawn_points import spawn_all
from genpoint3d.train.tools.batch import fine_inputs, rigid_groups, to_device, to_metres, visual_mask
from genpoint3d.train.tools.camera import project, unproject
from genpoint3d.train.tools.metrics import clip_mean, motion_px, tapvid3d_metrics
from genpoint3d.train.tools.regress import Sweeps, object_places, rigid_motion, rigid_proposal


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--frames-cache", required=True)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--spawn", type=int, default=0, help="points each clip is filled up to first where surface has none")
    p.add_argument("--query-before", type=int, default=0, help="only the tracks queried before this frame (0: all)")
    p.add_argument("--fresh", type=float, default=0.0, help="object pose: mates weighted 1 / (1 + turn / fresh)^2, degrees (0: alike)")
    p.add_argument("--train-spawn", type=int, default=0,
                   help="up to this many of the clip's tracks started where they come into view, as training draws them")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run.load(args.ckpt, device)
    model, s = run.model, run.settings
    if not model.rigid_proposal:
        raise SystemExit("the run has no rigid proposal")
    ds = run.dataset(args.cache, args.frames_cache, points=args.points, query_before=args.query_before,
                     spawn=args.train_spawn)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    amp = device.type == "cuda"
    print(f"{len(ds)} clips | step {run.step} of {args.ckpt}", flush=True)

    rows, backing = {}, {"hidden": [], "seen": []}
    bins = (("none", 0, 1), ("1-9", 1, 10), ("10+", 10, 1 << 30))
    counts = {label: 0 for label, _, _ in bins}
    names = ("guess", "robust", "object pose", "object true", "true depth", "true image", "true both", "errors dealt") + (() if args.spawn else ("true groups", "true robust", "true seen", "all true"))
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
            every = spawn_all(model, b, fine, view, args.spawn, 0.05, lambda bb: (lambda sw: (sw.whole, sw.vis))(track(bb)))
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
                # Each object kept as one rigid shape, its pose fit every frame from its seen points.
                "object pose": (object_places(pos, eq, sweeps.groups, member, belief, robust=3, fresh=args.fresh) - anchor[:, None])
                               / model.traj_scale}
        # The same with the clip's own points at their true depth (their predicted place in the
        # image kept), at their true place in the image (their predicted depth kept), and both.
        proj = fine["fine_proj"].float()
        true_pos = pos.clone()
        true_pos[:, :, :N] = model.lookup_pos(b["traj"].float(), anchor[:, :N])
        (uv_p, z_p), (uv_t, z_t) = project(pos, proj), project(true_pos, proj)
        # The seen points' own errors, the same sizes and directions, dealt out among the
        # points seen in that frame at random: their size kept, any pattern across points
        # broken. The rest keep their predicted places.
        err = pos - true_pos
        shuffled = pos.clone()
        for bi in range(err.shape[0]):
            for t in range(T):
                on = torch.nonzero(vis[bi, t])[:, 0]
                shuffled[bi, t, on] = true_pos[bi, t, on] + err[bi, t, on[torch.randperm(len(on), device=on.device)]]
        for name, mixed in (("true depth", unproject(uv_p, z_t, proj)), ("true image", unproject(uv_t, z_p, proj)),
                            ("true both", true_pos), ("errors dealt", shuffled)):
            # The fit's offset is from the places it was given, here not the guess.
            fits[name] = (mixed - anchor[:, None]) / model.traj_scale + rigid_proposal(
                mixed, eq, sweeps.groups, member, belief, 0, T, robust=3)[..., :3] / model.traj_scale
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
        # The same from true places (the clip's own; added points as predicted).
        fits["object true"] = (object_places(true_pos, eq, sweeps.groups, member, belief, robust=3, fresh=args.fresh) - anchor[:, None]) / model.traj_scale
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
        # A fifth of the truly seen points kept out of the fits, then placed like hidden ones: placed
        # as well as the points the fit rests on, or as badly as the hidden ones?
        held = (torch.rand(vis.shape, device=vis.device) < 0.2) & vis
        out_belief = belief.clone()
        out_belief[:, :, :N] = belief[:, :, :N] * ~held
        e_held = (to_metres((object_places(pos, eq, sweeps.groups, member, out_belief, robust=3, fresh=args.fresh)
                             - anchor[:, None])[:, :, :N] / model.traj_scale, b) - gt).norm(dim=-1)        # (B, T, N)
        per_frame = lambda m: torch.stack([e_held[:, t][m[:, t]].median() if m[:, t].any() else e_held.new_tensor(float("nan"))
                                           for t in range(T)])
        rows.setdefault(("held", "frames"), []).append(torch.arange(T))
        for kind, m in (("in the fit", vis & ~held & moving), ("kept out", held & moving), ("hidden", ~vis & moving)):
            rows.setdefault(("held", kind), []).append(per_frame(m).cpu())
        # The model's own error on seen moving points by how far each has turned since its query
        # frame: do points that have turned little (fresh) stay accurate while old ones drift?
        guess_err = (to_metres(guess[:, :, :N], b) - gt).norm(dim=-1)                          # (B, T, N)
        _, _, true_turn = (lambda R: (None, None, torch.acos(((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2)
                                                             .clamp(-1, 1)).rad2deg()))(
            rigid_motion(model.lookup_pos(b["traj"].float(), anchor[:, :N]), q, truth, sweeps.seen[:, :N],
                         vis.float(), robust=3)[0]) if not args.spawn else (None, None, None)
        if true_turn is not None:
            m = vis & moving
            rows.setdefault(("by turn", "pairs"), []).append(torch.stack([true_turn[m], 100 * guess_err[m]], -1).cpu())
        # The object-pose placement's error by how far a point is from the centre of its true
        # piece's points seen in that frame: a turn measured from those carries farther points worse.
        seen_w = vis.float()[:, :, None, :] * mates.float()[:, None]                          # (B, T, N, N)
        centre = (seen_w[..., None] * gt[:, :, None]).sum(3) / seen_w.sum(3, keepdim=True).clamp_min(1)
        far_from = (gt - centre).norm(dim=-1)                                                 # (B, T, N) metres
        err = (to_metres(fits["object pose"], b) - gt).norm(dim=-1)
        for kind, mask in (("hidden", ~vis & moving & (seen_mates >= 3)), ("seen", vis & moving & (seen_mates >= 3))):
            rows.setdefault(("by distance", kind), []).append(torch.stack([far_from[mask], err[mask]], -1).cpu())
        # Each point that hides, lined up on the frame it hides in: its object-pose error the frames
        # before (seen) and after (hidden), to see a jump at the moment from a slow drift.
        hides = vis[:, :-1] & ~vis[:, 1:]                                                     # (B, T-1, N) hides at t+1
        for bi, t1, i in torch.nonzero(hides).tolist():
            t0 = t1 + 1
            for k in range(-3, 9):
                if 0 <= t0 + k < T and (k < 0 or not vis[bi, t0:t0 + k + 1, i].any()):
                    rows.setdefault(("hiding", k), []).append(err[bi, t0 + k, i].item())
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
    if ("by turn", "pairs") in rows:
        v = torch.cat(rows["by turn", "pairs"])
        print("\nThe model's own error (cm, median) on seen moving points by how far each has turned since its query frame:")
        for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 90), (90, 180)):
            sel = (v[:, 0] >= lo) & (v[:, 0] < hi)
            if sel.any():
                print(f"  {lo:>3}-{hi:<3} deg: {v[sel, 1].median():.2f} cm ({int(sel.sum())} point-frames)")
    ks = sorted(key[1] for key in rows if key[0] == "hiding")
    if ks:
        print("\nThe object-pose placement's error (cm, median) around the frame a point hides in (0), seen before it:")
        print("  frames from hiding " + "".join(f"{k:>6}" for k in ks))
        print("  error              " + "".join(f"{100 * torch.tensor(rows['hiding', k]).median():>6.1f}" for k in ks))
        print("  point-frames       " + "".join(f"{len(rows['hiding', k]):>6}" for k in ks))
    print("\nThe object-pose placement with a fifth of the seen points kept out of its fits, error (cm, median) by frame:")
    print("  frame     " + "".join(f"{t:>6}" for t in range(len(rows["held", "frames"][0]))))
    for kind in ("in the fit", "kept out", "hidden"):
        e = torch.stack([v for v in rows["held", kind]]).nanmedian(0).values
        print(f"  {kind:<10}" + "".join(f"{x:>6.1f}" if x == x else f"{'-':>6}" for x in (100 * e).tolist()))
    print("\nThe object-pose placement's error (cm, median) by distance from the centre of the piece's seen points:")
    for kind in ("seen", "hidden"):
        v = torch.cat(rows["by distance", kind])
        if not len(v):
            continue
        edges = torch.quantile(v[:, 0], torch.linspace(0, 1, 5))
        parts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v[:, 0] >= lo) & (v[:, 0] <= hi)
            parts.append(f"{lo:.2f}-{hi:.2f} m: {100 * v[m, 1].median():.1f}")
        print(f"  {kind:<7}" + "   ".join(parts))
    for kind in ("hidden", "seen"):
        v = torch.cat(backing[kind])
        print(f"  how much backs the proposal, {kind}: median {v.median():.3f}, "
              f"share above 0.5 {(v > 0.5).float().mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
