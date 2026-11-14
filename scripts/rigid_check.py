"""How good the object's pose is at placing points (`object_places`, the
model's fit input): for every point and frame, where its object's rigid shape
and pose put it, built from the model's final guesses, group vectors and
visibility as the model would get it in one more round, against the truth,
beside the model's own final guess. Hidden and seen point-frames apart, all
points and the ones that move more than 4 px.

Beside it, the same with some of its inputs from the truth: the clip's points
at their true depth, at their true place in the image, and both; the seen
points' errors dealt out among them at random (their size kept, any pattern
across points broken); and, without `--fill`, the true pieces as groups, the
true visibility, and all true. The hidden moving point-frames are split by
how many points of their true piece are seen there.

Then, for the object's pose: its error by frame with a fifth of the seen
points kept out of its fits (kept out as good as in it says leaving the fit
costs nothing), its error lined up on the frame each point hides in, and by
how far a point is from the centre of its piece's seen points.

With `--fill`, points are first added on new surface as at test time; they
are mates but not scored and have no truth. `--spawn` draws the clip's
own tracks as training does instead, so every point has its truth. With
`--cutoff C`, frames from C on have no image (a forecast) and are scored on
their own: there every pose is carried on from the fitted frames.

Run:  python scripts/rigid_check.py --ckpt outputs/RUN/best.pt --cache DIR --frames-cache FRAMES
"""

import argparse

import torch
from torch.utils.data import DataLoader

from genpoint3d.tools.traj_scale import GRAVITY, KUBRIC_FPS
from genpoint3d.train.score.Run import Run
from genpoint3d.train.score.spawn_points import spawn_points
from genpoint3d.train.tools.batch import fine_inputs, rigid_groups, to_device, to_metres, visual_mask
from genpoint3d.train.tools.camera import project, unproject
from genpoint3d.train.tools.metrics import MOVING_PX, clip_mean, motion_px, tapvid3d_metrics
from genpoint3d.train.tools.kabsch import MIN_POSE_POINTS
from genpoint3d.train.tools.regress import Sweeps, object_places


# Metres an object's true centre may part from a thrown body's path and still count as in free flight.
FLIGHT_TOL = 0.05


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--frames-cache", required=True)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--fill", type=int, default=0, help="points each clip is filled up to first where surface has none")
    p.add_argument("--spawn", type=int, default=0,
                   help="up to this many of the clip's tracks started where they come into view, as training draws them")
    p.add_argument("--query-before", type=int, default=0, help="only the tracks queried before this frame (0: all)")
    p.add_argument("--robust", type=int, default=None, help="the pose's refits (default: the run's)")
    p.add_argument("--fresh", type=float, default=None, help="the pose's freshness weighting, degrees (default: the run's)")
    p.add_argument("--cutoff", type=int, default=0, help="frames from this one on without their image: a forecast (0: none)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run.load(args.ckpt, device)
    model, s = run.model, run.settings
    if not model.rigid_proposal:
        raise SystemExit("the run has no fit input")
    robust = model.fit_robust if args.robust is None else args.robust
    fresh = model.fit_fresh if args.fresh is None else args.fresh
    ds = run.dataset(args.cache, args.frames_cache, points=args.points, query_before=args.query_before,
                     spawn=args.spawn)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    amp = device.type == "cuda"
    print(f"{len(ds)} clips | step {run.step} of {args.ckpt} | refits {robust}, freshness {fresh}", flush=True)

    rows = {}
    bins = (("none", 0, 1), ("1-9", 1, 10), ("10+", 10, 1 << 30))
    counts = {label: 0 for label, _, _ in bins}
    names = ("guess", "object", "true depth", "true image", "true both", "seen true", "errors dealt")
    # How the fit places a point-frame: not at all (the point is in no object, or never joined its
    # shape), by a pose fitted in that frame, or by a pose carried on from other frames.
    ways = ("no fit", "fitted pose", "carried pose")
    way_counts = {w: 0 for w in ways}
    flight_counts = {}
    names += () if args.fill else ("true groups", "true seen", "all true", "true centre")
    for batch in loader:
        b = to_device(batch, device)
        vis, q, N = b["visibility"].bool(), b["query_t"], b["traj"].shape[2]
        T = b["traj"].shape[1]
        cut = torch.full((vis.shape[0],), args.cutoff, device=device) if args.cutoff else None
        fine, view = fine_inputs(model, b, amp), visual_mask(b, cut)

        def track(bb):
            cond = dict(context=bb["context"], visual_mask=view, id_card=bb["id_card"], patch_xyz=bb["patch_xyz"],
                        **{**fine, "query_t": bb["query_t"]})
            sweeps = Sweeps(model, bb["anchor"], T, s.schedule, s.window, s.overlap, cond)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in sweeps:
                    pass
            return sweeps

        sweeps, every = track(b), b
        if args.fill:
            every = spawn_points(model, b, fine, view, args.fill, run.spawn_gap, run.spawn_share, lambda bb: (lambda sw: (sw.whole, sw.vis))(track(bb)))
            sweeps = track(every)
        # Fitted over every point, the added ones too; only the clip's own are scored.
        anchor, eq = every["anchor"].float(), every["query_t"]
        guess = sweeps.whole.float()
        pos = model.lookup_pos(guess, anchor)
        belief, member = sweeps.belief(), sweeps.seen & sweeps.known
        to_guess = lambda places: (places - anchor[:, None]) / model.traj_scale
        tips, fall = sweeps.tips(guess), sweeps.fall()
        place = lambda at, groups=sweeps.groups, seen=belief: to_guess(object_places(at, eq, groups, member, seen,
                                                                                     robust, fresh, tips, fall)[0])
        fits = {"guess": guess, "object": place(pos)}
        # The clip's own points at their true depth (their predicted place in the image kept), at
        # their true place in the image (their predicted depth kept), and both; and the seen points'
        # own errors dealt out among the points seen in that frame at random.
        proj = fine["fine_proj"].float()
        true_pos = pos.clone()
        true_pos[:, :, :N] = model.lookup_pos(b["traj"].float(), anchor[:, :N])
        (uv_p, z_p), (uv_t, z_t) = project(pos, proj), project(true_pos, proj)
        err = pos - true_pos
        dealt = pos.clone()
        for bi in range(err.shape[0]):
            for t in range(T):
                on = torch.nonzero(vis[bi, t])[:, 0]
                dealt[bi, t, on] = true_pos[bi, t, on] + err[bi, t, on[torch.randperm(len(on), device=on.device)]]
        # The truth only where a point is truly seen: hidden point-frames keep the model's guess,
        # so nothing unplaced is handed its truth (as "true both" does).
        seen_all = torch.zeros(pos.shape[:3], dtype=torch.bool, device=device)
        seen_all[:, :, :N] = vis
        seen_true = torch.where(seen_all[..., None], true_pos, pos)
        raw, support = object_places(pos, eq, sweeps.groups, member, belief, robust, fresh, tips, fall)
        way = torch.where((raw == pos).all(-1), 0, torch.where(support >= MIN_POSE_POINTS, 1, 2))[:, :, :N]
        for name, at in (("true depth", unproject(uv_p, z_t, proj)), ("true image", unproject(uv_t, z_p, proj)),
                         ("true both", true_pos), ("seen true", seen_true), ("errors dealt", dealt)):
            fits[name] = place(at)
        if not args.fill:
            # The true pieces as group vectors (one axis each, far apart), then the true visibility, then all true.
            piece = rigid_groups(b).float().argmax(-1)                       # (B, N) the piece's first point
            truth = torch.nn.functional.one_hot(piece, piece.shape[1]).float() * 10
            # True visibility only in frames with their image: nothing is seen in a forecast frame.
            shown = b["visibility"].float() * (torch.arange(T, device=device) < (args.cutoff or T))[None, :, None]
            fits["true groups"] = place(pos, truth)
            fits["true seen"] = place(pos, truth, shown)
            fits["all true"] = place(true_pos, truth, shown)
            if "centre" in b:
                # And the arrows true too: each point's tip at its object's true centre of mass.
                known = b["centre_known"][:, None, :, None]
                true_tips = torch.where(known, model.lookup_pos(b["centre"].float(), anchor), true_pos if tips is None else tips)
                fits["true centre"] = to_guess(object_places(true_pos, eq, truth, member, shown, robust, fresh, true_tips, fall)[0])
        fits = {k: v[:, :, :N] for k, v in fits.items()}
        gt = to_metres(b["traj"], b)
        moving = (motion_px(gt, vis, b["intrinsics_256"], b["extrinsics"], q) > MOVING_PX)[:, None]
        # Each point-frame's seen points on its true piece, itself left out.
        mates = rigid_groups(b) & ~torch.eye(N, dtype=torch.bool, device=device)
        seen_mates = torch.einsum("btj,bij->bti", vis.float(), mates.float())             # (B, T, N)
        for label, lo, hi in bins:
            counts[label] += int((~vis & moving & (seen_mates >= lo) & (seen_mates < hi)).sum())
        # With a cutoff, where the clip says each point's object's centre of mass: is the object
        # still in free flight? Its true centre carried on from the last two seen frames as a
        # thrown body goes (speed plus gravity), against where it truly is: free until they part
        # by FLIGHT_TOL, in contact (the floor, another object, or at rest on one) from then on.
        flight = None
        if args.cutoff >= 2 and "centre" in b and b["centre_known"].any():
            C = args.cutoff
            c = to_metres(b["centre"], b)                                                      # (B, T, N, 3)
            g = GRAVITY / KUBRIC_FPS ** 2 * b["down"].float()[:, None, None]                   # (B, 1, 1, 3) per frame^2
            k = (torch.arange(T, device=device) - (C - 1)).float()[None, :, None, None]
            speed = c[:, C - 1:C] - c[:, C - 2:C - 1] + 0.5 * g                                # at C-1
            thrown = c[:, C - 1:C] + speed * k + 0.5 * g * k * k
            apart = ((thrown - c).norm(dim=-1) > FLIGHT_TOL) & (k[..., 0] > 0)
            flight = ~(apart.cummax(1).values) & b["centre_known"][:, None]                    # (B, T, N)
            known = b["centre_known"][:, None].expand_as(vis)
        apd = lambda m, sel: tapvid3d_metrics(m, gt, sel, b["intrinsics_256"], b["extrinsics"], scaling="none",
                                              per_clip=True)["average_pts_within_thresh"].cpu()
        for name, pts in fits.items():
            m = to_metres(pts, b)
            ahead = ((torch.arange(T, device=device) >= args.cutoff)[None, :, None] & torch.ones_like(vis) if args.cutoff
                     else torch.zeros_like(vis))
            steps = torch.arange(T, device=device)[None, :, None] - args.cutoff                # frames into the forecast
            soon, later = ahead & (steps < 4), ahead & (steps >= 4) & (steps < 12)
            for kind, mask in (("hidden", ~vis & ~ahead), ("seen", vis & ~ahead), ("forecast", ahead),
                               ("fc 1-4", soon), ("fc 5-12", later), ("fc 13+", ahead & (steps >= 12))):
                for which, sel in (("all", mask), ("moving", mask & moving), ("still", mask & ~moving)):
                    rows.setdefault((kind, which, name), []).append(apd(m, sel))
            for label, lo, hi in bins:
                rows.setdefault(("mates", label, name), []).append(
                    apd(m, ~vis & moving & (seen_mates >= lo) & (seen_mates < hi)))
            for k, w in enumerate(ways):
                rows.setdefault(("way", w, name), []).append(apd(m, ~vis & moving & ~ahead & (way == k)))
            if flight is not None:
                for label, sel in (("free flight", flight), ("in contact", known & ~flight)):
                    for part, when in (("1-4", soon), ("5-12", later), ("13+", ahead & (steps >= 12))):
                        rows.setdefault(("flight", label, part, name), []).append(apd(m, when & moving & sel))
                        if name == "guess":
                            flight_counts[label, part] = flight_counts.get((label, part), 0) + int((when & moving & sel).sum())
        if flight is not None:
            # What a thrown body's own rules could reach from the truth, in the forecast frames: the
            # object as it truly was at the last seen frame, moved with its true centre without
            # turning; along the thrown path without turning; along the thrown path, turned as it truly turns.
            last = gt[:, C - 1:C]
            for name, m in (("true centre, no turn", last + c - c[:, C - 1:C]), ("thrown, no turn", last + thrown - c[:, C - 1:C]),
                            ("thrown, true turn", gt + thrown - c)):
                for label, sel in (("free flight", flight), ("in contact", known & ~flight)):
                    for part, when in (("1-4", soon), ("5-12", later)):
                        rows.setdefault(("flight ref", label, part, name), []).append(apd(m, when & moving & sel))
        for k, w in enumerate(ways):
            way_counts[w] += int((~vis & moving & ~ahead & (way == k)).sum())
        err = (to_metres(fits["object"], b) - gt).norm(dim=-1)                                   # (B, T, N) metres
        # A fifth of the truly seen points kept out of the fits, then placed like hidden ones.
        held = (torch.rand(vis.shape, device=device) < 0.2) & vis
        out_belief = belief.clone()
        out_belief[:, :, :N] = belief[:, :, :N] * ~held
        e_held = (to_metres(place(pos, seen=out_belief)[:, :, :N], b) - gt).norm(dim=-1)
        per_frame = lambda m: torch.stack([e_held[:, t][m[:, t]].median() if m[:, t].any()
                                           else e_held.new_tensor(float("nan")) for t in range(T)])
        for kind, m in (("in the fit", vis & ~held & moving), ("kept out", held & moving), ("hidden", ~vis & moving)):
            rows.setdefault(("held", kind), []).append(per_frame(m).cpu())
        # Each point that hides, lined up on the frame it hides in.
        for bi, t1, i in torch.nonzero(vis[:, :-1] & ~vis[:, 1:]).tolist():
            for k in range(-3, 9):
                t = t1 + 1 + k
                if 0 <= t < T and (k < 0 or not vis[bi, t1 + 1:t + 1, i].any()):
                    rows.setdefault(("hiding", k), []).append(err[bi, t, i].item())
        # By how far a point is from the centre of its true piece's points seen in that frame.
        seen_w = vis.float()[:, :, None, :] * mates.float()[:, None]                          # (B, T, N, N)
        centre = (seen_w[..., None] * gt[:, :, None]).sum(3) / seen_w.sum(3, keepdim=True).clamp_min(1)
        far_from = (gt - centre).norm(dim=-1)
        for kind, mask in (("hidden", ~vis & moving & (seen_mates >= 3)), ("seen", vis & moving & (seen_mates >= 3))):
            rows.setdefault(("by distance", kind), []).append(torch.stack([far_from[mask], err[mask]], -1).cpu())

    print("\nAPD (no rescale), the model's own guess and where the object's pose puts each point:")
    names = tuple(n for n in names if ("hidden", "all", n) in rows)       # "true centre" only where the clips have centres
    print(f"  {'point-frames':<16}" + "".join(f"{n:>14}" for n in names))
    for kind in ("hidden", "seen") + (("forecast", "fc 1-4", "fc 5-12", "fc 13+") if args.cutoff else ()):
        for which in ("all", "moving") + (("still",) if args.cutoff else ()):
            print(f"  {kind + ' ' + which:<16}" + "".join(f"{clip_mean(rows[kind, which, n]):>14.4f}" for n in names))
    total = max(sum(counts.values()), 1)
    print("\nHidden moving point-frames by how many points of their true piece are seen there:")
    print(f"  {'seen mates':<10} {'share':>6}" + "".join(f"{n:>14}" for n in names))
    for label, _, _ in bins:
        print(f"  {label:<10} {counts[label] / total:>6.3f}"
              + "".join(f"{clip_mean(rows['mates', label, n]):>14.4f}" for n in names))
    total = max(sum(way_counts.values()), 1)
    print("\nHidden moving point-frames by how the fit places them:")
    print(f"  {'placed by':<12} {'share':>6}" + "".join(f"{n:>14}" for n in names))
    for w in ways:
        print(f"  {w:<12} {way_counts[w] / total:>6.3f}" + "".join(f"{clip_mean(rows['way', w, n]):>14.4f}" for n in names))
    if flight_counts:
        total = max(sum(flight_counts.values()), 1)
        print("\nForecast moving point-frames by whether their object is still in free flight, and frames into the forecast:")
        print(f"  {'object':<12} {'frames':<6} {'share':>6}" + "".join(f"{n:>14}" for n in names))
        for (label, part), count in flight_counts.items():
            print(f"  {label:<12} {part:<6} {count / total:>6.3f}"
                  + "".join(f"{clip_mean(rows['flight', label, part, n]):>14.4f}" for n in names))
    if any(k[0] == "flight ref" for k in rows):
        refs = ("true centre, no turn", "thrown, no turn", "thrown, true turn")
        print("\nThe same forecast point-frames from the truth at the last seen frame:")
        print(f"  {'object':<12} {'frames':<6}" + "".join(f"{n:>22}" for n in refs))
        for label in ("free flight", "in contact"):
            for part in ("1-4", "5-12"):
                print(f"  {label:<12} {part:<6}" + "".join(f"{clip_mean(rows['flight ref', label, part, n]):>22.4f}" for n in refs))
    ks = sorted(key[1] for key in rows if key[0] == "hiding")
    if ks:
        print("\nThe object's pose, error (cm, median) around the frame a point hides in (0), seen before it:")
        print("  frames from hiding " + "".join(f"{k:>6}" for k in ks))
        print("  error              " + "".join(f"{100 * torch.tensor(rows['hiding', k]).median():>6.1f}" for k in ks))
    print("\nThe object's pose with a fifth of the seen points kept out of its fits, error (cm, median) by frame:")
    print("  frame     " + "".join(f"{t:>6}" for t in range(len(rows["held", "hidden"][0]))))
    for kind in ("in the fit", "kept out", "hidden"):
        e = torch.stack(rows["held", kind]).nanmedian(0).values
        print(f"  {kind:<10}" + "".join(f"{x:>6.1f}" if x == x else f"{'-':>6}" for x in (100 * e).tolist()))
    print("\nThe object's pose, error (cm, median) by distance from the centre of the piece's seen points:")
    for kind in ("seen", "hidden"):
        v = torch.cat(rows["by distance", kind])
        if not len(v):
            continue
        edges = torch.quantile(v[:, 0], torch.linspace(0, 1, 5))
        parts = [f"{lo:.2f}-{hi:.2f} m: {100 * v[(v[:, 0] >= lo) & (v[:, 0] <= hi), 1].median():.1f}"
                 for lo, hi in zip(edges[:-1], edges[1:])]
        print(f"  {kind:<7}" + "   ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
