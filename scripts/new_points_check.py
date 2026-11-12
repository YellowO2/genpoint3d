"""Whether more visible points make the rigid fit place hidden points better:
on the standard test's clips from the dense cache, the scored points tracked
alone, and, in a pass of their own, the clip's next points given as new points
in frame `--at` where seen there (their true position, their look read there:
as exact as a query's), so neither pass meets points it is not used to. Each
scored point's hidden moving point-frames from that frame on, placed by a
robust rigid fit (3 refits, far mates counting less) of the visible points of
its true piece: the old ones only (carried from their query frames), the new
ones only (carried from frame `--at`, the old point from where it was guessed
there), and both; beside the old point's own guess. Visible is the truth, so
only the number of seen mates differs.

Run:  python scripts/new_points_check.py --ckpt outputs/RUN/best.pt --cache DIR --frames-cache FRAMES
"""

import argparse
import math

import torch
from torch.utils.data import DataLoader

from genpoint3d.tools.feature_lookup import sample_at
from genpoint3d.train.score.Run import Run
from genpoint3d.train.tools.batch import fine_inputs, rigid_groups, to_device, to_metres, visual_mask
from genpoint3d.train.tools.camera import project
from genpoint3d.train.tools.metrics import clip_mean, motion_px, tapvid3d_metrics
from genpoint3d.train.tools.regress import Sweeps


def _part(b: dict, keep) -> dict:
    """`b` with only the points `keep` (indices)."""
    out = dict(b)
    for k in ("traj", "visibility", "norm_offset"):
        out[k] = b[k][:, :, keep]
    for k in ("anchor", "query_t", "id_card", "segment"):
        out[k] = b[k][:, keep]
    return out


def _requeried(model, b: dict, fine: dict, keep, at: int) -> dict:
    """The points `keep` given again in frame `at`: their true position there
    as anchor, their DINOv3 look read there. B = 1."""
    x, anchor = b["traj"].float(), b["anchor"].float()
    pos = model.lookup_pos(x, anchor)[:, :, keep]                            # (1, T, M, 3)
    new_anchor = pos[:, at]
    uv, _ = project(new_anchor[:, None], fine["fine_proj"][:, at:at + 1].float())
    H, W = b["hw"][0].tolist()
    ctx = b["context"][0, at].float()
    g = math.isqrt(ctx.shape[0])
    card = sample_at(ctx.T.reshape(1, -1, g, g), uv[:, 0] * uv.new_tensor([W, H]), (H, W))
    out = _part(b, keep)
    out["traj"] = ((pos - new_anchor[:, None]) / model.traj_scale).to(b["traj"].dtype)
    out["anchor"] = new_anchor.to(b["anchor"].dtype)
    out["query_t"] = torch.full_like(out["query_t"], at)
    out["id_card"] = card.to(b["id_card"].dtype)
    out["norm_offset"] = new_anchor[:, None].to(b["norm_offset"].dtype)
    return out


def _robust_place(src, dst, w0, own, refits: int = 3):
    """(1, T, n, 3) each of n points carried by the turn and shift that best
    carry its mates from `src` (1, n, M, 3) to `dst` (1, T, M, 3), weighted by
    `w0` (1, T, n, M), refit with far mates counting less, applied to `own`
    (1, n, 3); and (1, T, n) where at least 3 mates' weight backs it."""
    src = src[:, None]                                                       # (1, 1, n, M, 3)
    dst = dst[:, :, None]                                                    # (1, T, 1, M, 3)
    w = w0

    def fit(w):
        s = w.sum(-1)
        mean = lambda x: (w[..., None] * x).sum(-2) / s.clamp_min(1e-6)[..., None]
        ms, md = mean(src.expand(*w.shape, 3)), mean(dst.expand(*w.shape, 3))
        H = torch.einsum("btnm,btnmx,btnmy->btnxy", w, src - ms[..., None, :], dst - md[..., None, :])
        U, _, Vh = torch.linalg.svd(H.double())
        V, Ut = Vh.transpose(-1, -2), U.transpose(-1, -2)
        flip = torch.ones(*H.shape[:-1], device=w.device, dtype=torch.float64)
        flip[..., 2] = torch.linalg.det(V @ Ut).sign()
        return (V @ torch.diag_embed(flip) @ Ut).float(), ms, md, s

    R, ms, md, s = fit(w)
    for _ in range(refits):
        left = ((R[..., None, :, :] @ (src - ms[..., None, :])[..., None])[..., 0] + md[..., None, :] - dst).norm(dim=-1)
        typical = (w * left).sum(-1, keepdim=True) / s.clamp_min(1e-6)[..., None] + 1e-6
        w = w0 / (1 + (left / typical).pow(2))
        R, ms, md, s = fit(w)
    return (R @ (own[:, None] - ms)[..., None])[..., 0] + md, w0.sum(-1) >= 3


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--frames-cache", required=True)
    p.add_argument("--points", type=int, default=256, help="the scored points; the clip's next as many are new ones")
    p.add_argument("--at", type=int, default=8, help="the frame the new points are given in")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run.load(args.ckpt, device)
    model, s = run.model, run.settings
    ds = run.dataset(args.cache, args.frames_cache, points=2 * args.points)
    amp = device.type == "cuda"
    at = args.at
    print(f"{len(ds)} clips | step {run.step} of {args.ckpt} | new points in frame {at}", flush=True)

    rows, news = {}, []
    for batch in DataLoader(ds, batch_size=1, shuffle=False, num_workers=4):
        b = to_device(batch, device)
        N = b["traj"].shape[2]
        n = min(args.points, N)
        fine = fine_inputs(model, b, amp)
        seen_at = b["visibility"].bool()[0, at]
        extra = torch.nonzero(seen_at[n:])[:, 0] + n
        if len(extra) < 3:
            continue
        old, new = _part(b, torch.arange(n, device=device)), _requeried(model, b, fine, extra, at)
        news.append(len(extra))
        T = old["traj"].shape[1]

        def track(bb):
            cond = dict(context=bb["context"], visual_mask=visual_mask(bb, None), id_card=bb["id_card"],
                        patch_xyz=bb["patch_xyz"], **{**fine, "query_t": bb["query_t"]})
            sweeps = Sweeps(model, bb["anchor"], T, s.schedule, s.window, s.overlap, cond)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in sweeps:
                    pass
            return model.lookup_pos(sweeps.whole.float(), bb["anchor"].float())

        p_old, p_new = track(old), track(new)                                # (1, T, ., 3) anchor's space
        vis_old, vis_new = old["visibility"].bool(), new["visibility"].bool()
        # True pieces over the clip: every point against every other, old then new.
        both = dict(old)
        both["traj"] = torch.cat([b["traj"][:, :, :n], b["traj"][:, :, extra]], 2)
        both["norm_offset"] = torch.cat([b["norm_offset"][:, :, :n], b["norm_offset"][:, :, extra]], 2)
        both["segment"] = torch.cat([b["segment"][:, :n], b["segment"][:, extra]], 1)
        piece = rigid_groups(both)                                           # (1, n + M, n + M)
        piece = piece & ~torch.eye(piece.shape[1], dtype=torch.bool, device=device)
        q = old["query_t"]
        src_old = p_old.gather(1, q[:, None, :, None].expand(-1, 1, -1, 3))[:, 0]   # (1, n, 3) where queried
        own_q = src_old
        own_at = p_old[:, at]
        w_old = piece[:, None, :n, :n].float() * vis_old[:, :, None, :].float()      # (1, T, n, n)
        w_new = piece[:, None, :n, n:].float() * vis_new[:, :, None, :].float()      # (1, T, n, M)
        by_old, ok_old = _robust_place(src_old[:, None].expand(-1, n, -1, -1), p_old, w_old, own_q)
        by_new, ok_new = _robust_place(p_new[:, at][:, None].expand(-1, n, -1, -1), p_new, w_new, own_at)
        # Both kinds of mate, each from where its own track starts: carried in one fit
        # from frame `at`, where both are placed (old mates at their guesses there).
        src_both = torch.cat([p_old[:, at], p_new[:, at]], 1)[:, None].expand(-1, n, -1, -1)
        by_both, ok_both = _robust_place(src_both, torch.cat([p_old, p_new], 2),
                                         torch.cat([w_old, w_new], -1), own_at)
        gt = to_metres(old["traj"], old)
        moving = (motion_px(gt, vis_old, old["intrinsics_256"], old["extrinsics"], q) > 4)[:, None]
        late = (torch.arange(T, device=device) >= at)[None, :, None]
        hid = ~vis_old & moving & late
        to_x = lambda pos: (pos - old["anchor"].float()[:, None]) / model.traj_scale
        guess = to_x(p_old)
        for name, pts, ok in (("own guess", guess, None), ("old mates", to_x(by_old), ok_old),
                              ("new mates", to_x(by_new), ok_new), ("both", to_x(by_both), ok_both)):
            placed = guess if ok is None else torch.where(ok[..., None], pts, guess)
            for which, m in (("hidden moving", hid), ("with 10+ old seen", hid & (w_old.sum(-1) >= 10)),
                             ("with 1-9 old seen", hid & (w_old.sum(-1) >= 1) & (w_old.sum(-1) < 10)),
                             ("with none old seen", hid & (w_old.sum(-1) < 1))):
                apd = tapvid3d_metrics(to_metres(placed, old), gt, m, old["intrinsics_256"], old["extrinsics"],
                                       scaling="none", per_clip=True)["average_pts_within_thresh"].cpu()
                rows.setdefault((name, which), []).append(apd)

    print(f"\nnew points per clip in frame {at}: median {sorted(news)[len(news) // 2]}")
    whiches = ("hidden moving", "with 10+ old seen", "with 1-9 old seen", "with none old seen")
    print(f"APD (no rescale), scored points' hidden moving point-frames from frame {at}, placed by a robust fit of their "
          f"true piece's seen points (the own guess where under 3):")
    print(f"  {'':<12}" + "".join(f"{w:>20}" for w in whiches))
    for name in ("own guess", "old mates", "new mates", "both"):
        print(f"  {name:<12}" + "".join(f"{clip_mean(rows[name, w]):>20.4f}" for w in whiches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
