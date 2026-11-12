"""Whether exactly placed new points help the old ones: on the standard test,
the clip's points past the scored ones, given to the model as new points in
frame 12 where they are seen there (their true position, their look read
there: as exact as a query's). The scored points' APD on their moving
point-frames from frame 12 on, three ways: tracked alone; tracked beside the
new points (what attention between points already carries over); and placed
by a rigid fit of the new points on their true piece, seen in that frame,
carried from frame 12 (what new points and right groups could give). Every
frame keeps its image.

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

AT = 12


def _with_new(model, b: dict, fine: dict, n_old: int) -> tuple[dict, int]:
    """`b` with its points past the first `n_old` given again as new points in
    frame AT, those seen there: their true position as anchor, their DINOv3
    look read there. B = 1. And how many new points."""
    x, anchor, vis = b["traj"].float(), b["anchor"].float(), b["visibility"].bool()
    pos = model.lookup_pos(x, anchor)                                          # (1, T, N, 3)
    sel = torch.nonzero(vis[0, AT, n_old:])[:, 0] + n_old
    new_anchor = pos[:, AT, sel]                                               # (1, M, 3)
    new_x = (pos[:, :, sel] - new_anchor[:, None]) / model.traj_scale
    uv, _ = project(new_anchor[:, None], fine["fine_proj"][:, AT:AT + 1].float())
    H, W = b["hw"][0].tolist()
    ctx = b["context"][0, AT].float()                                          # (P, F)
    g = math.isqrt(ctx.shape[0])
    card = sample_at(ctx.T.reshape(1, -1, g, g), uv[:, 0] * uv.new_tensor([W, H]), (H, W))
    keep = slice(0, n_old)
    b2 = dict(b)
    b2["traj"] = torch.cat([b["traj"][:, :, keep], new_x.to(b["traj"].dtype)], 2)
    b2["anchor"] = torch.cat([b["anchor"][:, keep], new_anchor.to(b["anchor"].dtype)], 1)
    b2["visibility"] = torch.cat([b["visibility"][:, :, keep], b["visibility"][:, :, sel]], 2)
    b2["query_t"] = torch.cat([b["query_t"][:, keep], torch.full_like(sel, AT)[None]], 1)
    b2["id_card"] = torch.cat([b["id_card"][:, keep], card.to(b["id_card"].dtype)], 1)
    b2["segment"] = torch.cat([b["segment"][:, keep], b["segment"][:, sel]], 1)
    b2["norm_offset"] = torch.cat([b["norm_offset"][:, :, keep], new_anchor[:, None].to(b["norm_offset"].dtype)], 2)
    return b2, len(sel)


def _fit_from_new(pos: torch.Tensor, vis: torch.Tensor, mates: torch.Tensor, n_old: int) -> tuple:
    """(1, T, n_old, 3) each old point carried from frame AT by the turn and
    shift that best carry its mates (`mates` (1, n_old, M), new points on its
    true piece) from AT to each frame, weighted by being seen there; (1, T,
    n_old) where at least 3 mates are seen."""
    src = pos[:, AT, n_old:]                                                   # (1, M, 3)
    dst = pos[:, :, n_old:]                                                    # (1, T, M, 3)
    w = mates[:, None].float() * vis[:, :, None, n_old:].float()               # (1, T, n_old, M)
    tot = w.sum(-1, keepdim=True).clamp_min(1e-8)
    cs = torch.einsum("btim,bmc->btic", w, src) / tot                          # (1, T, n_old, 3)
    cd = torch.einsum("btim,btmc->btic", w, dst) / tot
    a = src[:, None, None] - cs[..., None, :]                                  # (1, T, n_old, M, 3)
    d = dst[:, :, None] - cd[..., None, :]
    Hm = torch.einsum("btim,btimc,btimd->btidc", w, a, d)                      # dst x src
    U, _, Vh = torch.linalg.svd(Hm.double())
    S = torch.diag_embed(torch.ones_like(U[..., 0]))
    S[..., 2, 2] = torch.sign(torch.linalg.det(U @ Vh))
    R = (U @ S @ Vh).float()                                                   # src -> dst
    own = pos[:, AT, :n_old][:, None]                                          # (1, 1, n_old, 3)
    out = torch.einsum("btiCc,btic->btiC", R, own - cs) + cd
    return out, (w > 0).sum(-1) >= 3


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--frames-cache", required=True)
    p.add_argument("--points", type=int, default=256, help="the scored points; the clip's next as many are new ones")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run.load(args.ckpt, device)
    model, s = run.model, run.settings
    ds = run.dataset(args.cache, args.frames_cache, points=2 * args.points)
    amp = device.type == "cuda"
    print(f"{len(ds)} clips | step {run.step} of {args.ckpt}", flush=True)

    rows, news, covered = {}, [], []
    for batch in DataLoader(ds, batch_size=1, shuffle=False, num_workers=4):
        b = to_device(batch, device)
        n_old = min(args.points, b["traj"].shape[2])
        old = {k: (v[:, :, :n_old] if k in ("traj", "visibility", "norm_offset")
                   else v[:, :n_old] if k in ("anchor", "query_t", "id_card", "segment") else v) for k, v in b.items()}
        fine = fine_inputs(model, b, amp)
        T = old["traj"].shape[1]
        vis, q = old["visibility"].bool(), old["query_t"]
        gt = to_metres(old["traj"], old)
        moving = (motion_px(gt, vis, old["intrinsics_256"], old["extrinsics"], q) > 4)[:, None]
        late = (torch.arange(T, device=device) >= AT)[None, :, None]
        m = vis & moving & late
        both, n_new = _with_new(model, b, fine, n_old)
        news.append(n_new)

        def track(bb):
            cond = dict(context=bb["context"], visual_mask=visual_mask(bb, None), id_card=bb["id_card"],
                        patch_xyz=bb["patch_xyz"], **{**fine, "query_t": bb["query_t"]})
            sweeps = Sweeps(model, bb["anchor"], T, s.schedule, s.window, s.overlap, cond)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in sweeps:
                    pass
            return sweeps.whole.float()

        alone = track(old)
        beside = track(both)
        pos = model.lookup_pos(beside, both["anchor"].float())
        mates = rigid_groups(both)[:, :n_old, n_old:]
        fit, ok = _fit_from_new(pos, both["visibility"].bool(), mates, n_old)
        fitted = torch.where(ok[..., None], (fit - old["anchor"].float()[:, None]) / model.traj_scale,
                             beside[:, :, :n_old])
        covered.append(((ok & m).sum() / m.sum().clamp_min(1)).item())
        for name, pts in (("alone", alone), ("beside new points", beside[:, :, :n_old]),
                          ("fit from new points", fitted)):
            apd = tapvid3d_metrics(to_metres(pts, old), gt, m, old["intrinsics_256"], old["extrinsics"],
                                   scaling="none", per_clip=True)["average_pts_within_thresh"].cpu()
            rows.setdefault(name, []).append(apd)

    print(f"\nnew points in frame {AT} per clip: median {sorted(news)[len(news) // 2]}; "
          f"share of scored moving point-frames the fit reaches (3+ seen mates): {sum(covered) / len(covered):.3f}")
    print(f"APD (no rescale), scored points, moving, seen, frames {AT}-{T - 1}:")
    for name in ("alone", "beside new points", "fit from new points"):
        print(f"  {name:<22} {clip_mean(rows[name]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
