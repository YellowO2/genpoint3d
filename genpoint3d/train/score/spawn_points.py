"""New points where surface comes into view with no point on it, as a tracker
on real video adds them: training's spawning (`ClipDataset._spawned`) at test
time, without the truth. One rule for every frame: a place on a surface that
no point sits on gets one. As in training, where the scene's points come as
given and only new surface is capped, the first frame's points are limited by
the gap alone and the budget is for the frames after it."""

import torch

from genpoint3d.tools.feature_lookup import id_cards
from genpoint3d.tools.traj_scale import RADIUS
from genpoint3d.train.data.ClipDataset import farthest_first
from genpoint3d.train.tools.camera import project


@torch.no_grad()
def spawned(model, b: dict, fine: dict, tracks: torch.Tensor, view: torch.Tensor, cap: int, gap: float) -> dict:
    """`b` (one clip) with points added after its own, up to `cap` after the
    first frame: frame by frame, in each frame seen whole, the CNN cells no
    point covers, the farthest first, each starting at its cell in that frame,
    its look read there. A point covers a cell if it sits on that surface:
    within `gap` (a share of the frame's width) of it in the image and at its
    depth, seen or not, so a point turned to the far side of an object leaves
    the near side to new points while one merely believed hidden still covers
    its place. `tracks` is a first pass over `b`'s own points; a point added
    earlier is carried by the motion of the own point nearest it in 3D. The
    added points' truth is unknown: never visible, so never scored."""
    if b["traj"].shape[0] != 1:
        raise ValueError("spawning scores one clip at a time (--batch 1)")
    T, N = tracks.shape[1:3]
    pos = model.lookup_pos(tracks.float(), b["anchor"].float())[0]                     # (T, N, 3)
    proj = fine["fine_proj"][0].float()                                                 # (T, 3, 4)
    H, W = b["hw"][0].tolist()
    whole = view[0].flatten(1).all(-1)                                                  # (T,)
    q = b["query_t"][0]
    exists = (q[None] <= torch.arange(T, device=q.device)[:, None]) & whole[q][None]   # (T, N) given by then
    xyz = fine["fine_xyz"][0].flatten(2).transpose(1, 2)                                # (T, h*w, 3)
    h, w = fine["fine_xyz"].shape[-2:]
    v, u = torch.meshgrid(torch.arange(h, device=pos.device), torch.arange(w, device=pos.device), indexing="ij")
    cell_px = torch.stack([(u.flatten() + 0.125) / w * W, (v.flatten() + 0.125) / h * H], -1).float()
    far = gap * W
    born = []                                                       # (frame, cell, position, nearest own point)
    first, later = True, 0                                          # the first seen frame yet to come; points added after it
    for t in range(T):
        if not whole[t]:
            continue
        own = torch.nonzero(exists[t])[:, 0]
        here = torch.cat([pos[t, own]] + [(p + pos[t, i] - pos[t0, i])[None] for t0, _, p, i in born])
        depth = b["depth"][0, t].flatten().float()
        cand = torch.nonzero(depth < RADIUS)[:, 0]
        if not len(cand) or not len(own):
            continue
        uv, z = project(here[None, None], proj[None, t:t + 1])
        on = (z[0, 0][None] - depth[cand, None]).abs() < 0.03 * depth[cand, None]          # (C, P) on that surface
        d = torch.cdist(cell_px[cand], uv[0, 0] * uv.new_tensor([W, H])).masked_fill(~on, float("inf")).min(1).values
        picked = farthest_first(cell_px[cand], d, far, len(cand) if first else cap - later)
        later += 0 if first else len(picked)
        first = False
        for j in picked:
            p = xyz[t, cand[j]]
            i = int(own[(pos[t, own] - p).norm(dim=-1).argmin()])
            born.append((t, int(cand[j]), p, i))
    if not born:
        return b
    M = len(born)
    start = torch.tensor([t for t, *_ in born], device=pos.device)
    cell = torch.tensor([c for _, c, *_ in born], device=pos.device)
    anchor = torch.stack([p for *_, p, _ in born])[None]                                # (1, M, 3)
    card = id_cards(b["context"][0], start, cell_px[cell], (H, W))
    out = dict(b)
    add = lambda k, x, dim: torch.cat([b[k], x.to(b[k].dtype)], dim)
    out["traj"] = add("traj", torch.zeros(1, T, M, 3, device=pos.device), 2)
    out["visibility"] = add("visibility", torch.zeros(1, T, M, dtype=torch.bool, device=pos.device), 2)
    out["anchor"] = add("anchor", anchor, 1)
    out["query_t"] = add("query_t", start[None], 1)
    out["id_card"] = add("id_card", card[None], 1)
    if "segment" in b:
        out["segment"] = add("segment", torch.full((1, M), -1, device=pos.device), 1)
    if "norm_offset" in b:
        out["norm_offset"] = add("norm_offset", anchor[:, None], 2)
    return out
