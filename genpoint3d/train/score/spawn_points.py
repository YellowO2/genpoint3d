"""New points where surface comes into view with no point on it.

They are added as a tracker on real video adds them: training's spawning
(`ClipDataset._spawned`) at test time, without the truth. One rule for every
frame: a place on a surface that no point sits on gets one. As in training,
where the scene's points come as given and only new surface is capped, the
first frame's points are limited by the gap and the clip's point budget, and
new surface after it fills the rest of that budget, as training's points per
clip. The first frame's points are tracked before the later ones are added, so
every point already there covers where it really goes (`spawn_points`)."""

import torch

from genpoint3d.tools.feature_lookup import id_cards
from genpoint3d.tools.traj_scale import RADIUS
from genpoint3d.train.tools.batch import whole_frames
from genpoint3d.train.tools.camera import project
from genpoint3d.train.tools.farthest_first import farthest_first


def spawn_points(model, b: dict, fine: dict, view: torch.Tensor, points: int, gap: float, tracks_of) -> dict:
    """`b` with points added until it has `points`, in two rounds, each
    tracked before the next (`tracks_of(clip)` gives a clip's tracks and
    visibility logits): the first frame's gaps, then new surface after it."""
    b = _spawned(model, b, fine, *tracks_of(b), view, gap, points - b["traj"].shape[2], True)
    return _spawned(model, b, fine, *tracks_of(b), view, gap, points - b["traj"].shape[2], False)


@torch.no_grad()
def _spawned(model, b: dict, fine: dict, tracks: torch.Tensor, logits: torch.Tensor | None, view: torch.Tensor, gap: float,
             cap: int, first: bool) -> dict:
    """`b` (one clip) with up to `cap` points added after its own: in the
    first frame seen whole if `first`, else in the frames seen whole after it,
    frame by frame. They go where no point covers the CNN cells, the
    farthest first, each starting at its cell in that frame, its look read
    there. A point covers the cells within `gap` (a share of the frame's
    width) of it in the image if it is believed seen there (as training's
    spawning counts the points seen) and sits on the surface its own pixel
    shows (at the depth there): a hidden point's guess, wherever it lands,
    covers nothing. `tracks` is a first pass over `b`'s own points; a point added
    earlier in the same round is carried by the motion of the own point
    nearest it in 3D. The added points' truth is unknown: never visible, so
    never scored."""
    if b["traj"].shape[0] != 1:
        raise ValueError("spawning scores one clip at a time (--batch 1)")
    T, N = tracks.shape[1:3]
    pos = model.lookup_pos(tracks.float(), b["anchor"].float())[0]                     # (T, N, 3)
    proj = fine["fine_proj"][0].float()                                                 # (T, 3, 4)
    H, W = b["hw"][0].tolist()
    whole = whole_frames(view[0])                                                       # (T,)
    q = b["query_t"][0]
    exists = (q[None] <= torch.arange(T, device=q.device)[:, None]) & whole[q][None]   # (T, N) given by then
    believed = (torch.sigmoid(logits[0].float()) > 0.5 if logits is not None           # (T, N)
                else torch.ones(T, N, dtype=torch.bool, device=pos.device))
    xyz = fine["fine_xyz"][0].flatten(2).transpose(1, 2)                                # (T, h*w, 3)
    h, w = fine["fine_xyz"].shape[-2:]
    v, u = torch.meshgrid(torch.arange(h, device=pos.device), torch.arange(w, device=pos.device), indexing="ij")
    cell_px = torch.stack([(u.flatten() + 0.125) / w * W, (v.flatten() + 0.125) / h * H], -1).float()
    far = gap * W
    born = []                                                       # (frame, cell, position, nearest own point)
    seen = torch.nonzero(whole)[:, 0].tolist()
    for t in (seen[:1] if first else seen[1:]):
        own = torch.nonzero(exists[t])[:, 0]
        here = torch.cat([pos[t, own]] + [(p + pos[t, i] - pos[t0, i])[None] for t0, _, p, i in born])
        shown = torch.cat([believed[t, own], believed[t, [i for *_, i in born]]])          # an added one as its carrier
        depth = b["depth"][0, t].flatten().float()
        cand = torch.nonzero(depth < RADIUS)[:, 0]
        if not len(cand) or not len(own):
            continue
        uv, z = project(here[None, None], proj[None, t:t + 1])
        uv, z = uv[0, 0], z[0, 0]
        # On the surface its own pixel shows (the depth there, as a z-buffer would test),
        # not hidden behind it; a slope's depth changes across a gap, so test it there.
        at = ((uv[:, 1] * h).long().clamp(0, h - 1) * w + (uv[:, 0] * w).long().clamp(0, w - 1))
        on = ((z - depth[at]).abs() < 0.03 * depth[at]) & shown
        d = (torch.cdist(cell_px[cand], uv[on] * uv.new_tensor([W, H])).min(1).values if on.any()
             else torch.full((len(cand),), float("inf"), device=pos.device))
        picked = farthest_first(cell_px[cand], d, far, cap - len(born))
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
    if "centre" in b:
        out["centre"] = add("centre", torch.zeros(1, T, M, 3, device=pos.device), 2)
        out["centre_known"] = add("centre_known", torch.zeros(1, M, dtype=torch.bool, device=pos.device), 1)
    if "norm_offset" in b:
        out["norm_offset"] = add("norm_offset", anchor[:, None], 2)
    return out
