"""New points where a clip shows surface no point covers, as a tracker on real
video adds them: training's spawning (`ClipDataset._spawned`) at test time,
without the truth. From frame 0 on, the same rule: the first frame's gaps get
points as CoTracker's predictor adds a support grid beside the queries, later
frames' gaps are surface coming into view."""

import torch

from genpoint3d.tools.feature_lookup import id_cards
from genpoint3d.tools.traj_scale import RADIUS
from genpoint3d.train.data.ClipDataset import farthest_first
from genpoint3d.train.tools.camera import project


@torch.no_grad()
def spawned(model, b: dict, fine: dict, tracks: torch.Tensor, logits: torch.Tensor | None,
            view: torch.Tensor, cap: int, gap: float) -> dict:
    """`b` (one clip) with up to `cap` points added after its own: frame by
    frame, in each frame seen whole, the CNN cells farther than `gap` (a share
    of the frame's width) from every point placed there and believed visible,
    the farthest first, each starting at its cell in that frame, its look read
    there. `tracks` and `logits` are a first pass over `b`'s own points; a
    point added earlier is carried by the motion of the own point nearest it
    in 3D, seen where that one is. The added points' truth is unknown: never
    visible, so never scored."""
    if b["traj"].shape[0] != 1:
        raise ValueError("spawning scores one clip at a time (--batch 1)")
    T, N = tracks.shape[1:3]
    pos = model.lookup_pos(tracks.float(), b["anchor"].float())[0]                     # (T, N, 3)
    proj = fine["fine_proj"][0].float()                                                 # (T, 3, 4)
    H, W = b["hw"][0].tolist()
    px = lambda p, t: project(p[None, None], proj[None, t:t + 1])[0][0, 0] * p.new_tensor([W, H])
    whole = view[0].flatten(1).all(-1)                                                  # (T,)
    q = b["query_t"][0]
    believed = (torch.sigmoid(logits[0].float()) > 0.5 if logits is not None
                else torch.ones(T, N, dtype=torch.bool, device=pos.device))
    believed = believed & (q[None] <= torch.arange(T, device=q.device)[:, None]) & whole[q][None]
    xyz = fine["fine_xyz"][0].flatten(2).transpose(1, 2)                                # (T, h*w, 3)
    h, w = fine["fine_xyz"].shape[-2:]
    v, u = torch.meshgrid(torch.arange(h, device=pos.device), torch.arange(w, device=pos.device), indexing="ij")
    cell_px = torch.stack([(u.flatten() + 0.125) / w * W, (v.flatten() + 0.125) / h * H], -1).float()
    far = gap * W
    born = []                                                       # (frame, cell, position, nearest own point)
    for t in range(T):
        if len(born) >= cap:
            break
        if not whole[t]:
            continue
        cover = [pos[t, believed[t]]]
        cover += [(p + pos[t, i] - pos[t0, i])[None] for t0, _, p, i in born if believed[t, i]]
        cover = torch.cat(cover)
        cand = torch.nonzero(b["depth"][0, t].flatten() < RADIUS)[:, 0]
        d = (torch.cdist(cell_px[cand], px(cover, t)).min(1).values if len(cover)
             else torch.full((len(cand),), float("inf"), device=pos.device))
        own = torch.nonzero(believed[t])[:, 0]
        for j in farthest_first(cell_px[cand], d, far, cap - len(born) if len(own) else 0):
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
