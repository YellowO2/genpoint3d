"""Points added at test time as training adds them (`add_points`), without the truth."""

import torch

from genpoint3d.tools.feature_lookup import id_cards
from genpoint3d.tools.traj_scale import REACH
from genpoint3d.train.tools.batch import whole_frames
from genpoint3d.train.tools.camera import project
from genpoint3d.train.tools.farthest_first import SURFACE_TOL, add_points, spacing


def spawn_points(model, b: dict, fine: dict, view: torch.Tensor, points: int, gap: float, tracks_of) -> dict:
    """`b` with points added up to `points`: the first frame's, which need no
    tracking, then with those tracked (`tracks_of(clip)` gives tracks and
    visibility logits) the later frames'."""
    b = _spawned(model, b, fine, torch.zeros_like(b["traj"]), None, view, gap, points, True)
    return _spawned(model, b, fine, *tracks_of(b), view, gap, points, False)


@torch.no_grad()
def _spawned(model, b: dict, fine: dict, tracks: torch.Tensor, logits: torch.Tensor | None, view: torch.Tensor, gap: float,
             points: int, first: bool) -> dict:
    """`b` (one clip) with points added on the CNN cells of its first frame
    seen whole (`first`) or of every one. A point covers its pixel if it is
    believed seen and sits at the depth shown there. A point added in this
    call moves as the own point nearest it in 3D. The added points have no
    truth: never visible, never scored."""
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
    carrier = {}                                                    # an added point: the own point nearest it then

    def at(t0, cell):
        """An added point's place in its first frame and its carrier."""
        if (t0, cell) not in carrier:
            own = torch.nonzero(exists[t0])[:, 0]
            p = xyz[t0, cell]
            carrier[t0, cell] = p, int(own[(pos[t0, own] - p).norm(dim=-1).argmin()])
        return carrier[t0, cell]

    def candidates(t, born):
        z = b["depth"][0, t].flatten().float()
        cand = torch.nonzero(z < REACH)[:, 0] if exists[t].any() else cell_px.new_zeros(0).long()
        return cell_px[cand], cand, spacing(z[cand], float(b["intrinsics"][0, t, 0, 0]), W, gap)

    def covering(t, born):
        own = torch.nonzero(exists[t])[:, 0]
        carried = [(p + pos[t, i] - pos[t0, i], i) for t0, c in born for p, i in [at(t0, c)]]
        here = torch.cat([pos[t, own]] + [x[None] for x, _ in carried])
        shown = torch.cat([believed[t, own], believed[t, [i for _, i in carried]]])     # an added one as its carrier
        depth = b["depth"][0, t].flatten().float()
        uv, z = project(here[None, None], proj[None, t:t + 1])
        uv, z = uv[0, 0], z[0, 0]
        px = (uv[:, 1] * h).long().clamp(0, h - 1) * w + (uv[:, 0] * w).long().clamp(0, w - 1)
        on = ((z - depth[px]).abs() < SURFACE_TOL * depth[px]) & shown
        return uv[on] * uv.new_tensor([W, H])

    seen = torch.nonzero(whole)[:, 0].tolist()
    born = [(t0, c, *at(t0, c)) for t0, c in add_points(seen[:1] if first else seen, points - N, candidates, covering)]
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
    if "asked" in b:
        out["asked"] = add("asked", torch.zeros(1, M, dtype=torch.bool, device=pos.device), 1)
    return out
