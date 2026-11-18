"""A point stands for the smallest piece of surface the camera can tell apart, never smaller than UNIT."""

from typing import Callable, Optional

import torch

GAP = 0.05          # the piece the camera tells apart, a share of the frame's width
UNIT = 0.05         # the smallest piece, metres
SURFACE_TOL = 0.03  # a point is on the surface its pixel shows within this share of the depth there
PATCH = 4            # the nearest points of a body a flat patch is fitted through
THIN = 0.1           # three points span a patch if their triangle is at least this wide for its length


def spacing(z: torch.Tensor, focal: float, width: float, gap: float) -> torch.Tensor:
    """Pixels between points on surface `z` metres from the camera."""
    return (UNIT * focal / z.clamp_min(1e-6)).clamp_min(gap * width)


def on_patch(pos: torch.Tensor, body: torch.Tensor, at: torch.Tensor, of: torch.Tensor, flat: float) -> torch.Tensor:
    """(C,) whether each place `pos` (C, 3) of body `body` (C,) lies within
    `flat` metres of the flat patch its body's nearest points span, as a
    face of a game's mesh: the points are `at` (M, 3) of bodies `of` (M,),
    the nearest PATCH of them fitted by a plane. Points in a line span only
    the line."""
    if len(at) < 3:
        return torch.zeros(len(pos), dtype=torch.bool, device=pos.device)
    near, i = torch.cdist(pos, at).masked_fill(body[:, None] != of[None], float("inf")).topk(min(PATCH, len(at)), largest=False)
    has = near.isfinite().to(pos.dtype)[..., None]                                # (C, k, 1)
    middle = (at[i] * has).sum(1) / has.sum(1).clamp_min(1)
    arm = (at[i] - middle[:, None]) * has
    size, way = torch.linalg.eigh(arm.transpose(1, 2) @ arm)                      # ascending
    wide = size[:, 1:] >= THIN ** 2 * size[:, 2:].clamp_min(1e-12)                # (C, 2) the patch's two ways
    out = pos - middle
    along = ((out[:, None] @ way[:, :, 1:]).squeeze(1) ** 2 * wide).sum(-1)
    off = (out.pow(2).sum(-1) - along).clamp_min(0).sqrt()
    return (has.sum(1)[:, 0] >= 3) & (off <= flat)


def add_points(frames: list[int], room: int,
               candidates: Callable[[int, list], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
               covering: Callable[[int, list], torch.Tensor],
               surface: Optional[Callable[[int, list], tuple[torch.Tensor, ...]]] = None,
               flat: float = 0.0) -> list[tuple[int, int]]:
    """Up to `room` new points as (frame, id): in each frame, a candidate with
    no point within its spacing, the emptiest place first. `candidates(t,
    born)` gives pixels (C, 2), ids (C,) and spacings (C,); `covering(t,
    born)` the pixels (M, 2) of the points seen there. With `surface(t,
    born)`, giving the candidates' places (C, 3) and bodies (C,) and the seen
    points' (M, 3), (M,), a candidate its body's points already explain
    (`on_patch`) is left out too, so a flat floor takes few."""
    born = []
    for t in frames:
        px, ids, far = candidates(t, born)
        if not len(ids):
            continue
        cover = covering(t, born)
        d = (torch.cdist(px, cover).min(1).values if len(cover) else torch.full_like(far, float("inf"))) / far
        explained = torch.zeros(len(ids), dtype=torch.bool, device=px.device)
        if surface is not None:
            pos, body, at, of = surface(t, born)
            explained = on_patch(pos, body, at, of, flat)
        while len(born) < room:
            j = int(d.masked_fill(explained, 0).argmax())
            if explained[j] or d[j] <= 1:
                break
            born.append((t, int(ids[j])))
            d = torch.minimum(d, (px - px[j]).norm(dim=-1) / far)
            if surface is not None:
                at, of, same = torch.cat([at, pos[j:j + 1]]), torch.cat([of, body[j:j + 1]]), body == body[j]
                explained[same] = on_patch(pos[same], body[same], at, of, flat)
    return born
