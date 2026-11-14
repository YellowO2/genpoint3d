"""Pick new points' places, the farthest from what is there first."""

from typing import Callable

import torch

# A new point starts only this far (a share of the frame's width) from every
# point seen there, in training (`ClipDataset._spawned`) and at test time
# (`spawn_points`).
SPAWN_GAP = 0.05
# A point covers its place only if it sits on the surface its own pixel shows:
# within this share of the depth there (test time; training knows who is seen).
SURFACE_TOL = 0.03


def farthest_first(px: torch.Tensor, d: torch.Tensor, far: float, most: int) -> list[int]:
    """Up to `most` of the candidates at pixels `px` (C, 2), the farthest
    first: each farther than `far` both by `d` (C,), its distance to what is
    there already, and from the ones picked before it."""
    picked = []
    while len(picked) < most:
        j = int(d.argmax())
        if d[j] <= far:
            break
        picked.append(j)
        d = torch.minimum(d, (px - px[j]).norm(dim=-1))
    return picked


def fill_frames(frames: list[int], first: int, total: int, far: float,
                candidates: Callable[[int, list], tuple[torch.Tensor, torch.Tensor]],
                covering: Callable[[int, list], torch.Tensor]) -> list[tuple[int, int]]:
    """The one spawning rule, in training and at test time, over `frames` (the
    frames seen whole, in order) for a clip already holding the asked points.
    The first frame is filled, the farthest first from every point seen there
    and from each other, until the clip has `first` points: no gap minimum,
    the farthest-first order spreads them evenly. Each later frame adds new
    surface: a candidate farther than `far` (pixels) from every point seen
    there, the farthest first, until the clip has `total`. `candidates(t,
    born)` gives frame t's candidate pixels (C, 2) and their ids (C,);
    `covering(t, born)` the pixels (M, 2) of the points seen in frame t, the
    asked ones included; both are told the points added so far (`born`), and
    the counts are of points added, so `first` and `total` are given net of
    the asked points. Returns the added points as (frame, id)."""
    born = []
    for k, t in enumerate(frames):
        room, gap = (first, 0.0) if k == 0 else (total, far)
        if len(born) >= room:
            continue
        px, ids = candidates(t, born)
        if not len(ids):
            continue
        cover = covering(t, born)
        d = (torch.cdist(px, cover).min(1).values if len(cover)
             else torch.full((len(ids),), float("inf"), device=px.device))
        born += [(t, int(ids[j])) for j in farthest_first(px, d, gap, room - len(born))]
    return born


def first_frame_share(total: int, reserve: int, asked: int) -> tuple[int, int]:
    """(first, total) for `fill_frames`, net of the `asked` points: the first
    frame filled up to `total - reserve` points, the later frames up to
    `total`, `reserve` kept for new surface."""
    return max(total - reserve - asked, 0), max(total - asked, 0)
