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


def fill_frames(frames: list[int], room: int, far: float,
                candidates: Callable[[int, list], tuple[torch.Tensor, torch.Tensor]],
                covering: Callable[[int, list], torch.Tensor]) -> list[tuple[int, int]]:
    """The one spawning rule, in training and at test time: frame by frame
    over `frames`, up to `room` new points, each where a candidate's place in
    the image is farther than `far` (pixels) from every point covering that
    frame, the farthest first. `candidates(t, born)` gives frame t's candidate
    pixels (C, 2) and their ids (C,); `covering(t, born)` the pixels (M, 2) of
    the points seen in frame t; both are told the points added so far (`born`).
    Returns the added points as (frame, id)."""
    born = []
    for t in frames:
        if len(born) >= room:
            break
        px, ids = candidates(t, born)
        if not len(ids):
            continue
        cover = covering(t, born)
        d = (torch.cdist(px, cover).min(1).values if len(cover)
             else torch.full((len(ids),), float("inf"), device=px.device))
        born += [(t, int(ids[j])) for j in farthest_first(px, d, far, room - len(born))]
    return born
