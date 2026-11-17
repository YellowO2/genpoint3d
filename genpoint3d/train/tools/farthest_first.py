"""A point stands for the smallest piece of surface the camera can tell apart, never smaller than UNIT."""

from typing import Callable

import torch

GAP = 0.05          # the piece the camera tells apart, a share of the frame's width
UNIT = 0.05         # the smallest piece, metres
SURFACE_TOL = 0.03  # a point is on the surface its pixel shows within this share of the depth there


def spacing(z: torch.Tensor, focal: float, width: float, gap: float) -> torch.Tensor:
    """Pixels between points on surface `z` metres from the camera."""
    return (UNIT * focal / z.clamp_min(1e-6)).clamp_min(gap * width)


def add_points(frames: list[int], room: int,
               candidates: Callable[[int, list], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
               covering: Callable[[int, list], torch.Tensor]) -> list[tuple[int, int]]:
    """Up to `room` new points as (frame, id): in each frame, a candidate with
    no point within its spacing, the emptiest place first. `candidates(t,
    born)` gives pixels (C, 2), ids (C,) and spacings (C,); `covering(t,
    born)` the pixels (M, 2) of the points seen there."""
    born = []
    for t in frames:
        px, ids, far = candidates(t, born)
        if not len(ids):
            continue
        cover = covering(t, born)
        d = (torch.cdist(px, cover).min(1).values if len(cover) else torch.full_like(far, float("inf"))) / far
        while len(born) < room:
            j = int(d.argmax())
            if d[j] <= 1:
                break
            born.append((t, int(ids[j])))
            d = torch.minimum(d, (px - px[j]).norm(dim=-1) / far)
    return born
