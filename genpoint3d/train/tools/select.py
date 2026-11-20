"""Where a frame shows surface that the stored world's points do not explain: the cells a stored point covers."""

import torch
import torch.nn.functional as F

from genpoint3d.train.tools.camera import project
from genpoint3d.train.tools.farthest_first import SURFACE_TOL

# The cells around a stored point's own that it explains: a corner of a body's mesh within them needs no new point.
NEAR = 2


def cell_of(uv: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(N,) the cell (row-major, of an h x w map) each place `uv` (N, 2) in [0, 1] across the frame falls in."""
    return (uv[:, 1] * h).long().clamp(0, h - 1) * w + (uv[:, 0] * w).long().clamp(0, w - 1)


def covered(pos: torch.Tensor, proj: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """(h, w) the cells of a frame that the points at `pos` (N, 3) explain: within NEAR cells of where one is
    drawn by `proj` (3, 4), it not behind the surface the frame shows there (`depth` (h, w), metres) by more
    than SURFACE_TOL of it."""
    h, w = depth.shape
    m = torch.zeros(h * w, device=depth.device)
    if len(pos):
        uv, z = (x[0, 0] for x in project(pos.float()[None, None], proj.float()[None, None]))
        inside = (uv >= 0).all(-1) & (uv < 1).all(-1) & (z > 0)
        cell = cell_of(uv, h, w)
        front = inside & (z <= depth.flatten()[cell] * (1 + SURFACE_TOL))
        m[cell[front]] = 1.0
    return F.max_pool2d(m.view(1, 1, h, w), 2 * NEAR + 1, stride=1, padding=NEAR)[0, 0] > 0


def apart(order: torch.Tensor, room: int, h: int, w: int) -> list[int]:
    """Up to `room` of the cells `order` (C,), best first, each more than NEAR cells from those taken before."""
    taken, out = torch.zeros(h, w, dtype=torch.bool, device=order.device), []
    for c in order.tolist():
        if len(out) >= room:
            break
        r, k = divmod(c, w)
        if taken[r, k]:
            continue
        out.append(c)
        taken[max(r - NEAR, 0):r + NEAR + 1, max(k - NEAR, 0):k + NEAR + 1] = True
    return out
