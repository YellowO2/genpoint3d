"""Pick new points' places, the farthest from what is there first."""

import torch

# A new point starts only this far (a share of the frame's width) from every
# point tracked and seen there, in training (`ClipDataset._spawned`) and at
# test time (`spawn_points`).
SPAWN_GAP = 0.05


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
