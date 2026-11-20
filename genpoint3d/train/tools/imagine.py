"""Which true point each imagined point is to be: a body's imagined points against its object's answer."""

from typing import Optional

import torch
from scipy.optimize import linear_sum_assignment


class Answer:
    """What a clip's stored world should come to be (`kubric_models.answer`), for training. A body's points
    as it imagines them are a set: each is taught to be the answer's corner it is matched to when imagined,
    the matching that moves them least in all (DETR's), and to stay that corner from then on.

    mesh     (T, V, 3)  every object's corners, in the clip's unit
    body     (V,)       each corner's object, as `segment`
    seen     (T, V)     the corners each frame sees
    segment  (N,)       each of the clip's own points' object, 0 the background, -1 unknown
    things   (O, KNOWN) what Kubric says of each object (`WorldModel.KNOWN`), object 1 first
    """

    def __init__(self, mesh: torch.Tensor, body: torch.Tensor, seen: torch.Tensor, segment: torch.Tensor,
                 things: torch.Tensor) -> None:
        self.mesh, self.body, self.seen, self.segment, self.things = mesh.float(), body, seen, segment, things.float()
        # Each imagined point's corner, in the order the points are imagined.
        self.corner = torch.zeros(0, dtype=torch.long, device=mesh.device)

    def match(self, seed: int, t: int, at: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """A body imagines points at `at` (M, 3) in frame `t`; `seed` is the point the body is formed about.
        Which of them are true points (m,), and each one's corner (m,): every corner of the seed's object
        gets the imagined point nearest it, no two the same. None where the seed's object is not known; the
        background has no corners, so none of its body's imagined points is true."""
        if seed >= len(self.segment) or self.segment[seed] < 0 or not len(self.body):
            return None
        mine = torch.nonzero(self.body == self.segment[seed])[:, 0]
        slot, corner = linear_sum_assignment(torch.cdist(at.float(), self.mesh[t, mine]).cpu().numpy())
        slot, corner = torch.as_tensor(slot, device=at.device), mine[torch.as_tensor(corner, device=at.device)]
        self.corner = torch.cat([self.corner, corner])
        return slot, corner

    def known(self, seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """What Kubric says of the object of each body formed about the points `seed` (K,): (K, KNOWN), and
        (K,) the bodies it says it of: those about one of the clip's own points on an object."""
        thing = torch.where(seed < len(self.segment), self.segment[seed.clamp_max(len(self.segment) - 1)], -1)
        ok = (thing >= 1) & (thing <= len(self.things))
        return self.things[(thing - 1).clamp(0, max(len(self.things) - 1, 0))] if len(self.things) else self.things.new_zeros(len(seed), 0), ok

    def at(self, t: int, n: int) -> torch.Tensor:
        """(n, 3) where the first `n` imagined points truly are in frame `t`."""
        return self.mesh[t, self.corner[:n]]

    def shown(self, t: int, n: int) -> torch.Tensor:
        """(n,) which of the first `n` imagined points frame `t` truly sees."""
        return self.seen[t, self.corner[:n]]
