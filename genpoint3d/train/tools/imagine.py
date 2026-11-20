"""Which true point each imagined point is to be: a body's imagined points against its object's answer."""

from typing import Optional

import torch
from scipy.optimize import linear_sum_assignment


class Answer:
    """What a clip's stored world should come to be (`kubric_models.answer`), for training. A body's points
    as it imagines them are a set: each is taught to be the answer's corner it is matched to when imagined,
    the matching that moves them least in all (DETR's), and to stay that corner from then on. One body
    imagines each object: a later body formed on an object already imagined is a piece of it, and none of
    its imagined points is true.

    mesh     (T, V, 3)  every object's corners, in the clip's unit
    body     (V,)       each corner's object, as `segment`
    seen     (T, V)     the corners each frame sees
    segment  (N,)       each of the clip's own points' object, 0 the background, -1 unknown
    things   (O, KNOWN) what Kubric says of each object (`WorldModel.KNOWN`), object 1 first
    moves    (O,)       the objects known free to move; the floor (object 0) is fixed
    bars     (E, 2)     the corners joined by a bar
    """

    def __init__(self, mesh: torch.Tensor, body: torch.Tensor, seen: torch.Tensor, segment: torch.Tensor,
                 things: torch.Tensor, moves: torch.Tensor, bars: torch.Tensor) -> None:
        self.mesh, self.body, self.seen, self.segment, self.things = mesh.float(), body, seen, segment, things.float()
        self.moves = moves
        V = max(len(body), 1)
        self.bar_keys = torch.cat([bars[:, 0] * V + bars[:, 1], bars[:, 1] * V + bars[:, 0]]) if len(bars) else bars.new_zeros(0)
        # Each imagined point's corner, in the order the points are imagined, and the objects imagined so far.
        self.corner = torch.zeros(0, dtype=torch.long, device=mesh.device)
        self.claimed: set[int] = set()

    def match(self, seed: int, t: int, at: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """A body imagines points at `at` (M, 3) in frame `t`; `seed` is the point the body is formed about.
        Which of them are true points (m,), and each one's corner (m,): every corner of the seed's object
        (the floor's, for the background) gets the imagined point nearest it, no two the same; none for an
        object another body imagined already. None where the seed's object is not known."""
        if seed >= len(self.segment) or self.segment[seed] < 0 or not len(self.body):
            return None
        thing = int(self.segment[seed])
        mine = torch.nonzero(self.body == thing)[:, 0] if thing not in self.claimed else self.body[:0]
        self.claimed.add(thing)
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

    def free(self, seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Whether the object of each body formed about the points `seed` (K,) is free to move, (K,) 0 or 1,
        and (K,) the bodies that is known of: the floor's (fixed) and those of objects that move."""
        thing = torch.where(seed < len(self.segment), self.segment[seed.clamp_max(len(self.segment) - 1)], -1)
        moving = torch.zeros_like(thing, dtype=torch.bool)
        on = (thing >= 1) & (thing <= len(self.moves))
        moving[on] = self.moves[thing[on] - 1]
        return moving.float(), (thing == 0) | moving

    def joined(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(P,) whether corners `a` and `b` (P,) are joined by a bar."""
        return torch.isin(a * max(len(self.body), 1) + b, self.bar_keys)

    def at(self, t: int, n: int) -> torch.Tensor:
        """(n, 3) where the first `n` imagined points truly are in frame `t`."""
        return self.mesh[t, self.corner[:n]]

    def shown(self, t: int, n: int) -> torch.Tensor:
        """(n,) which of the first `n` imagined points frame `t` truly sees."""
        return self.seen[t, self.corner[:n]]
