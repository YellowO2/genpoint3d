"""Which true point each imagined point is to be: a body's imagined points against its object's answer."""

from typing import Optional

import torch
from scipy.optimize import linear_sum_assignment

from genpoint3d.train.tools.camera import project
from genpoint3d.train.tools.select import cell_of


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
    turns, places (O + 1, T, 3, 3), (O + 1, T, 3) each object's pose in every frame, the floor first
    cells    (T, h, w)  the object each depth cell shows, 0 the floor
    """

    def __init__(self, mesh: torch.Tensor, body: torch.Tensor, seen: torch.Tensor, segment: torch.Tensor,
                 things: torch.Tensor, moves: torch.Tensor, bars: torch.Tensor, turns: torch.Tensor,
                 places: torch.Tensor, cells: torch.Tensor) -> None:
        self.mesh, self.body, self.seen, self.segment, self.things = mesh.float(), body, seen, segment, things.float()
        self.moves, self.turns, self.places, self.cells = moves, turns.float(), places.float(), cells
        V = max(len(body), 1)
        self.bar_keys = torch.cat([bars[:, 0] * V + bars[:, 1], bars[:, 1] * V + bars[:, 0]]) if len(bars) else bars.new_zeros(0)
        # The objects imagined so far.
        self.claimed: set[int] = set()

    @classmethod
    def of(cls, b: dict) -> "Answer":
        """The answer of the one clip in batch `b` (`ClipDataset`'s)."""
        return cls(*(b[k][0] for k in ("mesh", "mesh_body", "mesh_seen", "segment", "things", "moves", "mesh_bars",
                                        "turns", "places", "cell_object")))

    def track(self, t: int, cells: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
        """(T, M, 3) where points given in frame `t` at `at` (M, 3), on the depth cells `cells` (M,), truly are
        in every frame: each moves with the object its cell shows (a point on a rigid object moves exactly as
        it does; the floor stays)."""
        thing = self.cells[t].flatten()[cells].clamp(0, len(self.turns) - 1)
        turn, place = self.turns[thing], self.places[thing]                                   # (M, T, 3, 3), (M, T, 3)
        local = (turn[:, t].transpose(-1, -2) @ (at.float() - place[:, t])[..., None])          # (M, 3, 1)
        return ((turn @ local[:, None])[..., 0] + place).transpose(0, 1)

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
        return torch.as_tensor(slot, device=at.device), mine[torch.as_tensor(corner, device=at.device)]

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

    def corners_seen(self, t: int, proj: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(h * w,) the cells of frame `t` (drawn by `proj` (3, 4)) that show a corner of the answer."""
        out = torch.zeros(h * w, dtype=torch.bool, device=self.mesh.device)
        shown = self.mesh[t, self.seen[t]]
        if len(shown):
            uv, z = (x[0, 0] for x in project(shown[None, None], proj.float()[None, None]))
            inside = (uv >= 0).all(-1) & (uv < 1).all(-1) & (z > 0)
            out[cell_of(uv[inside], h, w)] = True
        return out


