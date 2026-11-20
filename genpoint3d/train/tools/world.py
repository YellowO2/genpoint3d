"""The stored world of one clip: its points and the rigid bodies they are on, kept once, played for any frame."""

from functools import wraps
from typing import Optional

import torch

from genpoint3d.train.tools.engine import Body, Touch, fitted, mass_spread, step
from genpoint3d.train.tools.kabsch import MIN_POSE_POINTS, turn_matrix, turn_vector
from genpoint3d.train.tools.likeness import SAME, likeness


def _exact(method):
    """`method` in fp32 whatever autocast its caller runs under: bf16 holds a place to about a part in 256."""
    @wraps(method)
    def run(self, *args, **kwargs):
        with torch.autocast(device_type=self.pos.device.type, enabled=False):
            return method(self, *args, **kwargs)
    return run


class World:
    """`pos` (T, N, 3) is every point in every frame, as the world now has it.
    A frame is first written by the engine (`ahead`), then put right by what is
    seen there (`store`); what is learned later changes the frames before it
    through the bodies: a point new to a body gets its past from the body's,
    and a body seen again after frames unseen has those frames put right.
    A body's points are those seen on it and those imagined of it (`imagine`):
    an imagined point is on the body that imagined it for good, never counts
    as seen, and its place on the body is what was last said of it.
    All in the clip's own unit, nothing here learned."""

    def __init__(self, anchor: torch.Tensor, query: torch.Tensor, given: torch.Tensor, T: int) -> None:
        """anchor (N, 3) each point where it is given, in its query frame `query` (N,); `given` (N,) the points
        the world is ever told of."""
        N, dev = len(anchor), anchor.device
        self.query, self.given = query, given
        self.pos = anchor.float()[None].repeat(T, 1, 1)
        # Each point's body (-1: none), where it is on it (about the centre, unturned) and how like the body's seed it is.
        self.body = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.place = torch.zeros(N, 3, device=dev)
        self.like = torch.zeros(N, device=dev)
        # The points a body imagined of itself, not given by any frame, and the bars joining pairs of them.
        self.imagined = torch.zeros(N, dtype=torch.bool, device=dev)
        self.bars = torch.zeros(0, 2, dtype=torch.long, device=dev)
        # Each body's pose in every frame, and how it goes in the frame last stored.
        self.turn, self.centre = torch.zeros(T, 0, 3, 3, device=dev), torch.zeros(T, 0, 3, device=dev)
        self.speed, self.spin = torch.zeros(0, 3, device=dev), torch.zeros(0, 3, device=dev)
        # The point each body's others are like, its first frame, the last frame its pose was fitted to seen
        # points, and how many frames have said where its centre of gravity is.
        self.seed, self.first, self.backed, self.told = (torch.zeros(0, dtype=torch.long, device=dev) for _ in range(4))

    def add(self, anchor: torch.Tensor, t: int) -> None:
        """More points, given in frame `t` at `anchor` (M, 3)."""
        M, dev = len(anchor), anchor.device
        self.pos = torch.cat([self.pos, anchor.float()[None].repeat(len(self.pos), 1, 1)], 1)
        self.query = torch.cat([self.query, torch.full((M,), t, dtype=torch.long, device=dev)])
        self.given = torch.cat([self.given, torch.ones(M, dtype=torch.bool, device=dev)])
        self.body = torch.cat([self.body, torch.full((M,), -1, dtype=torch.long, device=dev)])
        self.place = torch.cat([self.place, torch.zeros(M, 3, device=dev)])
        self.like = torch.cat([self.like, torch.zeros(M, device=dev)])
        self.imagined = torch.cat([self.imagined, torch.zeros(M, dtype=torch.bool, device=dev)])

    @_exact
    @torch.no_grad()
    def imagine(self, k: int, at: torch.Tensor, t: int, bars: Optional[torch.Tensor] = None) -> None:
        """More points of body `k`, imagined in frame `t` at `at` (M, 3): on it from now, and in every other
        frame where the body's pose there puts them; `bars` (E, 2) the pairs of them joined, by their order."""
        N = len(self.body)
        if bars is not None:
            self.bars = torch.cat([self.bars, bars.to(self.bars) + N])
        self.add(at, t)
        new = torch.arange(N, N + len(at), device=at.device)
        self.body[new], self.like[new], self.imagined[new] = k, 1.0, True
        self.place[new] = (self.turn[t, k].T @ (at.float() - self.centre[t, k])[..., None])[..., 0]
        self.pos[:, new] = (self.turn[:, k, None] @ self.place[new][None, :, :, None])[..., 0] + self.centre[:, k, None]
        self.pos[t, new] = at.float()

    def alive(self, t: int) -> torch.Tensor:
        """(N,) the points the world has by frame `t`."""
        return self.given & (self.query <= t)

    def _members(self) -> torch.Tensor:
        """(K, N) each body's points, each by its likeness to the seed."""
        return self.like * (self.body == torch.arange(len(self.seed), device=self.body.device)[:, None])

    def poses(self, t: int, pos: torch.Tensor, seen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Each body's pose in frame `t`, (K, 3, 3) and (K, 3): the one that best keeps its shape on its points
        `pos` (N, 3) believed seen (`seen` (N,), 0 to 1), or with under MIN_POSE_POINTS of them seen (their
        count, (K,)), the one the engine stored."""
        seen = seen * ~self.imagined
        w = self._members() * seen
        count = ((w > 0) * seen).sum(-1)
        turn, centre = fitted(self.place[None], pos[None], w)
        ok = count >= MIN_POSE_POINTS
        return torch.where(ok[:, None, None], turn, self.turn[t]), torch.where(ok[:, None], centre, self.centre[t]), count

    @_exact
    def rigid(self, t: int, pos: torch.Tensor, seen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, 3) the place that keeps each point's body's shape in frame `t` (its own `pos` for a point on no
        body), and (N,) how many of the body's points are seen behind it."""
        if not len(self.seed):
            return pos, torch.zeros_like(seen)
        turn, centre, count = self.poses(t, pos, seen)
        on, k = self.body >= 0, self.body.clamp_min(0)
        there = (turn[k] @ self.place[..., None])[..., 0] + centre[k]
        return torch.where(on[:, None], there, pos), count[k] * on

    @_exact
    @torch.no_grad()
    def store(self, t: int, pos: torch.Tensor, seen: torch.Tensor, groups: torch.Tensor, tips: torch.Tensor) -> None:
        """Frame `t` as it is seen: the points at `pos` (N, 3), each body's pose from those believed seen
        (`seen` (N,)), each point on the body its group vector (`groups` (N, E)) is like, each body's centre of
        gravity where its points' arrows point (`tips` (N, 3)), and how each body now goes."""
        alive = self.alive(t)
        self.pos[t] = torch.where(alive[:, None], pos.float(), self.pos[t])
        if len(self.seed):
            turn, centre, count = self.poses(t, self.pos[t], seen)
            backed = count >= MIN_POSE_POINTS
            self._put_right(t, turn, centre, backed)
            self.turn[t], self.centre[t] = turn, centre
            self.backed = torch.where(backed, t, self.backed)
            # An imagined point is on its body where it was last said to be.
            said = self.imagined & alive
            k = self.body[said]
            self.place[said] = (turn[k].transpose(-1, -2) @ (self.pos[t, said] - centre[k])[..., None])[..., 0]
        self._group(t, groups.float(), alive)
        self._find_centres(t, tips.float(), seen)
        if t:
            old = self.first < t
            self.speed = torch.where(old[:, None], self.centre[t] - self.centre[t - 1], 0.0)
            self.spin = torch.where(old[:, None], turn_vector(self.turn[t] @ self.turn[t - 1].transpose(-1, -2)), 0.0)

    def _put_right(self, t: int, turn: torch.Tensor, centre: torch.Tensor, backed: torch.Tensor) -> None:
        """A body seen again in frame `t` after frames unseen was carried through them by the engine alone:
        what its pose is off by now, dealt back over those frames, growing evenly from the last frame it was
        seen in, and its points moved with it."""
        for k in torch.nonzero(backed & (self.backed < t - 1) & (self.first < t - 1))[:, 0].tolist():
            last = int(self.backed[k])
            share = (torch.arange(last + 1, t, device=turn.device) - last)[:, None] / (t - last)       # (S, 1)
            twist = turn_matrix(share * turn_vector(turn[k] @ self.turn[t, k].T))                     # (S, 3, 3)
            shift = share * (centre[k] - self.centre[t, k])
            mine = self.body == k
            was = self.pos[last + 1:t, mine] - self.centre[last + 1:t, k, None]
            self.pos[last + 1:t, mine] = (twist[:, None] @ was[..., None])[..., 0] + (self.centre[last + 1:t, k] + shift)[:, None]
            self.turn[last + 1:t, k] = twist @ self.turn[last + 1:t, k]
            self.centre[last + 1:t, k] += shift

    def _group(self, t: int, groups: torch.Tensor, alive: torch.Tensor) -> None:
        """Each point the world has on the body whose seed its group vector is like, or with others like it in
        none a new body (MIN_POSE_POINTS of them: fewer cannot hold a pose, and stay on none). A point new
        to a body is placed on it as it is now, and its frames before it was given are the body's."""
        K, T, dev = len(self.seed), len(self.pos), groups.device
        to, like = torch.full_like(self.body, -1), torch.zeros_like(self.like)
        if K:
            every = likeness(groups[:, None], groups[self.seed][None])                               # (N, K)
            like, to = every.max(1)
            to = torch.where(like > SAME, to, -1)
        # An imagined point stays on the body that imagined it.
        to, like = torch.where(self.imagined, self.body, to), torch.where(self.imagined, self.like, like)
        loose = alive & (to < 0)
        new = []
        while loose.any():
            seed = int(torch.nonzero(loose)[0])
            near = likeness(groups, groups[seed])
            take = loose & (near > SAME)
            take[seed] = True
            loose &= ~take
            if take.sum() >= MIN_POSE_POINTS:
                to[take], like[take] = K + len(new), near[take]
                new.append((seed, self.pos[t, take].mean(0)))
        if new:
            # A new body is unturned where its points are, and has been there: nothing says how it came.
            centre = torch.stack([c for _, c in new])
            self.turn = torch.cat([self.turn, torch.eye(3, device=dev).expand(T, len(new), 3, 3)], 1)
            self.centre = torch.cat([self.centre, centre.expand(T, len(new), 3)], 1)
            self.speed, self.spin = (torch.cat([x, torch.zeros(len(new), 3, device=dev)]) for x in (self.speed, self.spin))
            self.seed = torch.cat([self.seed, torch.tensor([s for s, _ in new], device=dev)])
            self.first, self.backed = (torch.cat([x, torch.full((len(new),), t, device=dev)]) for x in (self.first, self.backed))
            self.told = torch.cat([self.told, torch.zeros(len(new), dtype=torch.long, device=dev)])
        moved = alive & (to != self.body)
        self.body = torch.where(alive, to, self.body)
        self.like = torch.where(alive, like, self.like)
        joins = moved & (to >= 0)
        k = to[joins]
        self.place[joins] = (self.turn[t, k].transpose(-1, -2) @ (self.pos[t, joins] - self.centre[t, k])[..., None])[..., 0]
        past = (self.turn[:, k] @ self.place[joins][None, :, :, None])[..., 0] + self.centre[:, k]            # (T, n, 3)
        before = torch.arange(T, device=dev)[:, None] < self.query[joins][None]
        self.pos[:, joins] = torch.where(before[..., None], past, self.pos[:, joins])

    def _find_centres(self, t: int, tips: torch.Tensor, seen: torch.Tensor) -> None:
        """Each body's centre of gravity moved toward where its seen points' arrows point in frame `t`: the
        mean of what every frame so far has said. The poses and places are about the centre, so they move
        with it; no point does."""
        if not len(self.seed):
            return
        w = self._members() * seen * ~self.imagined                                                   # (K, N)
        said = w.sum(-1) > 0
        k = self.body.clamp_min(0)
        on_body = (self.turn[t, k].transpose(-1, -2) @ (tips - self.centre[t, k])[..., None])[..., 0]   # (N, 3)
        off = (w @ on_body) / w.sum(-1, keepdim=True).clamp_min(1e-6)
        self.told += said
        off = off * (said / self.told.clamp_min(1))[:, None]
        self.centre += (self.turn @ off[None, :, :, None])[..., 0]
        self.place -= off[k] * (self.body >= 0)[:, None]

    @_exact
    def touching(self, t: int, reach: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The pairs of points that may be where two bodies touch in frame `t`: each point with the nearest
        point of another body, within the point's `reach` (N,). Two (C,) lists of points, the first on the
        body of the lower number."""
        idx = torch.nonzero(self.alive(t) & (self.body >= 0))[:, 0]
        if len(idx) < 2:
            return idx[:0], idx[:0]
        p, b = self.pos[t, idx], self.body[idx]
        d = torch.cdist(p, p).masked_fill(b[:, None] == b[None], float("inf"))
        gap, other = d.min(1)
        n, m = idx[gap < reach[idx]], idx[other[gap < reach[idx]]]
        swap = self.body[n] > self.body[m]
        pairs = torch.unique(torch.stack([torch.where(swap, m, n), torch.where(swap, n, m)], 1), dim=0)
        return pairs[:, 0], pairs[:, 1]

    @_exact
    def sliding(self, t: int, n: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Midway between each pair of points `n`, `m` (C,) in frame `t`, (C, 3), and how the two's bodies go
        past each other there, as they now go, (C, 3)."""
        at = (self.pos[t, n] + self.pos[t, m]) / 2
        goes = lambda k: self.speed[k] + torch.linalg.cross(self.spin[k], at - self.centre[t, k])
        return at, goes(self.body[n]) - goes(self.body[m])

    @_exact
    def ahead(self, t: int, gravity: torch.Tensor, free: Optional[torch.Tensor] = None,
              heavy: Optional[torch.Tensor] = None, touch: Optional[Touch] = None) -> torch.Tensor:
        """(N, 3) where the engine puts every point in frame `t` + 1, written there as the world's until that
        frame is seen: a point on a body moves and turns with it (`engine.step`: gravity (3,) a frame squared,
        and what the model says of the bodies, `free` and `heavy` (K,) and where they `touch`, all free and
        alike without); a point on none keeps its speed. The graph runs through what the model said."""
        K, dev = len(self.seed), self.pos.device
        now = self.pos[t]
        before = self.pos[t - 1] if t else now
        out = now + (now - before) * (self.query < t)[:, None]
        if K:
            free = torch.ones(K, device=dev) if free is None else free.float()
            heavy = torch.ones(K, device=dev) if heavy is None else heavy.float()
            spread = mass_spread(self.place[None], self._members())
            size = spread.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
            spread = spread + (1e-3 * size + 1e-12) * torch.eye(3, device=dev)
            went = step(Body(self.centre[t], self.turn[t], self.speed, self.spin), gravity.float(), free, heavy, spread, touch)
            on, k = self.body >= 0, self.body.clamp_min(0)
            carried = (turn_matrix(went.spin)[k] @ (now - self.centre[t, k])[..., None])[..., 0] + went.centre[k]
            out = torch.where(on[:, None], carried, out)
            self.turn[t + 1], self.centre[t + 1] = went.turn.detach(), went.centre.detach()
            self.speed, self.spin = went.speed.detach(), went.spin.detach()
        self.pos[t + 1] = torch.where(self.alive(t)[:, None], out.detach(), self.pos[t + 1])
        return out
