"""Track a clip a frame at a time: the engine writes each frame, the model puts right what it sees there.

In each frame the model looks `rounds` times at the points the world has, each
round correcting its last guess, and pulling each point toward the place that
keeps its body's shape as strongly as it says the body is stiff. What it ends
on is stored (`World.store`), and the engine writes the next frame from it and
from what the model says of the bodies: which are free to move, how heavy, and
where they touch (`World.ahead`). A frame without its image is looked at the
same way: the model sees nothing there, and the engine's frame stands unless
the model knows better. Nothing goes back over a frame: the past changes only
through the world's bodies."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from genpoint3d.tools.traj_scale import GRAVITY_REF
from genpoint3d.train.tools.engine import pulled
from genpoint3d.train.tools.farthest_first import GAP, UNIT
from genpoint3d.train.tools.regress import _angle, _clock, _corrected, _error, _group_error, _handed_over, _rays, _rigid_error
from genpoint3d.train.tools.visibility import visibility_error
from genpoint3d.train.tools.world import World

# A round's loss counts this much of the next one's (TAPIP3D's).
ROUND_DECAY = 0.8
# Two bodies may touch where their points are within this many point spacings.
REACH = 1.5


@dataclass
class Look:
    """One round of one frame, over every point of the clip (those not `alive` hold what the world has)."""
    t: int
    i: int                          # the round in the frame
    alive: torch.Tensor             # (N,) the points the world has by this frame
    g: torch.Tensor                 # (1, N, 3) the guess after the round, with its graph
    logits: torch.Tensor            # (1, N) is each point seen
    groups: torch.Tensor            # (1, N, E) group vectors
    down: torch.Tensor              # (1, 3) which way is down
    arrow: torch.Tensor             # (1, N, 3) to each point's body's centre of gravity
    ahead: Optional[torch.Tensor]   # (1, N, 3) the engine's next frame, after the frame's last round (none after the clip's last)


class Tracker:
    """Every round of every frame, in order, as `Look`s. `tracks`, `vis`,
    `groups`, `down` and `arrow` then hold the clip as the world has it, each
    frame's visibility logits, each point's last group vector, which way is
    down and each frame's arrows."""

    def __init__(self, model: nn.Module, anchor: torch.Tensor, T: int, rounds: int, cond: dict) -> None:
        if anchor.shape[0] != 1:
            raise ValueError("a clip is tracked alone: its points come and go with its own frames")
        self.model, self.anchor, self.T, self.rounds = model, anchor, T, rounds
        self.cond = dict(cond)
        if self.cond.get("visual_mask") is None:
            self.cond["visual_mask"] = torch.ones(1, T, 2, 2, dtype=torch.bool, device=anchor.device)
        N = anchor.shape[1]
        self.world = World(anchor[0], cond["query_t"][0], _handed_over(self.cond)[0], T)
        self.vis = torch.zeros(1, T, N, device=anchor.device)
        self.groups = torch.zeros(1, N, model.group_out.out_features, device=anchor.device)
        self.known = torch.zeros(1, N, dtype=torch.bool, device=anchor.device)
        self.arrow = torch.zeros(1, T, N, 3, device=anchor.device)
        self.rays = _rays(self.cond, anchor)
        # Which way is down: as told, else as the model last said once it has.
        self.told = cond["down_known"]
        self.down = cond["down"].float()
        self.down_known = self.told

    @property
    def tracks(self) -> torch.Tensor:
        """(1, T, N, 3) the clip as the world has it, in the guess's units."""
        return self._guess(self.world.pos)[None]

    def _guess(self, pos: torch.Tensor) -> torch.Tensor:
        return (pos - self.anchor[0].float()) / self.model.traj_scale

    def _fall(self) -> torch.Tensor:
        """(3,) gravity a frame squared in the clip's unit, along which way is down as told or last said."""
        return (self.cond["gravity"].float()[:, None] * GRAVITY_REF * self.down)[0]

    def _reach(self, t: int) -> torch.Tensor:
        """(N,) how near another body's point must be to each point in frame `t` for the two to touch maybe:
        REACH times the points' spacing on the surface there (`farthest_first.spacing`), in the clip's unit."""
        P = self.cond["fine_proj"][0, t].float()
        unit = P[2, :3].norm()                                                   # metres a unit
        ahead = P[2, :3] / unit
        focal = (P[0, :3] - (P[0, :3] @ ahead) * ahead).norm() / unit            # in frame widths
        z = self.world.pos[t] @ P[2, :3] + P[2, 3]                               # metres
        return REACH * (GAP * z / focal).clamp_min(UNIT) / unit

    def _ahead(self, t: int, said, idx: torch.Tensor) -> torch.Tensor:
        """(N, 3) the engine's frame `t` + 1 as a guess, from the world as stored and what the model says of
        its bodies: each one's freedom and mass, the mean over its points, and where they touch."""
        w = self.world
        free = heavy = touch = None
        if len(w.seed):
            mine = w._members()[:, idx]                                                       # (K, n)
            total = mine.sum(-1)
            over = lambda x: (mine @ x[0]) / total.clamp_min(1e-6)
            free, heavy = over(said.free), torch.where(total > 0, over(said.heavy), 1.0)
            n, m = w.touching(t, self._reach(t))
            local = torch.full((len(w.body),), -1, dtype=torch.long, device=idx.device)
            local[idx] = torch.arange(len(idx), device=idx.device)
            at, slide = w.sliding(t, n, m)
            touch = self.model.touches(said.h[0], local[n], local[m], w.body[n], w.body[m], w.pos[t, m] - w.pos[t, n],
                                       slide, at, self._fall())
        return self._guess(w.ahead(t, self._fall(), free, heavy, touch))

    def __iter__(self):
        m, w, dev = self.model, self.world, self.anchor.device
        N, q, view = self.anchor.shape[1], self.cond["query_t"], self.cond["visual_mask"]
        for t in range(self.T):
            alive = w.alive(t)
            idx = torch.nonzero(alive)[:, 0]
            if not len(idx):
                continue
            anchor = self.anchor[:, idx]
            # What the world has of the frame: the engine's, or a point just given where it is given.
            start = self._guess(w.pos[t])
            before = self._guess(w.pos[max(t - 1, 0)])[idx][None]
            g = start[idx][None]
            logits = self.vis[:, t - 1, idx] if t else torch.zeros(1, len(idx), device=dev)
            seen = torch.zeros(N, device=dev)
            shown = float(view[0, t].any())
            given_now = torch.where(q[:, idx] == t, 0, -1)
            placed = lambda x: w.pos[t].index_copy(0, idx, m.lookup_pos(x.float(), anchor.float())[0])
            for i in range(self.rounds):
                with torch.no_grad():
                    pos = placed(g.detach())
                    place, support = w.rigid(t, pos, seen)
                    off = ((place - pos)[idx] / m.traj_scale)[None]
                    s = support[idx][None, :, None]
                said = m(g.detach(), before, _clock(i, self.rounds, 1, dev), anchor, t, vis_prev=logits,
                         group_prev=self.groups[:, idx], group_known=self.known[:, idx],
                         rigid=torch.cat([off * m.traj_scale, s / (s + 1)], -1),
                         **{**self.cond, "id_card": self.cond["id_card"][:, idx], "query_t": q[:, idx],
                            "down": self.down, "down_known": self.down_known})
                self.down = torch.where(self.told[:, None], self.down, said.down.detach())
                self.down_known = torch.ones_like(self.told)
                own = g.detach() + _corrected(said.step[:, None], given_now, self.rays[:, idx])[:, 0]
                g = pulled(own, g.detach() + off, said.stiff)
                logits = said.logits.detach().float()
                self.groups[:, idx], self.known[:, idx] = said.groups.detach(), True
                seen = torch.zeros(N, device=dev).index_copy(0, idx, torch.sigmoid(logits[0]) * shown)
                ahead = None
                if i == self.rounds - 1:
                    self.vis[:, t, idx], self.arrow[:, t, idx] = logits, said.arrow.detach().float()
                    w.store(t, placed(g.detach()), seen, self.groups[0], placed(g.detach() + said.arrow.detach()))
                    if t + 1 < self.T:
                        ahead = self._ahead(t, said, idx)[None]
                full = lambda x: torch.zeros(1, N, *x.shape[2:], device=dev).index_copy(1, idx, x.float())
                yield Look(t, i, alive, start[None].index_copy(1, idx, g.float()), full(said.logits),
                           self.groups.index_copy(1, idx, said.groups), said.down, full(said.arrow), ahead)


def track_loss(
    model: nn.Module,
    x1: torch.Tensor,
    anchor: torch.Tensor,
    mask: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    rounds: int = 3,
    backward: Optional[float] = None,
    visible: Optional[torch.Tensor] = None,
    vis_weight: float = 0.0,
    pairs: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    rigid_weight: float = 0.0,
    groups: Optional[Callable[[int, int], tuple[torch.Tensor, torch.Tensor]]] = None,
    group_weight: float = 0.0,
    down_true: Optional[torch.Tensor] = None,
    down_scored: Optional[torch.Tensor] = None,
    down_weight: float = 0.0,
    centre_true: Optional[torch.Tensor] = None,
    centre_known: Optional[torch.Tensor] = None,
    centre_weight: float = 0.0,
    ahead_weight: float = 0.0,
    **cond,
) -> tuple[torch.Tensor, dict]:
    """The loss of every round of every frame against the truth `x1` (1, T, N,
    3), a frame's round i weighted ROUND_DECAY^(rounds - 1 - i), normalised,
    and the frames averaged. Per round: the position error (`mask`, `weight`
    (1, T, N)), and with their weights the rigid pairs' gap error, the group
    vectors' error against `groups(t, t + 1)`, which way is down (the angle,
    radians) against `down_true` (1, 3) on a clip `down_scored` (1,), the tip
    of each point's arrow (from the guess, fixed) against its body's centre of
    gravity `centre_true` (1, T, N, 3) on the points `centre_known` (1, N),
    and the visibility logits against `visible`; and after a frame's last
    round, the error of the engine's next frame: what teaches the model what
    it says of the bodies. With `backward`, each round's loss times it is
    back-propagated at once (one round in memory) and the loss comes back
    detached. The dict also holds the tracks, the visibility logits, each
    point's last group vector, which way is down, the arrows, and the arrows'
    loss alone (in the total)."""
    run = Tracker(model, anchor, x1.shape[1], rounds, cond)
    w = [ROUND_DECAY ** (rounds - 1 - i) for i in range(rounds)]
    w = [v / sum(w) for v in w]
    share = 1 / x1.shape[1]
    total = vis_total = centre_total = x1.new_zeros(())
    for r in run:
        keep = r.alive[None]
        at = lambda x: None if x is None else x[:, r.t]
        loss = w[r.i] * _error(r.g, at(x1), at(mask) & keep, at(weight))
        if rigid_weight:
            idx, ok = pairs
            ok = ok & keep[:, :, None] & keep.gather(1, idx.flatten(1)).view_as(idx)
            loss = loss + w[r.i] * rigid_weight * _rigid_error(r.g[:, None], at(x1)[:, None], idx, ok)
        if group_weight:
            loss = loss + w[r.i] * group_weight * _group_error(r.groups, *groups(r.t, r.t + 1), keep)
        if centre_weight:
            centre = w[r.i] * centre_weight * _error(r.g.detach() + r.arrow, at(centre_true), centre_known & keep, None)
            centre_total = centre_total + share * centre.detach()
            loss = loss + centre
        if down_weight and down_scored.any():
            loss = loss + w[r.i] * down_weight * (_angle(r.down, down_true.float()) * down_scored).sum() / down_scored.sum()
        if ahead_weight and r.ahead is not None:
            nxt = slice(r.t + 1, r.t + 2)
            loss = loss + ahead_weight * _error(r.ahead[:, None], x1[:, nxt], mask[:, nxt] & keep[:, None],
                                                None if weight is None else weight[:, nxt])
        loss = share * loss
        vis_loss = x1.new_zeros(())
        if visible is not None:
            vis_loss = share * w[r.i] * visibility_error(r.logits[:, None], visible[:, r.t:r.t + 1], keep)
        if backward is not None:
            with torch.autocast(device_type=x1.device.type, enabled=False):
                ((loss + vis_weight * vis_loss) * backward).backward()
            loss, vis_loss = loss.detach(), vis_loss.detach()
        total = total + loss
        vis_total = vis_total + vis_loss
    if backward is None:
        total = total + vis_weight * vis_total
    return total, {"loss": total.detach(), "visibility": vis_total.detach(), "tracks": run.tracks, "logits": run.vis,
                   "groups": run.groups, "down": run.down, "arrows": run.arrow, "centre": centre_total}


@torch.no_grad()
def track(model: nn.Module, anchor: torch.Tensor, num_frames: int, rounds: int, **cond) -> tuple[torch.Tensor, torch.Tensor]:
    """The tracks (1, T, N, 3) and the visibility logits (1, T, N)."""
    run = Tracker(model, anchor, num_frames, rounds, cond)
    for _ in run:
        pass
    return run.tracks, run.vis
