"""Track a clip a frame at a time: the engine writes each frame, the model puts right what it sees there.

In each frame the model looks `rounds` times at the points the world has, each
round correcting its last guess, and pulling each point toward the place that
keeps its body's shape as strongly as it says the body is stiff. What it ends
on is stored (`World.store`), and the engine writes the next frame from it and
from what the model says of the bodies: which are free to move, how heavy, and
where they touch (`World.ahead`). A frame with none of its image (a forecast,
a gap) is the engine's alone: nothing is looked at, said or taught there, and
the model is not called; the touches use what the model last knew of each
point. A frame with part of its image is looked at as any other. Nothing goes
back over a frame: the past changes only through the world's bodies.

Each body has one description, what the model knows it to be, kept with the
world: put right after every frame from the body's points and told to each of
them in the next (`WorldModel.describe`). What a body is comes from it: how
free and how heavy, what Kubric says of the object, and the whole of its
points, all sides, which a body first found imagines from it
(`WorldModel.imagine`); from the next frame those are the world's as any other
point, each put right every frame by what the model then knows.
With the clip's `Answer` (training) the points kept are the true ones; without
it, those the model says are there."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from genpoint3d.tools.traj_scale import GRAVITY_REF
from genpoint3d.train.tools.engine import pulled
from genpoint3d.train.tools.farthest_first import GAP, UNIT
from genpoint3d.train.tools.imagine import Answer
from genpoint3d.train.tools.kabsch import MIN_POSE_POINTS
from genpoint3d.train.tools.regress import _angle, _clock, _corrected, _error, _group_error, _handed_over, _rays, _rigid_error
from genpoint3d.train.tools.batch import whole_frames
from genpoint3d.train.tools.visibility import visibility_error
from genpoint3d.train.tools.world import World

# A round's loss counts this much of the next one's (TAPIP3D's).
ROUND_DECAY = 0.8
# Two bodies may touch where their points are within this many point spacings.
REACH = 1.5


@dataclass
class Born:
    """What one body imagined of itself when first found."""
    at: torch.Tensor                # (M, 3) where, in the clip's unit, with its graph
    there: torch.Tensor             # (M,) logits: is there such a point
    kept: torch.Tensor              # (m,) the ones the world then has
    corner: Optional[torch.Tensor]  # (m,) the answer's corner each of those is, with the clip's `Answer`


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
    born: list                      # what each body first found in the frame imagined, after its last round
    known: Optional[torch.Tensor] = None   # (K, KNOWN) what the bodies' descriptions say of their objects, after the frame's last round
    seed: Optional[torch.Tensor] = None    # (K,) the point each of those bodies is formed about

    def first(self, n: int) -> "Look":
        """The look at the clip's first `n` points: those it came with, not the ones added since."""
        cut = lambda x: None if x is None else x[:, :n]
        return Look(self.t, self.i, self.alive[:n], cut(self.g), cut(self.logits), cut(self.groups), self.down, cut(self.arrow),
                    cut(self.ahead), self.born, self.known, self.seed)


class Tracker:
    """Every round of every frame, in order, as `Look`s. `tracks`, `vis`,
    `groups`, `down` and `arrow` then hold the clip as the world has it, each
    frame's visibility logits, each point's last group vector, which way is
    down and each frame's arrows. With `add`, points are added as the clip
    goes: before a frame is looked at, `add(t, pos, alive)` gives the ones to
    add there (`score.Adder`), after the clip's own. A model that imagines
    (`slots`) adds each new body's imagined points after those, the true ones
    by `answer`."""

    def __init__(self, model: nn.Module, anchor: torch.Tensor, T: int, rounds: int, cond: dict,
                 add: Optional[Callable] = None, answer: Optional[Answer] = None) -> None:
        if anchor.shape[0] != 1:
            raise ValueError("a clip is tracked alone: its points come and go with its own frames")
        self.model, self.anchor, self.T, self.rounds, self.add, self.answer = model, anchor, T, rounds, add, answer
        self.cond = dict(cond)
        if self.cond.get("visual_mask") is None:
            self.cond["visual_mask"] = torch.ones(1, T, 2, 2, dtype=torch.bool, device=anchor.device)
        N = anchor.shape[1]
        self.world = World(anchor[0], cond["query_t"][0], _handed_over(self.cond)[0], T)
        self.vis = torch.zeros(1, T, N, device=anchor.device)
        self.groups = torch.zeros(1, N, model.group_out.out_features, device=anchor.device)
        self.known = torch.zeros(1, N, dtype=torch.bool, device=anchor.device)
        self.arrow = torch.zeros(1, T, N, 3, device=anchor.device)
        # Each body's description, as last put right, and what the model last knew of each point.
        self.what = torch.zeros(0, model.dim, device=anchor.device)
        self.h = torch.zeros(N, model.dim, device=anchor.device)
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

    def _described(self, said, idx: torch.Tensor) -> torch.Tensor:
        """(K, D) each body's description put right from the frame just stored, with its graph; kept, without."""
        w = self.world
        old = torch.cat([self.what, torch.zeros(len(w.seed) - len(self.what), self.what.shape[1], device=idx.device)])
        what = self.model.describe(said.h[0], w._members()[:, idx], old)
        self.what = what.detach()
        return what

    def _ahead(self, t: int, h: torch.Tensor, idx: torch.Tensor, what: torch.Tensor) -> torch.Tensor:
        """(N, 3) the engine's frame `t` + 1 as a guess, from the world as stored and what the model says of
        its bodies: each one's freedom and mass, from its description `what` (K, D), and where they touch,
        from what it knows of the points `idx`, `h` (n, D)."""
        w = self.world
        free = heavy = touch = None
        if len(w.seed):
            free, heavy = self.model.of_body(what)
            n, m = w.touching(t, self._reach(t))
            local = torch.full((len(w.body),), -1, dtype=torch.long, device=idx.device)
            local[idx] = torch.arange(len(idx), device=idx.device)
            at, slide = w.sliding(t, n, m)
            touch = self.model.touches(h, local[n], local[m], w.body[n], w.body[m], w.pos[t, m] - w.pos[t, n],
                                       slide, at, self._fall())
        return self._guess(w.ahead(t, self._fall(), free, heavy, touch))

    def _more(self, t: int, anchor: torch.Tensor, card: torch.Tensor) -> None:
        """Room for more points after the clip's others, given in frame `t` at `anchor` (M, 3) with the look
        `card` (M, F): nothing said of them yet."""
        anchor, card = anchor[None], card[None]
        M, c, dev = anchor.shape[1], self.cond, anchor.device
        more = lambda x, dim=1: torch.cat([x, torch.zeros(*x.shape[:dim], M, *x.shape[dim + 1:], dtype=x.dtype, device=dev)], dim)
        here = torch.full((1, M), t, dtype=torch.long, device=dev)
        self.rays = torch.cat([self.rays, _rays({**c, "query_t": here}, anchor)], 1)
        self.anchor = torch.cat([self.anchor, anchor.to(self.anchor.dtype)], 1)
        c["id_card"], c["query_t"] = torch.cat([c["id_card"], card.to(c["id_card"].dtype)], 1), torch.cat([c["query_t"], here], 1)
        self.vis, self.arrow = more(self.vis, 2), more(self.arrow, 2)
        self.groups, self.known = more(self.groups), more(self.known)
        self.h = more(self.h[None])[0]

    def _added(self, t: int) -> None:
        """The points `add` gives for frame `t`, put after the clip's others: each given there. It is shown the
        points frames gave, not the imagined ones: those have no look, and take none of its room."""
        real = ~self.world.imagined
        new = self.add(t, self.world.pos[t][real], self.world.alive(t)[real])
        if new is None:
            return
        self._more(t, *new)
        self.world.add(new[0], t)

    def _imagined(self, t: int, what: torch.Tensor, idx: torch.Tensor) -> list[Born]:
        """Frame `t` stored, each body first found in it imagines the whole of its points (`WorldModel.imagine`)
        from its description (`what` (K, D)): about its centre, in its own size, the mean way of its seen
        points from the centre.
        The ones kept are the world's from now, with no look: the true ones by the clip's `Answer`, or
        without it those said to be there."""
        w, born = self.world, []
        for k in torch.nonzero(w.first == t)[:, 0].tolist():
            mine = torch.nonzero((w.body[idx] == k) & ~w.imagined[idx])[:, 0]
            if len(mine) < MIN_POSE_POINTS:
                continue
            with torch.autocast(device_type=idx.device.type, enabled=False):
                centre = w.centre[t, k]
                way = w.pos[t, idx[mine]] - centre
                size = way.norm(dim=-1).mean().clamp_min(1e-6)
            out, there = self.model.imagine(what[k])
            at = centre + out * size
            if self.answer is None:
                kept, corner = torch.nonzero(there > 0)[:, 0], None
            else:
                hit = self.answer.match(int(w.seed[k]), t, at.detach())
                if hit is None:
                    continue
                kept, corner = hit
            born.append(Born(at, there, kept, corner))
            if len(kept):
                new = at[kept].detach()
                self._more(t, new, torch.zeros(len(kept), self.cond["id_card"].shape[-1], device=new.device))
                w.imagine(k, new, t)
        return born

    def _as_body(self) -> None:
        """An imagined point is fed its body's group vector, its seed's: it is on that body whatever it says."""
        w = self.world
        self.groups[0, w.imagined] = self.groups[0, w.seed[w.body[w.imagined]]]
        self.known[0, w.imagined] = True

    def __iter__(self):
        m, w, dev = self.model, self.world, self.anchor.device
        view = self.cond["visual_mask"]
        for t in range(self.T):
            if self.add is not None:
                self._added(t)
            N, q = self.anchor.shape[1], self.cond["query_t"]
            alive = w.alive(t)
            idx = torch.nonzero(alive)[:, 0]
            if not len(idx):
                continue
            if not view[0, t].any():
                # None of the frame's image: the engine's alone, the model not called.
                if t + 1 < self.T:
                    with torch.no_grad():
                        self._ahead(t, self.h[idx], idx, self.what)
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
            imagined = w.imagined[idx][None]
            self._as_body()
            on = w.body[idx] >= 0
            what = (self.what[w.body[idx].clamp_min(0)] * on[:, None] if len(self.what)
                    else torch.zeros(len(idx), self.what.shape[1], device=dev))[None]
            placed = lambda x: w.pos[t].index_copy(0, idx, m.lookup_pos(x.float(), anchor.float())[0])
            for i in range(self.rounds):
                with torch.no_grad():
                    pos = placed(g.detach())
                    place, support = w.rigid(t, pos, seen)
                    off = ((place - pos)[idx] / m.traj_scale)[None]
                    s = support[idx][None, :, None]
                said = m(g.detach(), before, _clock(i, self.rounds, 1, dev), anchor, t, vis_prev=logits,
                         group_prev=self.groups[:, idx], group_known=self.known[:, idx],
                         rigid=torch.cat([off * m.traj_scale, s / (s + 1)], -1), imagined=imagined, what=what,
                         **{**self.cond, "id_card": self.cond["id_card"][:, idx], "query_t": q[:, idx],
                            "down": self.down, "down_known": self.down_known})
                self.down = torch.where(self.told[:, None], self.down, said.down.detach())
                self.down_known = torch.ones_like(self.told)
                own = g.detach() + _corrected(said.step[:, None], given_now, self.rays[:, idx])[:, 0]
                g = pulled(own, g.detach() + off, said.stiff)
                logits = said.logits.detach().float()
                self.groups[:, idx], self.known[:, idx] = said.groups.detach(), True
                self._as_body()
                seen = torch.zeros(N, device=dev).index_copy(0, idx, torch.sigmoid(logits[0]) * shown)
                ahead, born, known, seed = None, [], None, None
                full = lambda x: torch.zeros(1, N, *x.shape[2:], device=dev).index_copy(1, idx, x.float())
                look = (start[None].index_copy(1, idx, g.float()), full(said.logits), self.groups.index_copy(1, idx, said.groups),
                        said.down, full(said.arrow))
                if i == self.rounds - 1:
                    self.vis[:, t, idx], self.arrow[:, t, idx] = logits, said.arrow.detach().float()
                    self.h[idx] = said.h[0].detach().float()
                    w.store(t, placed(g.detach()), seen, self.groups[0], placed(g.detach() + said.arrow.detach()))
                    now = self._described(said, idx)
                    known, seed = m.known(now), w.seed.clone()
                    if t + 1 < self.T:
                        ahead = self._ahead(t, said.h[0], idx, now)[None]
                    if m.slots:
                        born = self._imagined(t, now, idx)
                yield Look(t, i, alive, *look, ahead, born, known, seed)


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
    add: Optional[Callable] = None,
    answer: Optional[Answer] = None,
    imagine_weight: float = 0.0,
    known_weight: float = 0.0,
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
    detached. With `add`, points are added as the clip goes (`Tracker`); the
    loss and what comes back are of the clip's own points. With the clip's
    `answer` and `imagine_weight`, what the bodies imagine is taught too, a
    mean of its own beside the clip's points': when imagined, how far each
    true point is from its corner and whether each is said to be there; from
    then on, as any point, its place, the engine's next frame and whether it
    is seen. With `known_weight`, what each body's description says of its
    object (`WorldModel.known`) against what Kubric says of it, after every
    frame. The dict also holds
    the tracks, the visibility logits, each point's last group vector, which
    way is down, the arrows, and the arrows' loss alone (in the total)."""
    run = Tracker(model, anchor, x1.shape[1], rounds, cond, add, answer)
    own = x1.shape[2]
    shown = whole_frames(run.cond["visual_mask"])[0]
    to_guess = lambda at, n: ((at - run.anchor[0, own:own + n].float()) / model.traj_scale)[None]
    w = [ROUND_DECAY ** (rounds - 1 - i) for i in range(rounds)]
    w = [v / sum(w) for v in w]
    share = 1 / x1.shape[1]
    total = vis_total = centre_total = x1.new_zeros(())
    for r in run:
        every, r = r, r.first(own)
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
        vis_loss = x1.new_zeros(())
        if visible is not None:
            vis_loss = share * w[r.i] * visibility_error(r.logits[:, None], visible[:, r.t:r.t + 1], keep)
        n = every.g.shape[1] - own
        if imagine_weight and answer is not None and n:
            all_n = torch.ones(1, n, dtype=torch.bool, device=x1.device)
            loss = loss + w[r.i] * imagine_weight * _error(every.g[:, own:], to_guess(answer.at(r.t, n), n), all_n, None)
            if ahead_weight and every.ahead is not None:
                loss = loss + ahead_weight * imagine_weight * _error(every.ahead[:, own:], to_guess(answer.at(r.t + 1, n), n), all_n, None)
            if visible is not None:
                vis_loss = vis_loss + share * w[r.i] * imagine_weight * visibility_error(
                    every.logits[:, None, own:], (answer.shown(r.t, n) & shown[r.t])[None, None], all_n)
        if imagine_weight and answer is not None:
            for b in every.born:
                true = torch.zeros_like(b.there)
                true[b.kept] = 1.0
                loss = loss + imagine_weight * torch.nn.functional.binary_cross_entropy_with_logits(b.there, true)
                if len(b.kept):
                    loss = loss + imagine_weight * (b.at[b.kept] - answer.mesh[r.t, b.corner]).norm(dim=-1).mean() / model.traj_scale
        if known_weight and answer is not None and every.known is not None and len(every.seed):
            true, ok = answer.known(every.seed)
            if ok.any():
                loss = loss + known_weight * (every.known - true).abs()[ok].mean()
        loss = share * loss
        if backward is not None:
            with torch.autocast(device_type=x1.device.type, enabled=False):
                ((loss + vis_weight * vis_loss) * backward).backward()
            loss, vis_loss = loss.detach(), vis_loss.detach()
        total = total + loss
        vis_total = vis_total + vis_loss
    if backward is None:
        total = total + vis_weight * vis_total
    return total, {"loss": total.detach(), "visibility": vis_total.detach(), "tracks": run.tracks[:, :, :own],
                   "logits": run.vis[:, :, :own], "groups": run.groups[:, :own], "down": run.down,
                   "arrows": run.arrow[:, :, :own], "centre": centre_total, "world": run.world, "what": run.what}


@torch.no_grad()
def track(model: nn.Module, anchor: torch.Tensor, num_frames: int, rounds: int, **cond) -> tuple[torch.Tensor, torch.Tensor]:
    """The tracks (1, T, N, 3) and the visibility logits (1, T, N)."""
    run = Tracker(model, anchor, num_frames, rounds, cond)
    for _ in run:
        pass
    own = anchor.shape[1]
    return run.tracks[:, :, :own], run.vis[:, :, :own]
