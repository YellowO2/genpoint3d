"""Track by refining a guess that starts at "nothing moves".

The clip is covered by windows of `window` frames, each sharing `overlap` with
the one before. Sweeps run over the windows forward or backward (`schedule`,
a run of one letter is one sweep, as many rounds a window as it is long:
"fbfb" one round a window each way twice, "ffbb" two forward then two back);
a backward sweep visits the windows from
the clip's end, each still in the video's own time: a round is the same task
whichever way the windows are visited, only which neighbour was settled last
differs. Each round the model gives a correction to each frame's position
(`_corrected`); nothing is held but what is known, a point's pixel in its
query frame, where only its depth may move. Frames no window has reached
start moving at the last settled speed (`_reach`), and a late point's past,
before a backward sweep reaches it, at its earliest (`_reach_back`). A frame
without its image is an input like any other: what the visibility head last
said about every frame is fed back, so a point that looks hidden can stop
trusting the image there, and look again. So is which way is down where the clip does
not say: what the model last said.
"""

from dataclasses import dataclass
from itertools import groupby
from typing import Callable, Optional

import torch
from torch import nn

from genpoint3d.tools.traj_scale import GRAVITY_REF
from genpoint3d.train.tools.batch import whole_frames
from genpoint3d.train.tools.kabsch import MIN_POSE_POINTS, robust_kabsch, turn_matrix, turn_vector
from genpoint3d.train.tools.likeness import SAME, likeness
from genpoint3d.train.tools.visibility import visibility_error


def _windows(T: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) of each window over `T` frames; the last ends at the clip's
    end. A `window` of 0, or one as long as the clip, is the whole clip."""
    if not window or window >= T:
        return [(0, T)]
    out, start = [], 0
    while True:
        start = min(start, T - window)
        out.append((start, start + window))
        if start + window >= T:
            return out
        start += window - overlap


def _cut(t: Optional[torch.Tensor], start: int, end: int) -> Optional[torch.Tensor]:
    return None if t is None else t[:, start:end]


def _reach(g: torch.Tensor, done: int, end: int) -> torch.Tensor:
    """Frames `done` to `end`, which no window has settled, start where the
    point's last settled speed carries it (from the last two settled frames),
    so a correction of zero keeps it moving."""
    if done == 0 or done >= end:
        return g
    last = g[:, done - 1:done]
    speed = last - g[:, done - 2:done - 1] if done >= 2 else torch.zeros_like(last)
    k = torch.arange(1, end - done + 1, device=g.device, dtype=g.dtype)[None, :, None, None]
    return torch.cat([g[:, :done], last + k * speed, g[:, end:]], dim=1)


def _reach_back(g: torch.Tensor, first: torch.Tensor, start: int) -> torch.Tensor:
    """Each point's frames from `start` up to `first` (B, N), the earliest a
    window has settled for it (a late point's past, before a backward sweep
    reaches it), start where its earliest settled speed carries it back."""
    T = g.shape[1]
    f = first.clamp(max=T - 1)
    at = lambda i: g.gather(1, i[:, None, :, None].expand(-1, 1, -1, 3))     # (B, 1, N, 3)
    p0 = at(f)
    speed = torch.where((first + 1 < T)[:, None, :, None], at((f + 1).clamp(max=T - 1)) - p0, torch.zeros_like(p0))
    t = torch.arange(T, device=g.device)[None, :, None]
    fill = (t >= start) & (t < first[:, None]) & (first < T)[:, None]          # (B, T, N)
    back = (first[:, None] - t).to(g.dtype)[..., None]
    return torch.where(fill[..., None], p0 - back * speed, g)


def _corrected(out: torch.Tensor, q: torch.Tensor, ray: torch.Tensor) -> torch.Tensor:
    """(B, L, N, 3) the model's correction to each frame's position; at each
    point's query frame `q` (B, N, an index into the window, -1 outside it),
    where only its pixel is known, along its line of sight `ray` (B, N, 3)
    alone: its depth can be put right, its pixel stays."""
    t = torch.arange(out.shape[1], device=out.device)[None, :, None]
    along = (out * ray[:, None]).sum(-1, keepdim=True) * ray[:, None].to(out.dtype)
    return torch.where((t == q[:, None])[..., None], along, out)


def _rays(cond: dict, anchor: torch.Tensor) -> torch.Tensor:
    """(B, N, 3) each point's line of sight in its query frame: from that
    frame's camera through its anchor, unit length."""
    q = cond["query_t"]
    P = cond["fine_proj"].float().gather(1, q[..., None, None].expand(-1, -1, 3, 4))   # (B, N, 3, 4)
    centre = -torch.linalg.solve(P[..., :3], P[..., 3:])[..., 0]
    return torch.nn.functional.normalize(anchor.float() - centre, dim=-1)


def _handed_over(cond: dict) -> torch.Tensor:
    """(B, N) the points whose query frame is seen whole: the others were never
    given to the model (a partly hidden frame's DINOv3, their id card, would
    see the hidden part)."""
    q, view = cond["query_t"], cond.get("visual_mask")
    if view is None:
        return torch.ones_like(q, dtype=torch.bool)
    return whole_frames(view).gather(1, q)


def _window_points(q: torch.Tensor, seen: torch.Tensor, start: int, end: int, every: bool):
    """For one window, with `q` (B, N) each point's query frame: `alive` (B, N),
    the points it has, only points handed over (`seen`): those first seen by
    its end, or all once an earlier sweep has given them a past (`every`).
    `at` (B, N), each point's query frame in the window, -1 when outside it."""
    inside = (q >= start) & (q < end)
    alive = seen if every else seen & (q < end)
    at = torch.where(inside, q - start, torch.full_like(q, -1))
    return alive, at


def objects(groups: torch.Tensor, member: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The points `member` (B, N) put into objects without chains, so one
    wrong pair cannot merge two: the first point in none seeds one of itself
    and its own group-mates in none (likeness over a half), and so on; one of
    under MIN_POSE_POINTS points, which cannot hold a pose, is left out. (B, K, N) each
    object's points weighted by their likeness to its seed (0 for the others;
    the group vectors' gradient goes through it), and (B, N) each point's
    object, -1 for none."""
    B, N, _ = groups.shape
    g = groups.float()
    a = likeness(g[:, :, None], g[:, None]) * (member[:, :, None] & member[:, None, :])
    which = torch.full((B, N), -1, dtype=torch.long, device=groups.device)
    rows = [[] for _ in range(B)]
    for b in range(B):
        free = member[b].clone()
        while free.any():
            seed = int(torch.nonzero(free)[0])
            take = free & (a[b, seed] > SAME)
            take[seed] = True                                           # always: the loop must end
            free &= ~take
            # Too few points cannot hold a pose: those keep their own guesses and take no room.
            if take.sum() >= MIN_POSE_POINTS:
                which[b, take] = len(rows[b])
                rows[b].append(a[b, seed] * take)
    weight = torch.zeros(B, max(max(len(r) for r in rows), 1), N, device=groups.device)
    for b, r in enumerate(rows):
        if r:
            weight[b, :len(r)] = torch.stack(r)
    return weight, which


def _carried(R: torch.Tensor, shift: torch.Tensor, good: torch.Tensor, centre: torch.Tensor,
             fall: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each object's pose in every frame, (B, T, K, 3, 3) and (B, T, K, 3), and
    (B, T, K) whether it has one: as fitted where it is fitted (`good`), and
    carried on elsewhere, from the nearest fitted frame before (after, for
    frames before the first) as a thrown body moves: its centre of mass
    (`centre` (B, K, 3), in the shape) keeps its speed, and gains `fall` (B,
    3, gravity per frame squared, where the fitted frames show the object
    falling free), and it keeps turning about its centre at its spin. Speed
    and spin come from that frame and the fitted one next to it; with no
    other, the pose holds. A pose places shape point s at R s + shift."""
    B, T, K = good.shape
    dev = R.device
    t = torch.arange(T, device=dev)[None, :, None]
    last = torch.where(good, t, -1).cummax(1).values                                 # (B, T, K) last fitted, -1 none
    nxt = torch.where(good, t, T).flip(1).cummin(1).values.flip(1)                     # next fitted, T none
    has = (last >= 0) | (nxt < T)
    ref = torch.where(last >= 0, last, nxt).clamp(0, T - 1)
    before = torch.cat([torch.full_like(last[:, :1], -1), last[:, :-1]], 1)           # last fitted strictly before
    after = torch.cat([nxt[:, 1:], torch.full_like(nxt[:, :1], T)], 1)                # next fitted strictly after
    at = lambda x, i: x.gather(1, i.clamp(0, T - 1).reshape(B, T, K, *([1] * (x.dim() - 3))).expand_as(x))
    forward = last >= 0
    partner = torch.where(forward, at(before, ref), at(after, ref))
    third = torch.where(forward, at(before, partner), at(after, partner))
    two, three = (partner >= 0) & (partner < T), (partner >= 0) & (partner < T) & (third >= 0) & (third < T)
    C = (R @ centre[:, None, :, :, None])[..., 0] + shift                             # (B, T, K, 3) the centre
    Cr, Cp, C3, Rr = at(C, ref), at(C, partner), at(C, third), at(R, ref)
    tr, tp, t3 = ref.float(), partner.float(), third.float()
    dt = torch.where(two, tr - tp, torch.ones_like(tr))[..., None]
    spin = turn_vector(Rr @ at(R, partner).transpose(-1, -2)) / dt                   # per frame, forward in time
    acc = torch.zeros_like(C)
    if fall is not None:
        # Falling free where the three fitted frames' second divided difference along down is
        # nearer gravity than nothing: an object at rest or rolling on the floor gets none.
        g = fall[:, None, None].expand_as(C)
        gap = lambda a, b: torch.where(three, a - b, torch.ones_like(a))[..., None]
        second = 2 * ((C3 - Cp) / gap(t3, tp) - (Cp - Cr) / gap(tp, tr)) / gap(t3, tr)
        free = three & ((second * g).sum(-1) > 0.5 * (g * g).sum(-1))
        acc = torch.where(free[..., None], g, acc)
    speed = (Cr - Cp) / dt + acc * (tr - (tr + tp) / 2)[..., None]                   # at the frame carried from
    speed, spin = speed * two[..., None], spin * two[..., None]
    k = (t - ref).float()[..., None]
    Rt = turn_matrix(spin * k) @ Rr
    Ct = Cr + speed * k + 0.5 * acc * k * k
    return Rt, Ct - (Rt @ centre[:, None, :, :, None])[..., 0], has


def object_places(pos: torch.Tensor, q: torch.Tensor, groups: torch.Tensor, member: torch.Tensor,
                  seen: torch.Tensor, robust: int = 0, fresh: float = 0.0, tips: Optional[torch.Tensor] = None,
                  fall: Optional[torch.Tensor] = None, passes: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, T, N, 3) each point where its object's pose puts it, and (B, T, N)
    how many of its object's points are seen behind that pose, each counted
    by the belief it is seen (`seen`): 2 points surely seen and 2 half seen
    count 3. The objects are `objects`'; each is kept as one rigid shape in
    which a point's place is fixed through the pose at its query frame. Every
    frame's pose is the one that best carries the shape onto the object's
    points believed seen there (`seen` (B, T, N); a weighted Kabsch fit, refit
    `robust` times), all frames at once, and places every point of it, so a
    point is placed as well as its whole object's pose is known.

    A point backs poses only after its query frame: the pose that gives it its
    place never holds the point itself (one that did would only confirm the
    place it gave, and the shape would drift with its own errors). So the
    shape grows in passes: the object's first points (given at its first
    frame) fit every frame; the points given in a frame that got a pose then
    join through that pose; and so on, each pass placing every joined point
    again through the last pass's poses, until no point joins or moves. With
    `fresh` (degrees), a point counts 1 / (1 + turn / fresh)^2 as much, its
    turn being how far its object has turned since the point was given (by the
    last pass's poses, a frame with too few points holding the one before): a
    tracked point drifts as it turns from the look it was given by (on the
    ball, 0.3 cm off within 15 degrees, 2.5 cm past 90). A frame where under
    MIN_POSE_POINTS seen points back the pose has its pose carried on from the
    fitted frames (`_carried`): not yet tracked, hidden or in the future, it is
    one situation. The object's centre of mass is where its points' arrows
    point (`tips` (B, T, N, 3), each point's guess plus its arrow, in the
    fitted frames), or without them the middle of its shape. A point in no
    object keeps its own guess. The turn and the shape carry no gradient (an
    SVD's is unstable where points lie nearly in a line); the group vectors
    reach the places through the weights' centres.

    pos (B, T, N, 3) every guess; q (B, N) query frames; groups (B, N, E) the
    group vectors, member (B, N) the points that count; fall (B, 3) gravity
    per frame squared in pos's unit, None for none."""
    B, T, N, _ = pos.shape
    dev = pos.device
    with torch.autocast(device_type=dev.type, enabled=False):
        pos, seen = pos.float(), seen.float()
        weight, which = objects(groups, member)                                            # (B, K, N), (B, N)
        K = weight.shape[1]
        on, mine = weight > 0, which.clamp_min(0)
        b, n, t = torch.arange(B, device=dev)[:, None], torch.arange(N, device=dev)[None], torch.arange(T, device=dev)
        per_point = lambda x: x.gather(2, mine.reshape(B, 1, N, *([1] * (x.dim() - 3))).expand(B, T, N, *x.shape[3:]))
        qc = q.clamp(max=T - 1)                                                            # a point asked after T: never given
        at_query = lambda x: x[b, qc, n]                                                   # (B, T, N, ...) -> (B, N, ...)
        backs = (t[None, :, None] > q[:, None]) & (which >= 0)[:, None]
        first = torch.where(on, q[:, None], T).min(-1).values                             # (B, K) each object's first frame
        start = (which >= 0) & (q == first.gather(1, mine)) & (q < T)                      # its first points
        joined, shape, given_at = start, at_query(pos), at_query(pos)                     # given_at: each point at its query
        turn = None
        for _ in range(passes):
            w = seen * backs * joined[:, None]                                             # (B, T, N)
            if fresh and turn is not None:
                cos = (((turn * at_query(turn)[:, None]).sum((-1, -2)) - 1) / 2).clamp(-1, 1)
                w = w / (1 + torch.rad2deg(torch.acos(cos)) / fresh).pow(2)
            Rk, msk, mdk, _ = robust_kabsch(weight[:, None] * w[:, :, None], shape[:, None, None], pos[:, :, None], robust)
            Rk = Rk.detach()
            # (B, T, K) the object's joined points seen in each frame, each by its belief, not by its likeness.
            count = (on[:, None] & (backs & joined[:, None])[:, :, None]).float().mul(seen[:, :, None]).sum(-1)
            ok = count >= MIN_POSE_POINTS
            R, ms, md, ok_n = per_point(Rk), per_point(msk), per_point(mdk), per_point(ok)  # (B, T, N, ...)
            # Each frame's last frame with a pose, for the turn: a frame with too few points holds it.
            last = torch.where(ok_n, t[None, :, None], -1).cummax(1).values.clamp_min(0)
            turn = R.gather(1, last[..., None, None].expand_as(R))
            joins = (which >= 0) & (q < T) & at_query(ok_n) & ~start
            back = (at_query(R).transpose(-1, -2) @ (given_at - at_query(md))[..., None])[..., 0] + at_query(ms)
            placed = torch.where(joins[..., None], back.detach(), shape)
            grew, moved = (joins & ~joined).any(), (placed - shape).norm(dim=-1).max()
            shape, joined = placed, joined | joins
            if not grew and moved < 1e-4:
                break
        shift = mdk - (Rk @ msk[..., None])[..., 0]                                        # (B, T, K, 3)
        body = weight * joined[:, None]                                                     # (B, K, N)
        centre = torch.einsum("bkn,bnc->bkc", body, shape) / body.sum(-1, keepdim=True).clamp_min(1e-6)
        if tips is not None:
            # Each tip in the shape, through its frame's pose, where the frame is fitted and the point seen.
            inside = (R.transpose(-1, -2) @ (tips.float() - per_point(shift))[..., None])[..., 0]   # (B, T, N, 3)
            w = body[:, None] * (seen * ok_n)[:, :, None]                                   # (B, T, K, N)
            total = w.sum((1, 3))[..., None]
            said = torch.einsum("btkn,btnc->bkc", w, inside) / total.clamp_min(1e-6)
            centre = torch.where(total > 0, said, centre)
        Rc, shiftc, has = _carried(Rk, shift, ok, centre, fall)
        placed = joined[:, None] & per_point(has)
        out = torch.where(placed[..., None], (per_point(Rc) @ shape[:, None, :, :, None])[..., 0] + per_point(shiftc), pos)
        return out, per_point(count) * placed


def object_proposal(pos: torch.Tensor, q: torch.Tensor, groups: torch.Tensor, member: torch.Tensor,
                    seen: torch.Tensor, start: int, end: int, robust: int = 0, fresh: float = 0.0,
                    tips: Optional[torch.Tensor] = None, fall: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(B, L, N, 4) for frames `start` to `end`: where its object's pose puts
    each point (`object_places`, fitted only up to `end`: later frames hold
    guesses no round has looked at, and no earlier pose or place uses them), as an
    offset from its guess there (0 for a point in no object), and how much backs
    it, s / (s + 1) for s seen points of its object (under MIN_POSE_POINTS: the
    pose carried on)."""
    places, support = object_places(pos[:, :end], q, groups, member, seen[:, :end], robust, fresh,
                                    None if tips is None else tips[:, :end], fall)
    off = (places - pos[:, :end].float())[:, start:]
    s = support[:, start:, :, None]
    return torch.cat([off, s / (s + 1)], -1)


def _schedule(schedule: str) -> list[tuple[str, int]]:
    """The sweeps in order, each ("f" or "b", rounds a window): a window has
    all of a sweep's rounds before the next one starts from it."""
    if not schedule or set(schedule) - {"f", "b"} or schedule[0] != "f":
        raise ValueError(f"a schedule is f and b, starting with f: {schedule!r}")
    return [(c, len(list(run))) for c, run in groupby(schedule)]


def _clock(i: int, rounds: int, B: int, device) -> torch.Tensor:
    """(B,) `k` for round `i` of `rounds`; a single round is the last."""
    return torch.full((B,), i / (rounds - 1) if rounds > 1 else 1.0, device=device)


@dataclass
class _Round:
    i: int                          # the round, counted over every sweep
    start: int
    end: int
    g: torch.Tensor                 # (B, L, N, 3) the window's guess, before the pin
    logits: Optional[torch.Tensor]  # (B, L, N) visibility, with the head
    groups: Optional[torch.Tensor]  # (B, N, E) group vectors, with their graph
    alive: torch.Tensor             # (B, N)
    down: torch.Tensor              # (B, 3) which way is down, with its graph
    arrow: Optional[torch.Tensor]   # (B, L, N, 3) to each point's object's centre of mass, with the head


class Sweeps:
    """Every round of every window of every sweep, in order, as `_Round`s.
    `whole`, `vis`, `groups`, `down` and `arrow` then hold the clip's guess,
    each frame's last visibility logits, each point's last group vector,
    which way is down and each frame's last arrow to the point's object's
    centre of mass (None without the head)."""

    def __init__(self, model: nn.Module, anchor: torch.Tensor, T: int, schedule: str,
                 window: int, overlap: int, cond: dict) -> None:
        self.model, self.anchor, self.cond = model, anchor, cond
        self.plan = _schedule(schedule)
        self.R = sum(n for _, n in self.plan)
        self.spans = _windows(T, window, overlap)
        B, N = anchor.shape[:2]
        self.whole = torch.zeros(B, T, N, 3, device=anchor.device)
        # What the head last said about each point in each frame, in a round the point was in; 0 before.
        self.vis = torch.zeros(B, T, N, device=anchor.device)
        # Each point's frames some round with it in looked at: the others hold only a guess carried
        # on (`_reach`), or a placeholder's, for a point not yet in the window.
        self.looked = torch.zeros(B, T, N, dtype=torch.bool, device=anchor.device)
        self.groups = None
        self.arrow = torch.zeros(B, T, N, 3, device=anchor.device) if model.centre else None
        self.seen = _handed_over(cond)
        self.rays = _rays(cond, anchor)
        # Points that have said a group vector: the others have none yet.
        self.known = torch.zeros(B, N, dtype=torch.bool, device=anchor.device)
        # Each point's earliest frame a window has settled; T for none yet.
        self.first = torch.full((B, N), T, dtype=torch.long, device=anchor.device)
        # Which way is down: as told, else as the model last said once it has.
        self.told = cond["down_known"]
        self.down = cond["down"].float()
        self.down_known = self.told

    def belief(self) -> torch.Tensor:
        """(B, T, N) how sure the model is that each point is seen in each
        frame, what the rigid fit counts: its visibility head's last word, 0
        where no round with the point in has looked yet (a guess carried on is
        no sighting) and in a frame without its image (a forecast frame: the
        head says whether the point will be in view, not that it was seen)."""
        view = self.cond.get("visual_mask")
        shown = 1.0 if view is None else view.flatten(-2).any(-1)[..., None]               # (B, T, 1)
        return torch.sigmoid(self.vis) * self.looked * shown

    def fall(self) -> torch.Tensor:
        """(B, 3) gravity per frame squared in the clip's unit, along which way
        is down as told or last said (none before either)."""
        return self.cond["gravity"].float()[:, None] * GRAVITY_REF * self.down

    def tips(self, guess: torch.Tensor) -> Optional[torch.Tensor]:
        """(B, T, N, 3) where each point's last arrow points from `guess`, as a
        position (None without the head)."""
        return None if self.arrow is None else self.model.lookup_pos(guess + self.arrow, self.anchor)

    def proposal(self, guess: torch.Tensor, groups: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """`object_proposal` for frames `start` to `end` from the clip's guess
        `guess` (B, T, N, 3) and the group vectors `groups`."""
        m = self.model
        return object_proposal(m.lookup_pos(guess, self.anchor), self.cond["query_t"], groups, self.seen & self.known,
                               self.belief(), start, end, robust=m.fit_robust, fresh=m.fit_fresh,
                               tips=self.tips(guess), fall=self.fall())

    def __iter__(self):
        model, anchor = self.model, self.anchor
        B, N = anchor.shape[:2]
        dev = anchor.device
        done = r0 = 0
        for sweep, n in self.plan:
            back = sweep == "b"
            for start, end in (reversed(self.spans) if back else self.spans):
                self.whole = _reach_back(self.whole, self.first, start) if back else _reach(self.whole, done, end)
                g = self.whole[:, start:end]
                alive, at = _window_points(self.cond["query_t"], self.seen, start, end, every=back or r0 > 0)
                frames = None if len(self.spans) == 1 else (start, end)
                before = self.whole[:, start - 1:start] if start else None
                for i in range(r0, r0 + n):
                    vis = self.vis[:, start:end] if model.visibility else None
                    rigid = None
                    if model.rigid_proposal and self.groups is not None:
                        with torch.no_grad():
                            rigid = self.proposal(self.whole, self.groups, start, end)
                    step, logits, emb, down, arrow = model(
                        g.detach(), _clock(i, self.R, B, dev), anchor, alive=alive, vis_prev=vis,
                        group_prev=self.groups, group_known=self.known, frames=frames, x_before=before,
                        rigid=rigid, **{**self.cond, "down": self.down, "down_known": self.down_known})
                    self.down = torch.where(self.told[:, None], self.down, down.detach())
                    self.down_known = torch.ones_like(self.told)
                    if emb is not None:
                        # A point not yet in the window keeps what it had: nothing, at first.
                        kept = self.groups if self.groups is not None else torch.zeros_like(emb)
                        self.groups = torch.where(alive[..., None], emb.detach(), kept)
                        self.known = self.known | alive
                    if logits is not None:
                        self.vis[:, start:end] = torch.where(alive[:, None], logits.detach(), self.vis[:, start:end])
                    self.looked[:, start:end] |= alive[:, None]
                    if arrow is not None:
                        self.arrow[:, start:end] = arrow.detach().float()
                    own = g.detach() + _corrected(step, at, self.rays)
                    g = torch.where(alive[:, None, :, None], own, g.detach())
                    yield _Round(i, start, end, g, logits, emb, alive, down, arrow)
                self.whole = torch.cat([self.whole[:, :start], g.detach(), self.whole[:, end:]], dim=1)
                self.first = torch.where(alive, self.first.clamp(max=start), self.first)
                if not back:
                    done = end
            r0 += n


def _error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
           weight: Optional[torch.Tensor]) -> torch.Tensor:
    """The weighted mean distance of `pred` from `target`, (B, T, N, 3)."""
    err = (pred - target).norm(dim=-1)
    w = mask.float()
    if weight is not None:
        w = w * weight
    return (err * w).sum() / w.sum().clamp(min=1e-8)


def _motion_error(g, x1, mask, weight):
    """The error of each frame-to-frame step against the truth's, in window
    lengths; a step counts where both its frames do."""
    s = x1.shape[1] - 1
    w = None if weight is None else (weight[:, :-1] + weight[:, 1:]) / 2
    return _error(g.diff(dim=1) * s, x1.diff(dim=1) * s, mask[:, 1:] & mask[:, :-1], w)


def _rigid_error(g, x1, idx, ok):
    """The error of the gap between a point and each of its partners on one
    rigid object, `idx` (B, N, k) where `ok` (B, N, k), against the true gap:
    their errors should be the same."""
    B, T, N, _ = g.shape
    k = idx.shape[-1]
    e = g - x1                                                          # (B, T, N, 3)
    mate = e.gather(2, idx.reshape(B, 1, N * k, 1).expand(B, T, N * k, 3)).reshape(B, T, N, k, 3)
    err = (e[:, :, :, None] - mate).norm(dim=-1)                        # (B, T, N, k)
    w = ok[:, None].to(err.dtype).expand_as(err)
    return (err * w).sum() / w.sum().clamp(min=1e-8)


def _angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B,) the angle in radians between (B, 3) directions, from the sine and
    cosine together: steady near 0, where acos's slope is infinite."""
    sin = torch.linalg.cross(a, b).pow(2).sum(-1).add(1e-12).sqrt()
    return torch.atan2(sin, (a * b).sum(-1))


def _group_error(emb: torch.Tensor, idx: torch.Tensor, same: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the group vectors' `likeness` against whether
    each point and its listed partners `idx` (B, N, m) truly move as one
    (`same`), the two kinds averaged apart."""
    v = emb.float()
    B, N, m = idx.shape
    vj = v.gather(1, idx.reshape(B, N * m, 1).expand(B, N * m, v.shape[-1])).view(B, N, m, -1)
    a = likeness(v[:, :, None], vj).clamp(1e-4, 1 - 1e-4)
    off = alive[:, :, None] & alive.gather(1, idx.flatten(1)).view(B, N, m)
    pos, neg = same & off, ~same & off
    p = -(a.log() * pos).sum() / pos.sum().clamp_min(1)
    n = -((1 - a).log() * neg).sum() / neg.sum().clamp_min(1)
    return (p + n) / 2


def regress_loss(
    model: nn.Module,
    x1: torch.Tensor,
    anchor: torch.Tensor,
    mask: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 0.8,
    backward: Optional[float] = None,
    visible: Optional[torch.Tensor] = None,
    vis_weight: float = 0.0,
    vel_weight: float = 0.0,
    pairs: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    rigid_weight: float = 0.0,
    groups: Optional[Callable[[int, int], tuple[torch.Tensor, torch.Tensor]]] = None,
    group_weight: float = 0.0,
    down_true: Optional[torch.Tensor] = None,
    down_scored: Optional[torch.Tensor] = None,
    down_weight: float = 0.0,
    fit_weight: float = 0.0,
    centre_true: Optional[torch.Tensor] = None,
    centre_known: Optional[torch.Tensor] = None,
    centre_weight: float = 0.0,
    window: int = 0,
    overlap: int = 0,
    schedule: str = "",
    **cond,
) -> tuple[torch.Tensor, dict]:
    """The loss of every round against the truth `x1` (B, T, N, 3), round i
    weighted gamma^(R - 1 - i) (TAPIP3D's 0.8), normalised, and the windows
    averaged. Per round: the position error (`mask`, `weight` (B, T, N)), and
    with their weights the step error, the rigid pairs' gap error, the group
    vectors' error against `groups(start, end)`, the pairs to put together in
    the window's frames (what the window can see and its rigid fit needs),
    which way is down (the angle, radians) against `down_true` (B, 3) on the
    clips `down_scored` (B,), those not told it, the places the objects' poses
    give from the group vectors the round said (`object_proposal`, the guess
    fixed) against the truth, so the groups learn what the fit needs, the tip
    of each point's arrow (from the guess, fixed) against its object's centre
    of mass `centre_true` (B, T, N, 3) on the points `centre_known` (B, N),
    and the visibility head's against `visible`. With `backward`, each round's
    loss times it is back-propagated at once (one round in memory) and the
    loss comes back detached. The dict
    also holds the tracks and visibility logits, as `refine` gives them, each
    point's last group vector, which way is down, each frame's last arrow,
    and the arrows' loss alone (in the total)."""
    run = Sweeps(model, anchor, x1.shape[1], schedule, window, overlap, cond)
    w = [gamma ** (run.R - 1 - i) for i in range(run.R)]
    w = [v / sum(w) for v in w]
    share = 1 / len(run.spans)
    total = vis_total = centre_total = x1.new_zeros(())
    for r in run:
        g, x1_w = r.g, x1[:, r.start:r.end]
        keep = r.alive[:, None].expand(-1, g.shape[1], -1)
        mask_w, weight_w = _cut(mask, r.start, r.end) & keep, _cut(weight, r.start, r.end)
        loss = w[r.i] * _error(g, x1_w, mask_w, weight_w)
        if vel_weight:
            loss = loss + w[r.i] * vel_weight * _motion_error(g, x1_w, mask_w, weight_w)
        if rigid_weight:
            idx, ok = pairs
            ok_w = ok & r.alive[:, :, None] & r.alive.gather(1, idx.flatten(1)).view_as(idx)
            loss = loss + w[r.i] * rigid_weight * _rigid_error(g, x1_w, idx, ok_w)
        if group_weight:
            loss = loss + w[r.i] * group_weight * _group_error(r.groups, *groups(r.start, r.end), r.alive)
        if fit_weight and r.groups is not None:
            fit = run.proposal(torch.cat([run.whole[:, :r.start], g.detach(), run.whole[:, r.end:]], 1), r.groups,
                               r.start, r.end)
            placed = g.detach() + fit[..., :3] / model.traj_scale
            loss = loss + w[r.i] * fit_weight * _error(placed, x1_w, mask_w, weight_w)
        if centre_weight:
            known = centre_known[:, None] & keep
            centre = w[r.i] * centre_weight * _error(g.detach() + r.arrow.float(), centre_true[:, r.start:r.end], known,
                                                     None)
            centre_total = centre_total + share * centre.detach()
            loss = loss + centre
        if down_weight and down_scored.any():
            off = _angle(r.down, down_true.float())
            loss = loss + w[r.i] * down_weight * (off * down_scored).sum() / down_scored.sum()
        loss = share * loss
        vis_loss = x1.new_zeros(())
        if visible is not None:
            vis_loss = share * w[r.i] * visibility_error(r.logits, _cut(visible, r.start, r.end), r.alive)
        if backward is not None:
            with torch.autocast(device_type=x1.device.type, enabled=False):
                ((loss + vis_weight * vis_loss) * backward).backward()
            loss, vis_loss = loss.detach(), vis_loss.detach()
        total = total + loss
        vis_total = vis_total + vis_loss
    if backward is None:
        total = total + vis_weight * vis_total
    return total, {"loss": total.detach(), "visibility": vis_total.detach(), "tracks": run.whole,
                   "logits": run.vis if model.visibility else None, "groups": run.groups, "down": run.down,
                   "arrows": run.arrow, "centre": centre_total}


@torch.no_grad()
def refine(model: nn.Module, anchor: torch.Tensor, num_frames: int, window: int = 0, overlap: int = 0, schedule: str = "ffbb",
           **cond) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The tracks (B, T, N, 3) and, with a visibility head, the logits (B, T,
    N) of each frame's last window."""
    run = Sweeps(model, anchor, num_frames, schedule, window, overlap, cond)
    for _ in run:
        pass
    return run.whole, run.vis if model.visibility else None
