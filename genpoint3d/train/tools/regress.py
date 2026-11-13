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
import torch.nn.functional as F
from torch import nn

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
    return view.flatten(2).all(-1).gather(1, q)


def _window_points(q: torch.Tensor, seen: torch.Tensor, start: int, end: int, every: bool):
    """For one window, with `q` (B, N) each point's query frame: `alive` (B, N),
    the points it has, only points handed over (`seen`): those first seen by
    its end, or all once an earlier sweep has given them a past (`every`).
    `at` (B, N), each point's query frame in the window, -1 when outside it."""
    inside = (q >= start) & (q < end)
    alive = seen if every else seen & (q < end)
    at = torch.where(inside, q - start, torch.full_like(q, -1))
    return alive, at


def rigid_proposal(pos: torch.Tensor, q: torch.Tensor, anchor: torch.Tensor, groups: torch.Tensor,
                   member: torch.Tensor, seen: torch.Tensor, start: int, end: int, mates: int = 64,
                   robust: int = 0) -> torch.Tensor:
    """(B, L, N, 5) for frames `start` to `end`: where moving with its piece
    would put each point, as an offset from its guess there, how much backs it
    (0 for nothing, towards 1), and how far its piece has turned since the
    point's query frame (radians: how stale its look there is).

    pos (B, T, N, 3) every guess, anchor's space; q (B, N) query frames;
    groups (B, N, E) the group vectors, member (B, N) the points that count;
    seen (B, T, N) the belief each point is visible. For each point, its
    `mates` likeliest group-mates, weighted by that likelihood and by being
    seen in the frame: the turn and shift that best carry them from where they
    are in its query frame to where they are now (a weighted Kabsch fit), a
    shift alone with little behind it, applied to its anchor. With `robust`,
    that many refits, each mate's weight cut by how far it sat from the last
    fit against the typical mate, so a few wrong mates cannot drag it. In
    float32: the fit's SVD has no bfloat16 on the GPU."""
    with torch.autocast(device_type=pos.device.type, enabled=False):
        R, ms, md, s = _rigid_fit(pos.float(), q, anchor.float(), groups.float(), member, seen.float(), start, end,
                                  mates, robust)
        return _proposal(pos.float()[:, start:end], anchor.float(), R, ms, md, s)


def rigid_motion(pos: torch.Tensor, q: torch.Tensor, anchor: torch.Tensor, groups: torch.Tensor,
                 member: torch.Tensor, seen: torch.Tensor, mates: int = 64, robust: int = 0) -> tuple:
    """`rigid_proposal`'s fit in every frame of the clip, as its pieces: R (B,
    T, N, 3, 3) the turn of each point's piece since its query frame, ms and
    md (B, T, N, 3) the mates' centre there and now, s (B, T, N) the weight
    behind it. Where moving with its piece puts a point: R (anchor - ms) + md."""
    with torch.autocast(device_type=pos.device.type, enabled=False):
        return _rigid_fit(pos.float(), q, anchor.float(), groups.float(), member, seen.float(), 0, pos.shape[1],
                          mates, robust)


def _proposal(pos_w, anchor, R, ms, md, s):
    """`rigid_proposal`'s (B, L, N, 5) from the fit's pieces in those frames."""
    prop = (R @ (anchor[:, None, :, :, None] - ms[..., None]))[..., 0] + md
    off = (prop - pos_w) * (s > 1e-3)[..., None]
    turn = torch.acos(((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    return torch.cat([off, (s / (s + 1))[..., None], turn[..., None]], -1)


def _rigid_fit(pos, q, anchor, groups, member, seen, start, end, mates, robust=0):
    B, T, N, _ = pos.shape
    K = min(mates, N - 1)
    a = 2 * torch.sigmoid(-_sq_dist(groups)) * (member[:, :, None] & member[:, None, :])
    a = a.masked_fill(torch.eye(N, dtype=torch.bool, device=pos.device), -1.0)
    aff, idx = a.topk(K, dim=-1)                                                    # (B, N, K)
    aff = aff.clamp_min(0)
    take = lambda x: x.gather(2, idx.reshape(B, 1, N * K, *([1] * (x.dim() - 3))).expand(
        B, x.shape[1], N * K, *x.shape[3:])).reshape(B, x.shape[1], N, K, *x.shape[3:])
    w0 = aff[:, None] * take(seen[:, start:end])                                    # (B, L, N, K)
    dst = take(pos[:, start:end])                                                   # (B, L, N, K, 3)
    src = pos[torch.arange(B, device=pos.device)[:, None, None], q[:, :, None], idx][:, None]   # (B, 1, N, K, 3)
    R, ms, md, s, w = *_fit(w0, src, dst), w0
    for _ in range(robust):
        # Mates far from where the fit puts them, against the typical one, count less.
        left = _left(R, ms, md, src, dst)
        typical = (w * left).sum(-1, keepdim=True) / s.clamp_min(1e-6)[..., None] + 1e-6
        w = w0 / (1 + (left / typical).pow(2))
        R, ms, md, s = _fit(w, src, dst)
    return R, ms, md, s


def _fit(w, src, dst):
    """The weighted Kabsch fit carrying `src` onto `dst` (B, L, N, K, 3, src
    broadcast): its turn R (B, L, N, 3, 3), the centres ms and md, and the
    weight s behind it; a shift alone with less than 3."""
    s = w.sum(-1)
    mean = lambda x: (w[..., None] * x).sum(-2) / s.clamp_min(1e-6)[..., None]
    ms, md = mean(src.expand_as(dst)), mean(dst)
    H = torch.einsum("blnk,blnkx,blnky->blnxy", w, src - ms[..., None, :], dst - md[..., None, :])
    U, _, Vh = torch.linalg.svd(H)
    V, Ut = Vh.transpose(-1, -2), U.transpose(-1, -2)
    flip = torch.ones(*H.shape[:-1], device=w.device)
    flip[..., 2] = torch.linalg.det(V @ Ut).sign()
    # The turn carries no gradient (an SVD's is unstable where mates lie nearly
    # in a line); the weights reach the fit through its centres.
    R = torch.where((s >= 3)[..., None, None], V @ torch.diag_embed(flip) @ Ut, torch.eye(3, device=w.device)).detach()
    return R, ms, md, s


def _left(R, ms, md, src, dst):
    """(B, L, N, K) how far each mate sits from where the fit puts it."""
    a = src - ms[..., None, :]
    return ((R[..., None, :, :] @ a[..., None])[..., 0] + md[..., None, :] - dst).norm(dim=-1)


# A piece's fit counts as supported with this much weight behind it (3 mates,
# backing s / (s + 1) of 0.75).
SUPPORTED = 3.0


def object_paths(R, ms, md, s, anchor, acc):
    """(B, T, N, 3) where each point's piece carries it, anchor's space, in the
    frames its fit is not `SUPPORTED`, and (B, T, N) where it does so: from
    the piece's last two supported frames its centre's velocity and its spin,
    then each frame its acceleration and change of spin `acc` (B, T, N, 6, the
    clip's unit and radians per frame squared, what the point says of its
    piece) added, the point carried rigidly with it. None before the piece has
    had two supported frames. R, ms, md, s as `rigid_motion` gives them."""
    B, T, N = s.shape
    dev = s.device
    sup = s >= SUPPORTED
    eye = torch.eye(3, device=dev).expand(B, N, 3, 3)
    zero = torch.zeros(B, N, 3, device=dev)
    pick = lambda m, a, b: torch.where(m.view(*m.shape, *([1] * (a.dim() - m.dim()))), a, b)
    R1 = R2 = Rk = eye
    ms1 = ms2 = md1 = md2 = ref = c = v = w = zero
    t1 = t2 = torch.zeros(B, N, device=dev)
    have = torch.zeros(B, N, dtype=torch.long, device=dev)
    on = torch.zeros(B, N, dtype=torch.bool, device=dev)
    paths, oks = [], []
    for k in range(T):
        sk = sup[:, k]
        # The last two supported frames.
        R1, ms1, md1, t1 = pick(sk, R2, R1), pick(sk, ms2, ms1), pick(sk, md2, md1), pick(sk, t2, t1)
        R2, ms2, md2 = pick(sk, R[:, k], R2), pick(sk, ms[:, k], ms2), pick(sk, md[:, k], md2)
        t2 = pick(sk, torch.full_like(t2, float(k)), t2)
        have = torch.where(sk, (have + 1).clamp(max=2), have)
        on = on & ~sk
        # Carried from the frame after the last supported one: the centre the
        # point's own mates had there, and how it and the turn were moving.
        begin = ~sk & ~on & (have >= 2)
        gap = (t2 - t1).clamp_min(1)[..., None]
        c1 = (R1 @ (ms2 - ms1)[..., None])[..., 0] + md1
        turn = _so3_log(R2 @ R1.transpose(-1, -2)) / gap
        ref, c, v = pick(begin, ms2, ref), pick(begin, md2, c), pick(begin, (md2 - c1) / gap, v)
        Rk, w = pick(begin, R2, Rk), pick(begin, turn, w)
        on = on | begin
        a = acc[:, k].float()
        v = pick(on, v + a[..., :3], v)
        c = pick(on, c + v, c)
        w = pick(on, w + a[..., 3:], w)
        Rk = pick(on, _so3_exp(w) @ Rk, Rk)
        paths.append((Rk @ (anchor.float() - ref)[..., None])[..., 0] + c)
        oks.append(on)
    return torch.stack(paths, 1), torch.stack(oks, 1)


def _skew(w: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def _so3_exp(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) a turn as axis times angle -> (..., 3, 3) its rotation."""
    th = w.norm(dim=-1)[..., None, None]
    K = _skew(w)
    ts = th.clamp_min(1e-4)
    a = torch.where(th < 1e-4, 1 - th ** 2 / 6, torch.sin(ts) / ts)
    b = torch.where(th < 1e-4, 0.5 - th ** 2 / 24, (1 - torch.cos(ts)) / ts ** 2)
    return torch.eye(3, device=w.device) + a * K + b * (K @ K)


def _so3_log(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) a rotation -> (..., 3) its axis times angle (angles below pi)."""
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    th = torch.acos(cos)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    k = torch.where(th < 1e-4, torch.full_like(th, 0.5), th / (2 * torch.sin(th).clamp_min(1e-8)))
    return k[..., None] * v


def _sq_dist(v: torch.Tensor) -> torch.Tensor:
    """(B, N, E) -> (B, N, N) squared distances."""
    return (v[:, :, None] - v[:, None]).pow(2).sum(-1)


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
    trust: Optional[torch.Tensor] = None     # (B, L, N) logits of trusting the own match, with the blend
    own: Optional[torch.Tensor] = None       # (B, L, N, 3) the own match, before the blend
    obj: Optional[torch.Tensor] = None       # (B, L, N, 3) where the point's piece puts it
    obj_ok: Optional[torch.Tensor] = None    # (B, L, N) where there is one
    phys: Optional[torch.Tensor] = None      # (B, L, N, 3) the carried part of `obj`, with its graph
    phys_ok: Optional[torch.Tensor] = None   # (B, L, N) where `obj` is carried


class Sweeps:
    """Every round of every window of every sweep, in order, as `_Round`s.
    `whole`, `vis`, `groups` and `down` then hold the clip's guess, each
    frame's last visibility logits, each point's last group vector and which
    way is down."""

    def __init__(self, model: nn.Module, anchor: torch.Tensor, T: int, schedule: str,
                 window: int, overlap: int, cond: dict) -> None:
        self.model, self.anchor, self.cond = model, anchor, cond
        self.plan = _schedule(schedule)
        self.R = sum(n for _, n in self.plan)
        self.spans = _windows(T, window, overlap)
        B, N = anchor.shape[:2]
        self.whole = torch.zeros(B, T, N, 3, device=anchor.device)
        # What the head last said about each frame; 0 before it has.
        self.vis = torch.zeros(B, T, N, device=anchor.device)
        self.groups = None
        self.seen = _handed_over(cond)
        self.rays = _rays(cond, anchor)
        # Points that have said a group vector: the others have none yet.
        self.known = torch.zeros(B, N, dtype=torch.bool, device=anchor.device)
        # Each point's earliest frame a window has settled; T for none yet.
        self.first = torch.full((B, N), T, dtype=torch.long, device=anchor.device)
        # How far each point trusts its own match in each frame, as last said;
        # its weight as a mate in the rigid fit (`--fit-blend`).
        self.trust = torch.ones(B, T, N, device=anchor.device)
        # What each point last said of its piece's acceleration and change of
        # spin in each frame (`--physics`).
        self.acc = torch.zeros(B, T, N, 6, device=anchor.device)
        # Which way is down: as told, else as the model last said once it has.
        self.told = cond["down_known"]
        self.down = cond["down"].float()
        self.down_known = self.told

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
                    rigid = motion = None
                    if model.rigid_proposal and self.groups is not None:
                        clip_pos = model.lookup_pos(self.whole, anchor)
                        fit_in = (clip_pos, self.cond["query_t"], anchor, self.groups, self.seen & self.known,
                                  torch.sigmoid(self.vis) * self.trust)
                        if model.physics:
                            # Every frame's fit: the motion is carried from its supported frames.
                            motion = tuple(t.detach() for t in rigid_motion(*fit_in, robust=model.fit_robust))
                            rigid = _proposal(clip_pos[:, start:end].float(), anchor.float(),
                                              *(t[:, start:end] for t in motion))
                        else:
                            rigid = rigid_proposal(*fit_in, start, end, robust=model.fit_robust).detach()
                    step, logits, emb, down, trust, acc = model(
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
                        self.vis[:, start:end] = logits.detach()
                    own = g.detach() + _corrected(step, at, self.rays)
                    rec = {}
                    if trust is not None and rigid is not None:
                        # Where its piece puts it: the fit, where enough mates back it;
                        # with --physics, the piece's motion carried on where not.
                        obj = self.whole[:, start:end] + rigid[..., :3] / model.traj_scale
                        obj_ok = rigid[..., 3] >= SUPPORTED / (SUPPORTED + 1)
                        if acc is not None:
                            clip_acc = torch.cat([self.acc[:, :start], acc.float(), self.acc[:, end:]], 1)
                            paths, carried = object_paths(*motion, anchor, clip_acc)
                            phys = (paths[:, start:end] - anchor[:, None].float()) / model.traj_scale
                            phys_ok = carried[:, start:end] & ~obj_ok
                            obj = torch.where(phys_ok[..., None], phys, obj)
                            obj_ok = obj_ok | phys_ok
                            rec.update(phys=phys, phys_ok=phys_ok)
                            self.acc = self.acc.clone()
                            self.acc[:, start:end] = torch.where(alive[:, None, :, None], acc.detach().float(),
                                                                 self.acc[:, start:end])
                        # Towards it, as far as the point distrusts its own match; never
                        # at its query frame; with `blend_hidden`, only where this round
                        # believes it hidden (a frame without its image included).
                        if model.blend_hidden and logits is not None:
                            obj_ok = obj_ok & (logits.float() < 0)
                        t_w = torch.arange(end - start, device=dev)[None, :, None]
                        p = torch.sigmoid(trust.float())
                        pull = (1 - p) * (obj_ok & (t_w != at[:, None])).float()
                        rec.update(trust=trust, own=own.detach(), obj=obj, obj_ok=obj_ok)
                        own = own + pull[..., None] * (obj - own.detach())
                        self.trust = self.trust.clone()
                        self.trust[:, start:end] = torch.where(alive[:, None], p.detach(), self.trust[:, start:end])
                    g = torch.where(alive[:, None, :, None], own, g.detach())
                    yield _Round(i, start, end, g, logits, emb, alive, down, **rec)
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


def _group_error(emb: torch.Tensor, groups: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the affinity 2 sigmoid(-|v_i - v_j|^2) against the
    pairs that truly move as one (B, N, N), the two kinds averaged apart."""
    v = emb.float()
    a = (2 * torch.sigmoid(-(v[:, :, None] - v[:, None]).pow(2).sum(-1))).clamp(1e-4, 1 - 1e-4)
    off = ~torch.eye(emb.shape[1], dtype=torch.bool, device=emb.device) & alive[:, :, None] & alive[:, None, :]
    pos, neg = groups & off, ~groups & off
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
    groups: Optional[Callable[[int, int], torch.Tensor]] = None,
    group_weight: float = 0.0,
    down_true: Optional[torch.Tensor] = None,
    down_scored: Optional[torch.Tensor] = None,
    down_weight: float = 0.0,
    fit_weight: float = 0.0,
    trust_weight: float = 0.0,
    phys_weight: float = 0.0,
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
    clips `down_scored` (B,), those not told it, the rigid fit's own positions
    from the group vectors the round said (`rigid_proposal`, the guess fixed)
    against the truth, so the groups learn what the fit needs, and the
    visibility head's against
    `visible`. With `backward`, each round's loss times it is back-propagated
    at once (one round in memory) and the loss comes back detached. The dict
    also holds the tracks and visibility logits, as `refine` gives them, each
    point's last group vector and which way is down."""
    run = Sweeps(model, anchor, x1.shape[1], schedule, window, overlap, cond)
    w = [gamma ** (run.R - 1 - i) for i in range(run.R)]
    w = [v / sum(w) for v in w]
    share = 1 / len(run.spans)
    total = vis_total = x1.new_zeros(())
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
            loss = loss + w[r.i] * group_weight * _group_error(r.groups, groups(r.start, r.end), r.alive)
        if fit_weight and r.groups is not None:
            clip = model.lookup_pos(torch.cat([run.whole[:, :r.start], g.detach(), run.whole[:, r.end:]], 1), anchor)
            fit = rigid_proposal(clip, cond["query_t"], anchor, r.groups, run.seen & run.known,
                                 torch.sigmoid(run.vis) * run.trust, r.start, r.end, robust=model.fit_robust)
            placed = g.detach() + fit[..., :3] / model.traj_scale
            loss = loss + w[r.i] * fit_weight * _error(placed, x1_w, mask_w, weight_w)
        if trust_weight and r.trust is not None:
            # Trust the own match where it is nearer the truth than the piece's place.
            right = ((r.own - x1_w).norm(dim=-1) <= (r.obj.detach() - x1_w).norm(dim=-1)).float()
            m = (r.obj_ok & mask_w).float()
            bce = F.binary_cross_entropy_with_logits(r.trust.float(), right, reduction="none")
            loss = loss + w[r.i] * trust_weight * (bce * m).sum() / m.sum().clamp_min(1)
        if phys_weight and r.phys is not None:
            loss = loss + w[r.i] * phys_weight * _error(r.phys, x1_w, mask_w & r.phys_ok, weight_w)
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
                   "logits": run.vis if model.visibility else None, "groups": run.groups, "down": run.down}


@torch.no_grad()
def refine(model: nn.Module, anchor: torch.Tensor, num_frames: int, window: int = 0, overlap: int = 0, schedule: str = "ffbb",
           **cond) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The tracks (B, T, N, 3) and, with a visibility head, the logits (B, T,
    N) of each frame's last window."""
    run = Sweeps(model, anchor, num_frames, schedule, window, overlap, cond)
    for _ in run:
        pass
    return run.whole, run.vis if model.visibility else None
