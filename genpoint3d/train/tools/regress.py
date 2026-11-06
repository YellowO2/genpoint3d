"""Track by refining a guess that starts at "nothing moves".

The clip is covered by windows of `window` frames, each sharing `overlap` with
the one before. Sweeps run over the windows forward or backward (`schedule`,
a run of one letter is one sweep, as many rounds a window as it is long:
"fbfb" one round a window each way twice, "ffbb" two forward then two back;
empty is `iters` rounds forward); a backward sweep visits the windows from
the clip's end, each still in the video's own time: a round is the same task
whichever way the windows are visited, only which neighbour was settled last
differs. Each
round the model gives a correction to each frame's position (`_corrected`);
nothing is held but what is known, a point's pixel in its query frame, where
only its depth may move. Frames no window has reached start moving at the
last settled speed (`_reach`). A frame without its image is an input like any other: what the
visibility head last said about every frame is fed back, so a point that looks
hidden can stop trusting the image there, and look again.
"""

from dataclasses import dataclass
from itertools import groupby
from typing import Optional

import torch
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
                   member: torch.Tensor, seen: torch.Tensor, start: int, end: int, mates: int = 64) -> torch.Tensor:
    """(B, L, N, 4) for frames `start` to `end`: where moving with its piece
    would put each point, as an offset from its guess there, and how much
    backs it (0 for nothing, towards 1).

    pos (B, T, N, 3) every guess, anchor's space; q (B, N) query frames;
    groups (B, N, E) the group vectors, member (B, N) the points that count;
    seen (B, T, N) the belief each point is visible. For each point, its
    `mates` likeliest group-mates, weighted by that likelihood and by being
    seen in the frame: the turn and shift that best carry them from where they
    are in its query frame to where they are now (a weighted Kabsch fit), a
    shift alone with little behind it, applied to its anchor."""
    B, T, N, _ = pos.shape
    K = min(mates, N - 1)
    a = 2 * torch.sigmoid(-_sq_dist(groups.float())) * (member[:, :, None] & member[:, None, :])
    a = a.masked_fill(torch.eye(N, dtype=torch.bool, device=pos.device), -1.0)
    aff, idx = a.topk(K, dim=-1)                                                    # (B, N, K)
    aff = aff.clamp_min(0)
    take = lambda x: x.gather(2, idx.reshape(B, 1, N * K, *([1] * (x.dim() - 3))).expand(
        B, x.shape[1], N * K, *x.shape[3:])).reshape(B, x.shape[1], N, K, *x.shape[3:])
    w = aff[:, None] * take(seen[:, start:end])                                     # (B, L, N, K)
    dst = take(pos[:, start:end])                                                   # (B, L, N, K, 3)
    src = pos[torch.arange(B, device=pos.device)[:, None, None], q[:, :, None], idx][:, None]   # (B, 1, N, K, 3)
    s = w.sum(-1)
    mean = lambda x: (w[..., None] * x).sum(-2) / s.clamp_min(1e-6)[..., None]
    ms, md = mean(src.expand_as(dst)), mean(dst)
    H = torch.einsum("blnk,blnkx,blnky->blnxy", w, src - ms[..., None, :], dst - md[..., None, :])
    U, _, Vh = torch.linalg.svd(H)
    V, Ut = Vh.transpose(-1, -2), U.transpose(-1, -2)
    flip = torch.ones(*H.shape[:-1], device=pos.device)
    flip[..., 2] = torch.linalg.det(V @ Ut).sign()
    R = torch.where((s >= 3)[..., None, None], V @ torch.diag_embed(flip) @ Ut, torch.eye(3, device=pos.device))
    prop = (R @ (anchor[:, None, :, :, None] - ms[..., None]))[..., 0] + md
    off = (prop - pos[:, start:end]) * (s > 1e-3)[..., None]
    return torch.cat([off, (s / (s + 1))[..., None]], -1)


def _sq_dist(v: torch.Tensor) -> torch.Tensor:
    """(B, N, E) -> (B, N, N) squared distances."""
    return (v[:, :, None] - v[:, None]).pow(2).sum(-1)


def _schedule(iters: int, schedule: str) -> list[tuple[str, int]]:
    """The sweeps in order, each ("f" or "b", rounds a window): a window has
    all of a sweep's rounds before the next one starts from it."""
    if schedule:
        if set(schedule) - {"f", "b"} or schedule[0] != "f":
            raise ValueError(f"a schedule is f and b, starting with f: {schedule!r}")
        return [(c, len(list(run))) for c, run in groupby(schedule)]
    return [("f", iters)]


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


class Sweeps:
    """Every round of every window of every sweep, in order, as `_Round`s.
    `whole`, `vis` and `groups` then hold the clip's guess, each frame's last
    visibility logits and each point's last group vector."""

    def __init__(self, model: nn.Module, anchor: torch.Tensor, T: int, iters: int, schedule: str,
                 window: int, overlap: int, cond: dict) -> None:
        self.model, self.anchor, self.cond = model, anchor, cond
        self.plan = _schedule(iters, schedule)
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

    def __iter__(self):
        model, anchor = self.model, self.anchor
        B, N = anchor.shape[:2]
        dev = anchor.device
        done = r0 = 0
        for sweep, n in self.plan:
            back = sweep == "b"
            for start, end in (reversed(self.spans) if back else self.spans):
                if not back:
                    self.whole = _reach(self.whole, done, end)
                g = self.whole[:, start:end]
                alive, at = _window_points(self.cond["query_t"], self.seen, start, end, every=back or r0 > 0)
                frames = None if len(self.spans) == 1 else (start, end)
                before = self.whole[:, start - 1:start] if start else None
                for i in range(r0, r0 + n):
                    vis = self.vis[:, start:end] if model.visibility else None
                    rigid = None
                    if model.rigid_proposal and self.groups is not None:
                        rigid = rigid_proposal(model.lookup_pos(self.whole, anchor), self.cond["query_t"], anchor,
                                               self.groups, self.seen & self.known, torch.sigmoid(self.vis),
                                               start, end).detach()
                    step, logits, emb = model(g.detach(), _clock(i, self.R, B, dev), anchor, alive=alive,
                                              vis_prev=vis, group_prev=self.groups, group_known=self.known,
                                              frames=frames, x_before=before, rigid=rigid, **self.cond)
                    if emb is not None:
                        # A point not yet in the window keeps what it had: nothing, at first.
                        kept = self.groups if self.groups is not None else torch.zeros_like(emb)
                        self.groups = torch.where(alive[..., None], emb.detach(), kept)
                        self.known = self.known | alive
                    if logits is not None:
                        self.vis[:, start:end] = logits.detach()
                    g = torch.where(alive[:, None, :, None], g.detach() + _corrected(step, at, self.rays),
                                    g.detach())
                    yield _Round(i, start, end, g, logits, emb, alive)
                self.whole = torch.cat([self.whole[:, :start], g.detach(), self.whole[:, end:]], dim=1)
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


def _rigid_error(g, x1, pairs):
    """The error of the gap between two points on one rigid object, `pairs`
    (B, N, N), against the true gap: their errors should be the same."""
    e = (g - x1)[:, :, :, None] - (g - x1)[:, :, None]           # (B, T, N, N, 3)
    err = e.norm(dim=-1)
    w = pairs[:, None].to(err.dtype).expand_as(err)
    return (err * w).sum() / w.sum().clamp(min=1e-8)


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
    iters: int = 4,
    gamma: float = 0.8,
    backward: Optional[float] = None,
    visible: Optional[torch.Tensor] = None,
    vis_weight: float = 0.0,
    vel_weight: float = 0.0,
    pairs: Optional[torch.Tensor] = None,
    rigid_weight: float = 0.0,
    groups: Optional[torch.Tensor] = None,
    group_weight: float = 0.0,
    window: int = 0,
    overlap: int = 0,
    schedule: str = "",
    **cond,
) -> tuple[torch.Tensor, dict]:
    """The loss of every round against the truth `x1` (B, T, N, 3), round i
    weighted gamma^(R - 1 - i) (TAPIP3D's 0.8), normalised, and the windows
    averaged. Per round: the position error (`mask`, `weight` (B, T, N)), and
    with their weights the step error, the rigid pairs' gap error, the group
    vectors' error against `groups`, and the visibility head's against
    `visible`. With `backward`, each round's loss times it is back-propagated
    at once (one round in memory) and the loss comes back detached. The dict
    also holds the tracks and visibility logits, as `refine` gives them."""
    run = Sweeps(model, anchor, x1.shape[1], iters, schedule, window, overlap, cond)
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
            pairs_w = pairs & r.alive[:, :, None] & r.alive[:, None, :]
            loss = loss + w[r.i] * rigid_weight * _rigid_error(g, x1_w, pairs_w)
        if group_weight:
            loss = loss + w[r.i] * group_weight * _group_error(r.groups, groups, r.alive)
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
                   "logits": run.vis if model.visibility else None}


@torch.no_grad()
def refine(model: nn.Module, anchor: torch.Tensor, num_frames: int, iters: int = 4, window: int = 0, overlap: int = 0, schedule: str = "",
           **cond) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The tracks (B, T, N, 3) and, with a visibility head, the logits (B, T,
    N) of each frame's last window."""
    run = Sweeps(model, anchor, num_frames, iters, schedule, window, overlap, cond)
    for _ in run:
        pass
    return run.whole, run.vis if model.visibility else None
