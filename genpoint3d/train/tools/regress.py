"""Track by refining a guess that starts at "nothing moves".

The clip is covered by windows of `window` frames, each sharing `overlap` with
the one before. Sweeps run over the windows forward or backward (`schedule`,
e.g. "fbfb", one round a window each; empty is `iters` rounds forward). Each
round the model gives a velocity per frame, summed from each point's fixed
frame (`_integrate`), and the point is held at its anchor in its query frame
(`_pin`). A frame without its image is an input like any other: what the
visibility head last said about every frame is fed back, so a point that looks
hidden can stop trusting the image there, and look again.
"""

from dataclasses import dataclass
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
    point was in the last settled frame."""
    if done == 0 or done >= end:
        return g
    return torch.cat([g[:, :done], g[:, done - 1:done].expand(-1, end - done, -1, -1), g[:, end:]], dim=1)


def _integrate(step: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
    """(B, L, N, 3) velocities -> displacement changes, summed out from each
    point's fixed frame `at` (B, N), which does not move: a later frame steps
    from the one before, an earlier one from the one after. Time is in window
    lengths, a frame being 1 / (L - 1)."""
    L = step.shape[1]
    t = torch.arange(L, device=step.device)[None, :, None, None]
    a = at[:, None, :, None]                                                  # (B, 1, N, 1)
    dt = 1.0 / max(L - 1, 1)
    c = torch.cumsum(step * dt, dim=1)                                        # (B, L, N, 3)
    before = torch.cat([torch.zeros_like(c[:, :1]), c[:, :-1]], dim=1)        # c at t - 1
    pick = lambda v, i: v.gather(1, i.clamp(0, L - 1).expand(-1, -1, -1, 3))
    c_a, c_a1 = pick(c, a), pick(before, a)
    return torch.where(t > a, c - c_a, torch.where(t < a, c_a1 - before, torch.zeros_like(c)))


def _window_points(cond: dict, B: int, N: int, start: int, end: int, backward: bool, every: bool):
    """For one window: `alive` (B, N), the points it has (forward, those first
    seen by its end, or all once a backward sweep has run, `every`; backward,
    all), only points whose query frame has its image; `at` (B, N), the frame
    each point's steps run from: its query frame when inside the window, else
    the window's first frame forward and its last backward; `inside` (B, N)."""
    q = cond["query_t"]
    vm = cond.get("visual_mask")
    seen = torch.ones(B, N, dtype=torch.bool, device=q.device)
    if vm is not None:
        seen = (vm[..., None].expand(-1, -1, N) if vm.dim() == 2 else vm).gather(1, q[:, None])[:, 0]
    inside = (q >= start) & (q < end)
    alive = seen if backward or every else seen & (q < end)
    at = torch.where(inside, q - start, torch.full_like(q, end - start - 1 if backward else 0))
    return alive, at, inside


def _pin(g: torch.Tensor, at: torch.Tensor, inside: torch.Tensor) -> torch.Tensor:
    """Hold each point whose query frame is in the window at its anchor
    there: a displacement of zero."""
    if not bool(inside.any()):
        return g
    idx = at[:, None, :, None].expand(-1, 1, -1, 3)
    held = torch.where(inside[:, None, :, None], 0.0, g.gather(1, idx))
    return g.scatter(1, idx, held)


def _schedule(iters: int, schedule: str) -> list[tuple[str, int]]:
    """The sweeps in order, each ("f" or "b", rounds a window)."""
    if schedule:
        if set(schedule) - {"f", "b"} or schedule[0] != "f":
            raise ValueError(f"a schedule is f and b, starting with f: {schedule!r}")
        return [(c, 1) for c in schedule]
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

    def __iter__(self):
        model, anchor = self.model, self.anchor
        B, N = anchor.shape[:2]
        dev = anchor.device
        done = r0 = 0
        for sweep, n in self.plan:
            back = sweep == "b"
            direction = (torch.full((B,), -1.0 if back else 1.0, device=dev)
                         if model.bidirectional else None)
            for start, end in (reversed(self.spans) if back else self.spans):
                if not back:
                    self.whole = _reach(self.whole, done, end)
                g = self.whole[:, start:end]
                alive, at, inside = _window_points(self.cond, B, N, start, end, back, every=r0 > 0)
                frames = None if len(self.spans) == 1 else (start, end)
                before = self.whole[:, start - 1:start] if start else None
                for i in range(r0, r0 + n):
                    vis = self.vis[:, start:end] if model.visibility else None
                    step, logits, emb = model(g.detach(), _clock(i, self.R, B, dev), anchor, alive=alive,
                                              vis_prev=vis, group_prev=self.groups, direction=direction,
                                              frames=frames, x_before=before, **self.cond)
                    if emb is not None:
                        # A point not yet in the window keeps what it had: nothing, at first.
                        kept = self.groups if self.groups is not None else torch.zeros_like(emb)
                        self.groups = torch.where(alive[..., None], emb.detach(), kept)
                    if logits is not None:
                        self.vis[:, start:end] = logits.detach()
                    g = torch.where(alive[:, None, :, None], g.detach() + _integrate(step, at), g.detach())
                    yield _Round(i, start, end, g, logits, emb, alive)
                    g = _pin(g, at, inside)
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
    lengths like `_integrate`; a step counts where both its frames do."""
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
            vis_loss = share * w[r.i] * visibility_error(r.logits, _cut(visible, r.start, r.end))
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
