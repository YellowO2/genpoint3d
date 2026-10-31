"""Train and run the model by refining its own guess, with no noise.

Iterative regression -- the same network, trained and run without noise.

The control for flow matching (`flow.py`). There the network sees the truth
mixed with noise and every look at the image is centred on that noisy sample.
Here it sees its own current guess, the way TAPIP3D and CoTracker work:

    g = 0                                 "nothing moves", for a displacement target
    repeat `iters` times:
        g = g.detach() + model(g, k_i, anchor)

Nothing else differs: the network, its inputs and the error it is scored by are
the ones flow matching uses, so a gap between the two is about the method. The
output that flow matching reads as a velocity is read here as a correction.

`k` is the iteration clock, `k_i = i / (iters - 1)`. The network already takes
a scalar in [0, 1] that tells it how far along it is and, with
`locality_mode="sched"`, how wide to look: wide on the first guess, one patch
spacing on the last.

A model built with `vis_feedback` is also given, at every iteration, the
visibility logits the iteration before it gave, detached like the guess and
0 before the first, as TAPIP3D does.

`param` says what each correction the network gives per frame means, before
it is added to the guess:

    position      a change to that frame's displacement, as above
    velocity      a change to the speed from that frame on: a correction at
                  frame t moves every later frame by the same amount
    acceleration  a change to how fast the speed changes from that frame on:
                  a correction at frame t changes the speed of every later frame

A clip longer than the network should see at once is tracked in windows:
`window` frames at a time, each window starting `overlap` frames before the
one before it ends, as TAPIP3D does. Its frames past the last window start
where the point last was, and the frames it shares were settled by that
window. Every window still reads the clip's frame 0 as its template.

So an output of zero at a frame the network cannot see keeps the guess there
moving the way it already moves, rather than holding it in place. Either way
the correction is summed into a displacement first, so the guess, the loss
and the score all stay in displacement units.
"""

from typing import Optional

import torch
from torch import nn

from genpoint3d.train.tools.flow import masked_error
from genpoint3d.train.tools.visibility import visibility_error


def _pin(g: torch.Tensor, known_x0: Optional[torch.Tensor]) -> torch.Tensor:
    """Hold frame 0 at its known position, as the flow method does."""
    if known_x0 is None:
        return g
    return torch.cat([known_x0[:, None], g[:, 1:]], dim=1)


def _integrate(step: torch.Tensor, param: str) -> torch.Tensor:
    """(B, T, N, 3) the network's per-frame output, as a change in displacement.

    Velocity and acceleration are summed over frames 1..t, with time measured
    in clip lengths, one frame being 1 / (T - 1). An output the same size in
    every frame then moves the last frame by about that much (all of it for a
    velocity, half for an acceleration), as a position output would: without
    the step, a velocity would move it T - 1 times as far and an acceleration
    about T^2 / 2 times, and every weight change near the output would be
    that much larger from the first step on. Frame 0 always gets zero.
    """
    if param == "position":
        return step
    dt = 1.0 / max(step.shape[1] - 1, 1)
    rate = torch.cat([torch.zeros_like(step[:, :1]), step[:, 1:]], dim=1)
    rate = rate.cumsum(dim=1) * dt
    if param == "acceleration":
        rate = rate.cumsum(dim=1) * dt
    return rate


def _carry(g: torch.Tensor, i: int, iters: int) -> torch.Tensor:
    """After iteration `i`, every frame past a horizon restarts from the
    horizon's own guess, the horizon moving from frame 0 to the last frame
    over the first `iters - 1` iterations.

    The guess starts at frame 0's position in every frame, and each frame's
    window looks around its own guess: a point that has gone 200 px by frame
    20 is outside it, and only the whole-frame match can bring it back. With
    the horizon, the frames just past it start where the point last was, and
    the window walks forward with it, as TAPIP3D's overlapping windows do
    (each starting from the one before). With 4 iterations and 24 frames,
    frames past 8 restart from frame 8, then frames past 15 from frame 15.
    """
    T = g.shape[1]
    if iters < 2:
        return g
    h = round((T - 1) * (i + 1) / (iters - 1))
    if h >= T - 1:
        return g
    return torch.cat([g[:, :h + 1], g[:, h:h + 1].expand_as(g[:, h + 1:])], dim=1)


def _windows(T: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) of each window over `T` frames. The last one ends at the
    clip's end, so it may share more than `overlap` frames. A `window` of 0, or
    one as long as the clip, is the whole clip."""
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
    """A (B, T, ...) tensor's frames `start` to `end`, or None."""
    return None if t is None else t[:, start:end]


def _reach(g: torch.Tensor, done: int, end: int, start: int = 0, reach: str = "hold") -> torch.Tensor:
    """The frames from `done` to `end`, which no window has settled yet, start
    where the point was in the last one that has ("hold"), or carry on from
    there at its average speed over the frames this window shares with that
    one, `start` to `done` ("continue")."""
    if done == 0 or done >= end:
        return g
    last = g[:, done - 1:done]
    new = last.expand(-1, end - done, -1, -1)
    if reach == "continue" and done - start > 1:
        speed = (last - g[:, start:start + 1]) / (done - 1 - start)
        n = torch.arange(1, end - done + 1, device=g.device, dtype=g.dtype)
        new = last + speed * n[None, :, None, None]
    return torch.cat([g[:, :done], new, g[:, end:]], dim=1)


def _turn(model: nn.Module) -> dict:
    """What this round's fit hands the next round, as model call arguments:
    its rotation for a model built with `rot_match`; for one built with
    `fit_input`, all of it -- positions, rotation, group vectors and
    confidences -- detached like the guess."""
    out = {}
    if getattr(model, "rot_match", False):
        out["rot"] = model._fitted[2]
    if getattr(model, "fit_input", False):
        fit, emb, R, conf = model._fitted
        out["fit_hint"] = (fit.detach(), R, emb.detach(), conf.detach())
    if getattr(model, "group_feedback", 0):
        out["group_prev"] = model._emb.detach()
    return out


def _group_error(emb: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """How far the affinities of the points' vectors (B, N, E) are from the
    pairs that truly move as one (B, N, N): cross-entropy of
    2 sigmoid(-|v_i - v_j|^2) against 1 for such a pair and 0 otherwise, the
    two kinds averaged apart so the rarer one counts as much."""
    v = emb.float()
    a = (2 * torch.sigmoid(-(v[:, :, None] - v[:, None]).pow(2).sum(-1))).clamp(1e-4, 1 - 1e-4)
    off = ~torch.eye(emb.shape[1], dtype=torch.bool, device=emb.device)
    pos, neg = groups & off, ~groups & off
    p = -(a.log() * pos).sum() / pos.sum().clamp_min(1)
    n = -((1 - a).log() * neg).sum() / neg.sum().clamp_min(1)
    return (p + n) / 2


def _no_logits(model: nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]:
    """(B, T, N) zeros for the first iteration's fed-back visibility, or None
    for a model that is not given it."""
    return x.new_zeros(x.shape[:3]) if model.vis_feedback else None


def _motion_error(g, x1, mask, weight, loss_type, order):
    """The error of how `g` moves against how the truth moves.

    order 1: each frame-to-frame change, the speed.
    order 2: each change of that change, how the speed changes.

    A change counts when every frame it spans counts in the position loss, at
    the mean of their weights. Each is times T - 1 once, so the speed is in the
    same clip-length time units as `_integrate`, and a guess that drifts away a
    little every frame costs about as much as one off by that drift at the
    end. A change of speed is left per frame, so it stays in the same units as
    a speed rather than growing with T.
    """
    s = x1.shape[1] - 1
    dg, dx = g.diff(n=order, dim=1) * s, x1.diff(n=order, dim=1) * s
    m = w = None
    if mask is not None:
        m = mask[:, order:]
        for j in range(order):
            m = m & mask[:, j:j + dg.shape[1]]
    if weight is not None:
        w = sum(weight[:, j:j + dg.shape[1]] for j in range(order + 1)) / (order + 1)
    return masked_error(dg, dx, m, w, loss_type)


def _rigid_error(g, x1, pairs, loss_type):
    """The error of the gap between two points of one rigid object, `pairs`
    (B, N, N), against the true gap, in every frame. The gap's error is the
    difference of the two points' own errors, so this asks the points of an
    object to be wrong together: a hidden point that stays behind while its
    object's visible points move costs, even where both are close to right.
    """
    e = (g - x1)[:, :, :, None] - (g - x1)[:, :, None]           # (B, T, N, N, 3)
    err = e.norm(dim=-1) if loss_type == "l21" else e.pow(2).mean(dim=-1)
    w = pairs[:, None].to(err.dtype).expand_as(err)
    return (err * w).sum() / w.sum().clamp(min=1e-8)


def _clock(i: int, iters: int, batch: int, device: torch.device) -> torch.Tensor:
    """(B,) `k` for iteration `i`. A single iteration is the last one."""
    return torch.full((batch,), i / (iters - 1) if iters > 1 else 1.0, device=device)


def regress_loss(
    model: nn.Module,
    x1: torch.Tensor,
    anchor: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    known_x0: Optional[torch.Tensor] = None,
    loss_type: str = "l2",
    iters: int = 4,
    gamma: float = 0.8,
    backward: Optional[float] = None,
    visible: Optional[torch.Tensor] = None,
    vis_weight: float = 0.0,
    param: str = "position",
    carry: bool = False,
    vel_weight: float = 0.0,
    acc_weight: float = 0.0,
    pairs: Optional[torch.Tensor] = None,
    rigid_weight: float = 0.0,
    window: int = 0,
    overlap: int = 0,
    groups: Optional[torch.Tensor] = None,
    group_weight: float = 0.0,
    fit_weight: float = 0.0,
    **cond,
) -> tuple[torch.Tensor, dict]:
    """
    x1, anchor, mask, weight, known_x0, loss_type, visible, vis_weight, cond:
              as `flow_matching_loss`. The visibility loss is scored at every
              iteration too, with the same weights, and added to the
              position loss before each backward pass.
    iters:    refinement iterations, each one supervised against `x1`.
    gamma:    iteration i is weighted gamma^(iters - 1 - i), so the last guess
              counts most. TAPIP3D's weighting and its value
              (`training/criterion.py`, `gamma: 0.8`). Normalised to sum to 1,
              which theirs is not: the loss then stays an error per point,
              comparable with the flow loss and with the gradient clip.
    backward: None returns the loss with its graph, all `iters` forward passes
              of it. A number back-propagates each iteration's share, times
              that number, as soon as it exists and returns the loss detached.
              The guess is detached between iterations, so no gradient crosses
              from one to the next and the two give the same gradient -- but
              this way a step holds one forward pass in memory, not `iters`
              (TAPIP3D appendix, "memory-saving strategy"). The caller still
              zeroes, clips and steps. The loss returned is then the position
              loss alone, and the visibility loss is only in the dict.
              With no `visible`, a model built with `vis_feedback` still
              computes its visibility, to feed it back, but nothing scores it.
    param:    what the network's output means, see the top of this file.
    carry:    restart the frames past a moving horizon from its guess
              between iterations (`_carry`).
    vel_weight: also score each frame-to-frame change of the guess against the
              truth's, weighted this against the position loss, at every
              iteration. Part of the returned loss.
    acc_weight: the same for each change of that change, how the speed of the
              guess changes against how the truth's does.
    pairs, rigid_weight: (B, N, N) the pairs of points on one rigid object
              (`batch.rigid_pairs`), and the weight of the loss that keeps
              the gap between them true, at every iteration.
    groups, group_weight, fit_weight: a model built with `rigid_fit` also
              has its points' vectors scored against `groups` (B, N, N), the
              pairs the truth shows keeping their distance (`_group_error`),
              and the fitted positions scored like the guess in every round,
              those it is used in and those after.
    window, overlap: track in windows of this many frames, sharing `overlap`
              (top of this file); 0 tracks the whole clip at once. Each
              window is scored on its own frames, and the windows' losses are
              averaged.

    Frame 0 is scored before it is pinned again: the network is asked for a
    zero correction there, as the flow loss asks for a zero velocity. With a
    velocity or an acceleration its correction there is zero by construction.
    """
    if window and carry:
        raise ValueError("windows and carry both restart the later frames: use one")
    B, T = x1.shape[:2]
    w = [gamma ** (iters - 1 - i) for i in range(iters)]
    w = [v / sum(w) for v in w]
    spans = _windows(T, window, overlap)
    share = 1 / len(spans)

    whole = _pin(torch.zeros_like(x1), known_x0)
    total = vis_total = x1.new_zeros(())
    done = 0
    for start, end in spans:
        whole = _reach(whole, done, end)
        g, x1_w = whole[:, start:end], x1[:, start:end]
        mask_w, weight_w, visible_w = (_cut(t, start, end) for t in (mask, weight, visible))
        kx0 = known_x0 if start == 0 else None
        at = {} if len(spans) == 1 else {"frames": (start, end)}
        prev = _no_logits(model, x1_w)
        rot = {}
        for i in range(iters):
            k = _clock(i, iters, B, x1.device)
            vis_loss = x1.new_zeros(())
            if visible_w is None and prev is None:
                step = model(g.detach(), k, anchor, **at, **rot, **cond)
            else:
                step, logits = model(g.detach(), k, anchor, visibility=True, vis_prev=prev, **at, **rot, **cond)
                if visible_w is not None:
                    vis_loss = share * w[i] * visibility_error(logits, visible_w)
                if prev is not None:
                    prev = logits.detach()
            g = model.settle(g.detach() + _integrate(step, param), anchor, i)
            rot = _turn(model)
            loss = w[i] * masked_error(g, x1_w, mask_w, weight_w, loss_type)
            if vel_weight:
                loss = loss + w[i] * vel_weight * _motion_error(g, x1_w, mask_w, weight_w, loss_type, 1)
            if acc_weight:
                loss = loss + w[i] * acc_weight * _motion_error(g, x1_w, mask_w, weight_w, loss_type, 2)
            if rigid_weight:
                loss = loss + w[i] * rigid_weight * _rigid_error(g, x1_w, pairs, loss_type)
            if fit_weight:
                loss = loss + w[i] * fit_weight * masked_error(model._fitted[0], x1_w, mask_w, weight_w, loss_type)
            if group_weight:
                emb = model._emb if getattr(model, "group_feedback", 0) else model._fitted[1]
                loss = loss + w[i] * group_weight * _group_error(emb, groups)
            loss = share * loss
            if backward is not None:
                # Out of autocast, where the training loop's own backward runs.
                with torch.autocast(device_type=x1.device.type, enabled=False):
                    ((loss + vis_weight * vis_loss) * backward).backward()
                loss, vis_loss = loss.detach(), vis_loss.detach()
            total = total + loss
            vis_total = vis_total + vis_loss
            g = _pin(g, kx0)
            if carry:
                g = _carry(g.detach(), i, iters)
        whole = torch.cat([whole[:, :start], g.detach(), whole[:, end:]], dim=1)
        done = end

    if backward is None:
        total = total + vis_weight * vis_total
    return total, {"loss": total.detach(), "visibility": vis_total.detach()}


@torch.no_grad()
def refine(
    model: nn.Module,
    anchor: torch.Tensor,
    num_frames: int,
    iters: int = 4,
    known_x0: Optional[torch.Tensor] = None,
    visibility: bool = False,
    param: str = "position",
    carry: bool = False,
    window: int = 0,
    overlap: int = 0,
    rounds: Optional[list] = None,
    reach: str = "hold",
    **cond,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """The prediction: `iters` corrections to a guess that starts at zero.

    No noise anywhere, so the same input always gives the same answer. With a
    displacement target, a network that outputs zero gives exactly the static
    baseline.

    `visibility` also returns the visibility logits (B, T, N) of the last
    iteration, the one the training weighs most; in windows, of the last
    window that covers each frame. `param`, `carry`, `window` and `overlap`
    are as in `regress_loss`, and must be what the model was trained with.

    A `rounds` list is filled with the whole clip's guess before the first
    correction and after each one, `iters + 1` (B, T, N, 3) tensors; in
    windows, each frame from the last window that covers it.

    `reach`: how a window's frames past the last one start (`_reach`).
    """
    if window and carry:
        raise ValueError("windows and carry both restart the later frames: use one")
    B, N, _ = anchor.shape
    spans = _windows(num_frames, window, overlap)
    whole = _pin(torch.zeros(B, num_frames, N, 3, device=anchor.device), known_x0)
    seen = whole.new_zeros(B, num_frames, N) if visibility else None
    done = 0
    for start, end in spans:
        whole = _reach(whole, done, end, start, reach)
        g = whole[:, start:end]
        kx0 = known_x0 if start == 0 else None
        at = {} if len(spans) == 1 else {"frames": (start, end)}
        prev = _no_logits(model, g)
        logits = None
        rot = {}
        if rounds is not None:
            if not rounds:
                rounds.extend(whole.clone() for _ in range(iters + 1))
            rounds[0] = torch.cat([rounds[0][:, :start], g, rounds[0][:, end:]], dim=1)
        for i in range(iters):
            out = model(g, _clock(i, iters, B, anchor.device), anchor,
                        visibility=visibility or prev is not None, vis_prev=prev, **at, **rot, **cond)
            if visibility or prev is not None:
                out, logits = out
            if prev is not None:
                prev = logits
            g = _pin(model.settle(g + _integrate(out, param), anchor, i), kx0)
            rot = _turn(model)
            if carry:
                g = _carry(g, i, iters)
            if rounds is not None:
                rounds[i + 1] = torch.cat([rounds[i + 1][:, :start], g, rounds[i + 1][:, end:]], dim=1)
        whole = torch.cat([whole[:, :start], g, whole[:, end:]], dim=1)
        if visibility:
            seen = torch.cat([seen[:, :start], logits.to(seen.dtype), seen[:, end:]], dim=1)
        done = end
    return (whole, seen) if visibility else whole
