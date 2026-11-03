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


def _integrate_at(step: torch.Tensor, param: str, at: torch.Tensor) -> torch.Tensor:
    """`_integrate` from each point's own fixed frame `at` (B, N), an index
    into the window: that frame gets no correction, a later frame's output is
    its step from the frame before it and an earlier frame's its step from the
    frame after it, so an output of zero holds a frame level with its
    neighbour towards `at`. `at` 0 for every point is `_integrate`; the
    window's last frame is the backward pass's, whose last frame the window
    after it has settled; a point's own query frame, inside the window, keeps
    the point where it was seen."""
    if param == "acceleration":
        if bool((at != 0).any()):
            raise ValueError("an acceleration integrates from the window's first frame only")
        return _integrate(step, param)
    L = step.shape[1]
    t = torch.arange(L, device=step.device)[None, :, None, None]
    a = at[:, None, :, None]                                                  # (B, 1, N, 1)
    if param == "position":
        return step * (t != a).to(step.dtype)
    c = torch.cumsum(step * (1.0 / max(L - 1, 1)), dim=1)                     # (B, L, N, 3)
    before = torch.cat([torch.zeros_like(c[:, :1]), c[:, :-1]], dim=1)        # c at t - 1
    pick = lambda v, i: v.gather(1, i.clamp(0, L - 1).expand(-1, -1, -1, 3))
    c_a = pick(c, a)                                                          # c at the fixed frame
    c_a1 = pick(before, a)                                                    # c at the frame before it
    return torch.where(t > a, c - c_a, torch.where(t < a, c_a1 - before, torch.zeros_like(c)))


def _window_points(cond: dict, B: int, N: int, start: int, end: int, backward: bool, device,
                   every: bool = False):
    """Per window, from each point's query frame (`cond["query_t"]` (B, N);
    none means frame 0) and the frames with an image (`cond["visual_mask"]`):
    `alive` (B, N), the points the window has -- on the forward pass those
    first seen by its end (`every`: all of them, once a backward sweep has
    given late points their past), on the backward pass all of them, either way only
    points whose query frame has its image; `at` (B, N), the frame each
    point's steps run from (`_integrate_at`): its query frame when that is in
    the window, else the window's first frame forward and its last backward;
    `inside` (B, N), whether the query frame is in the window."""
    q = cond.get("query_t")
    if q is None:
        q = torch.zeros(B, N, dtype=torch.long, device=device)
    vm = cond.get("visual_mask")
    seen = torch.ones(B, N, dtype=torch.bool, device=device)
    if vm is not None:
        seen = (vm[..., None].expand(-1, -1, N) if vm.dim() == 2 else vm).gather(1, q[:, None])[:, 0]
    inside = (q >= start) & (q < end)
    alive = seen if backward or every else seen & (q < end)
    at = torch.where(inside, q - start, torch.full_like(q, end - start - 1 if backward else 0))
    return alive, at, inside


def _alive_only(mask_w: Optional[torch.Tensor], pairs: Optional[torch.Tensor], alive: torch.Tensor,
                g: torch.Tensor) -> tuple:
    """The window's loss mask (B, L, N) and rigid pairs (B, N, N) cut to the
    points it has: a point not yet in it is neither scored nor a partner."""
    keep = alive[:, None].expand(-1, g.shape[1], -1)
    mask_w = keep if mask_w is None else mask_w & keep
    if pairs is not None:
        pairs = pairs & alive[:, :, None] & alive[:, None, :]
    return mask_w, pairs


def _pin_at(g: torch.Tensor, known_x0: Optional[torch.Tensor], at: torch.Tensor,
            inside: torch.Tensor) -> torch.Tensor:
    """Hold each point whose query frame is in the window at its known position
    there, as `_pin` holds frame 0."""
    if known_x0 is None or not bool(inside.any()):
        return g
    idx = at[:, None, :, None].expand(-1, 1, -1, 3)
    held = g.gather(1, idx)
    held = torch.where(inside[:, None, :, None], known_x0[:, None].to(g.dtype), held)
    return g.scatter(1, idx, held)


def _schedule(iters: int, back_iters: int, schedule: str) -> list[tuple[str, int]]:
    """The sweeps over the windows, in order, each ("f" forward or "b"
    backward, rounds a window). `schedule` spells them one round a letter,
    e.g. "fbfb": forward, back, forward, back, each sweep starting from where
    the one before left every frame. Empty: `iters` forward, then
    `back_iters` backward."""
    if schedule:
        if set(schedule) - {"f", "b"} or schedule[0] != "f":
            raise ValueError(f"a schedule is f and b, starting with f: {schedule!r}")
        return [(c, 1) for c in schedule]
    return [("f", iters)] + ([("b", back_iters)] if back_iters else [])


def _check_backward(model: nn.Module, back_iters: int) -> None:
    """The backward pass is built for the plain model and its hidden blanking:
    nothing fed back from round to round (visibility, groups, a rigid fit)."""
    if back_iters and (model.vis_feedback or getattr(model, "group_feedback", 0) or model.rigid_fit):
        raise ValueError("the backward pass does not carry fed-back visibility, groups or a rigid fit")


def _direction(model: nn.Module, sign: float, B: int, device) -> dict:
    """The model call's `direction` argument, (B,) +1 forward or -1 backward,
    for a model built with `bidirectional`; nothing for any other."""
    if not getattr(model, "bidirectional", False):
        return {}
    return {"direction": torch.full((B,), sign, device=device)}


def _blank_hidden(cond: dict, logits: torch.Tensor, start: int, end: int, num_frames: int) -> dict:
    """`cond` for the next round of a model built with `hidden_blank`: in the
    window's frames `start` to `end`, each point the visibility head (logits
    (B, end - start, N)) calls hidden gets no image, exactly as a forecast
    frame gets none -- a hidden point is carried by the rest of the scene, not
    matched to whatever covers it. Each point's query frame keeps its image."""
    B, _, N = logits.shape
    vm = cond.get("visual_mask")
    if vm is None:
        vm = logits.new_ones(B, num_frames, dtype=torch.bool)
    full = (vm[..., None].expand(B, num_frames, N) if vm.dim() == 2 else vm).clone()
    seen = logits.detach() > 0
    q = cond.get("query_t")
    if q is None:
        q = torch.zeros(B, N, dtype=torch.long, device=logits.device)
    inside = (q >= start) & (q < end)
    seen.scatter_(1, (q - start).clamp(0, end - start - 1)[:, None],
                  seen.gather(1, (q - start).clamp(0, end - start - 1)[:, None]) | inside[:, None])
    full[:, start:end] &= seen
    return {**cond, "visual_mask": full}


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
    reach: str = "hold",
    back_iters: int = 0,
    schedule: str = "",
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

    schedule: the sweeps one round a letter, e.g. "fbfb" (`_schedule`); when
              given, `iters` and `back_iters` are not used.
    back_iters: after the forward pass over the windows, a backward pass over
              them from the clip's end to its start, this many rounds a
              window, each from where the forward pass left it, the model told
              time runs backward (`direction`) and each output a step from
              the frame after (`_integrate_at`): later frames inform earlier
              ones without a window that holds both. Rounds are weighted and
              clocked over both passes together; windows of both passes are
              averaged.

    Frame 0 is scored before it is pinned again: the network is asked for a
    zero correction there, as the flow loss asks for a zero velocity. With a
    velocity or an acceleration its correction there is zero by construction.
    """
    if window and carry:
        raise ValueError("windows and carry both restart the later frames: use one")
    _check_backward(model, back_iters or "b" in schedule)
    B, T = x1.shape[:2]
    R = sum(n for _, n in _schedule(iters, back_iters, schedule))
    w = [gamma ** (R - 1 - i) for i in range(R)]
    w = [v / sum(w) for v in w]
    spans = _windows(T, window, overlap)
    share = 1 / (len(spans) * len(_schedule(iters, back_iters, schedule)))

    whole = _pin(torch.zeros_like(x1), known_x0)
    total = vis_total = x1.new_zeros(())
    seen_all = None
    done = 0
    r0 = 0
    for sweep, n in _schedule(iters, back_iters, schedule):
        if sweep == "f":
            for start, end in spans:
                whole = _reach(whole, done, end, start, reach)
                g, x1_w = whole[:, start:end], x1[:, start:end]
                mask_w, weight_w, visible_w = (_cut(t, start, end) for t in (mask, weight, visible))
                alive, at_q, inside = _window_points(cond, B, x1.shape[2], start, end, False, x1.device,
                                                     every=r0 > 0)
                mask_w, pairs_w = _alive_only(mask_w, pairs, alive, g)
                kx0 = known_x0
                at = ({} if len(spans) == 1 else {"frames": (start, end)}) | {"alive": alive}
                prev = _no_logits(model, x1_w)
                rot = {}
                blank = getattr(model, "hidden_blank", False)
                c = cond if seen_all is None or not blank else _blank_hidden(cond, seen_all[:, start:end], start, end, T)
                fwd = _direction(model, 1.0, B, x1.device)
                for i in range(r0, r0 + n):
                    k = _clock(i, R, B, x1.device)
                    vis_loss = x1.new_zeros(())
                    if visible_w is None and prev is None and not blank:
                        step = model(g.detach(), k, anchor, **at, **rot, **fwd, **c)
                    else:
                        step, logits = model(g.detach(), k, anchor, visibility=True, vis_prev=prev, **at, **rot,
                                             **fwd, **c)
                        if visible_w is not None:
                            vis_loss = share * w[i] * visibility_error(logits, visible_w)
                        if prev is not None:
                            prev = logits.detach()
                        if blank:
                            c = _blank_hidden(cond, logits, start, end, T)
                    g = torch.where(alive[:, None, :, None],
                                    model.settle(g.detach() + _integrate_at(step, param, at_q), anchor, i), g.detach())
                    rot = _turn(model)
                    loss = w[i] * masked_error(g, x1_w, mask_w, weight_w, loss_type)
                    if vel_weight:
                        loss = loss + w[i] * vel_weight * _motion_error(g, x1_w, mask_w, weight_w, loss_type, 1)
                    if acc_weight:
                        loss = loss + w[i] * acc_weight * _motion_error(g, x1_w, mask_w, weight_w, loss_type, 2)
                    if rigid_weight:
                        loss = loss + w[i] * rigid_weight * _rigid_error(g, x1_w, pairs_w, loss_type)
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
                    g = _pin_at(g, kx0, at_q, inside)
                    if carry:
                        g = _carry(g.detach(), i, iters)
                whole = torch.cat([whole[:, :start], g.detach(), whole[:, end:]], dim=1)
                if blank:
                    if seen_all is None:
                        seen_all = logits.new_zeros(B, T, logits.shape[-1])
                    seen_all[:, start:end] = logits.detach()
                done = end
        else:
            for start, end in reversed(spans):
                g, x1_w = whole[:, start:end], x1[:, start:end]
                mask_w, weight_w, visible_w = (_cut(t, start, end) for t in (mask, weight, visible))
                alive, at_q, inside = _window_points(cond, B, x1.shape[2], start, end, True, x1.device)
                mask_w, pairs_w = _alive_only(mask_w, pairs, alive, g)
                kx0 = known_x0
                at = ({} if len(spans) == 1 else {"frames": (start, end)}) | {"alive": alive}
                blank = getattr(model, "hidden_blank", False)
                # Hidden points stay blanked from the forward pass's last call on.
                c = _blank_hidden(cond, seen_all[:, start:end], start, end, T) if blank else cond
                bwd = _direction(model, -1.0, B, x1.device)
                for i in range(r0, r0 + n):
                    vis_loss = x1.new_zeros(())
                    out = model(g.detach(), _clock(i, R, B, x1.device), anchor, visibility=model.visibility,
                                **at, **bwd, **c)
                    step, logits = out if model.visibility else (out, None)
                    if visible_w is not None:
                        vis_loss = share * w[i] * visibility_error(logits, visible_w)
                    if blank:
                        c = _blank_hidden(cond, logits, start, end, T)
                    g = torch.where(alive[:, None, :, None], g.detach() + _integrate_at(step, param, at_q), g.detach())
                    loss = w[i] * masked_error(g, x1_w, mask_w, weight_w, loss_type)
                    if vel_weight:
                        loss = loss + w[i] * vel_weight * _motion_error(g, x1_w, mask_w, weight_w, loss_type, 1)
                    if acc_weight:
                        loss = loss + w[i] * acc_weight * _motion_error(g, x1_w, mask_w, weight_w, loss_type, 2)
                    if rigid_weight:
                        loss = loss + w[i] * rigid_weight * _rigid_error(g, x1_w, pairs_w, loss_type)
                    loss = share * loss
                    if backward is not None:
                        with torch.autocast(device_type=x1.device.type, enabled=False):
                            ((loss + vis_weight * vis_loss) * backward).backward()
                        loss, vis_loss = loss.detach(), vis_loss.detach()
                    total = total + loss
                    vis_total = vis_total + vis_loss
                    g = _pin_at(g, kx0, at_q, inside)
                whole = torch.cat([whole[:, :start], g.detach(), whole[:, end:]], dim=1)
                if blank and logits is not None:
                    seen_all[:, start:end] = logits.detach()
        r0 += n

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
    back_iters: int = 0,
    schedule: str = "",
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
    `back_iters`: rounds a window of the backward pass, as `regress_loss`.
    """
    if window and carry:
        raise ValueError("windows and carry both restart the later frames: use one")
    _check_backward(model, back_iters or "b" in schedule)
    B, N, _ = anchor.shape
    R = sum(n for _, n in _schedule(iters, back_iters, schedule))
    spans = _windows(num_frames, window, overlap)
    whole = _pin(torch.zeros(B, num_frames, N, 3, device=anchor.device), known_x0)
    seen = whole.new_zeros(B, num_frames, N) if visibility else None
    seen_all = None
    done = 0
    r0 = 0
    for sweep, n in _schedule(iters, back_iters, schedule):
        if sweep == "f":
            for start, end in spans:
                whole = _reach(whole, done, end, start, reach)
                g = whole[:, start:end]
                alive, at_q, inside = _window_points(cond, B, N, start, end, False, anchor.device,
                                                     every=r0 > 0)
                kx0 = known_x0
                at = ({} if len(spans) == 1 else {"frames": (start, end)}) | {"alive": alive}
                prev = _no_logits(model, g)
                logits = None
                rot = {}
                blank = getattr(model, "hidden_blank", False)
                c = cond if seen_all is None or not blank else _blank_hidden(cond, seen_all[:, start:end], start, end, num_frames)
                fwd = _direction(model, 1.0, B, anchor.device)
                if rounds is not None:
                    if not rounds:
                        rounds.extend(whole.clone() for _ in range(R + 1))
                    rounds[0] = torch.cat([rounds[0][:, :start], g, rounds[0][:, end:]], dim=1)
                for i in range(r0, r0 + n):
                    out = model(g, _clock(i, R, B, anchor.device), anchor,
                                visibility=visibility or prev is not None or blank, vis_prev=prev, **at, **rot,
                                **fwd, **c)
                    if visibility or prev is not None or blank:
                        out, logits = out
                    if prev is not None:
                        prev = logits
                    if blank:
                        c = _blank_hidden(cond, logits, start, end, num_frames)
                    g = torch.where(alive[:, None, :, None],
                                    _pin_at(model.settle(g + _integrate_at(out, param, at_q), anchor, i), kx0, at_q, inside), g)
                    rot = _turn(model)
                    if carry:
                        g = _carry(g, i, iters)
                    if rounds is not None:
                        rounds[i + 1] = torch.cat([rounds[i + 1][:, :start], g, rounds[i + 1][:, end:]], dim=1)
                whole = torch.cat([whole[:, :start], g, whole[:, end:]], dim=1)
                if visibility:
                    seen = torch.cat([seen[:, :start], logits.to(seen.dtype), seen[:, end:]], dim=1)
                if blank:
                    if seen_all is None:
                        seen_all = logits.new_zeros(B, num_frames, N)
                    seen_all[:, start:end] = logits
                done = end
        else:
            for start, end in reversed(spans):
                g = whole[:, start:end]
                alive, at_q, inside = _window_points(cond, B, N, start, end, True, anchor.device)
                kx0 = known_x0
                at = ({} if len(spans) == 1 else {"frames": (start, end)}) | {"alive": alive}
                blank = getattr(model, "hidden_blank", False)
                c = _blank_hidden(cond, seen_all[:, start:end], start, end, num_frames) if blank else cond
                bwd = _direction(model, -1.0, B, anchor.device)
                for i in range(r0, r0 + n):
                    want = visibility or blank
                    out = model(g, _clock(i, R, B, anchor.device), anchor, visibility=want, **at, **bwd, **c)
                    out, logits = out if want else (out, None)
                    if blank:
                        c = _blank_hidden(cond, logits, start, end, num_frames)
                    g = torch.where(alive[:, None, :, None], _pin_at(g + _integrate_at(out, param, at_q), kx0, at_q, inside), g)
                    if rounds is not None:
                        rounds[i + 1] = torch.cat([rounds[i + 1][:, :start], g, rounds[i + 1][:, end:]], dim=1)
                whole = torch.cat([whole[:, :start], g, whole[:, end:]], dim=1)
                if blank and logits is not None:
                    seen_all[:, start:end] = logits.detach()
                if visibility:
                    seen = torch.cat([seen[:, :start], logits.to(seen.dtype), seen[:, end:]], dim=1)
        r0 += n

    return (whole, seen) if visibility else whole
