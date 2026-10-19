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
    vel_weight: float = 0.0,
    acc_weight: float = 0.0,
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
    vel_weight: also score each frame-to-frame change of the guess against the
              truth's, weighted this against the position loss, at every
              iteration. Part of the returned loss.
    acc_weight: the same for each change of that change, how the speed of the
              guess changes against how the truth's does.

    Frame 0 is scored before it is pinned again: the network is asked for a
    zero correction there, as the flow loss asks for a zero velocity. With a
    velocity or an acceleration its correction there is zero by construction.
    """
    B = x1.shape[0]
    w = [gamma ** (iters - 1 - i) for i in range(iters)]
    w = [v / sum(w) for v in w]

    g = _pin(torch.zeros_like(x1), known_x0)
    prev = _no_logits(model, x1)
    total = vis_total = x1.new_zeros(())
    for i in range(iters):
        k = _clock(i, iters, B, x1.device)
        vis_loss = x1.new_zeros(())
        if visible is None and prev is None:
            step = model(g.detach(), k, anchor, **cond)
        else:
            step, logits = model(g.detach(), k, anchor, visibility=True, vis_prev=prev, **cond)
            if visible is not None:
                vis_loss = w[i] * visibility_error(logits, visible)
            if prev is not None:
                prev = logits.detach()
        g = g.detach() + _integrate(step, param)
        loss = w[i] * masked_error(g, x1, mask, weight, loss_type)
        if vel_weight:
            loss = loss + w[i] * vel_weight * _motion_error(g, x1, mask, weight, loss_type, 1)
        if acc_weight:
            loss = loss + w[i] * acc_weight * _motion_error(g, x1, mask, weight, loss_type, 2)
        if backward is not None:
            # Out of autocast, where the training loop's own backward runs.
            with torch.autocast(device_type=x1.device.type, enabled=False):
                ((loss + vis_weight * vis_loss) * backward).backward()
            loss, vis_loss = loss.detach(), vis_loss.detach()
        total = total + loss
        vis_total = vis_total + vis_loss
        g = _pin(g, known_x0)

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
    **cond,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """The prediction: `iters` corrections to a guess that starts at zero.

    No noise anywhere, so the same input always gives the same answer. With a
    displacement target, a network that outputs zero gives exactly the static
    baseline.

    `visibility` also returns the visibility logits (B, T, N) of the last
    iteration, the one the training weighs most. `param` is as in
    `regress_loss`, and must be what the model was trained with.
    """
    B, N, _ = anchor.shape
    g = _pin(torch.zeros(B, num_frames, N, 3, device=anchor.device), known_x0)
    prev = _no_logits(model, g)
    logits = None
    for i in range(iters):
        out = model(g, _clock(i, iters, B, anchor.device), anchor,
                    visibility=visibility or prev is not None, vis_prev=prev, **cond)
        if visibility or prev is not None:
            out, logits = out
        if prev is not None:
            prev = logits
        g = _pin(g + _integrate(out, param), known_x0)
    return (g, logits) if visibility else g
