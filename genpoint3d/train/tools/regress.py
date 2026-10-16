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

    Frame 0 is scored before it is pinned again: the network is asked for a
    zero correction there, as the flow loss asks for a zero velocity.
    """
    B = x1.shape[0]
    w = [gamma ** (iters - 1 - i) for i in range(iters)]
    w = [v / sum(w) for v in w]

    g = _pin(torch.zeros_like(x1), known_x0)
    total = vis_total = x1.new_zeros(())
    for i in range(iters):
        k = _clock(i, iters, B, x1.device)
        if visible is None:
            step = model(g.detach(), k, anchor, **cond)
            vis_loss = x1.new_zeros(())
        else:
            step, logits = model(g.detach(), k, anchor, visibility=True, **cond)
            vis_loss = w[i] * visibility_error(logits, visible)
        g = g.detach() + step
        loss = w[i] * masked_error(g, x1, mask, weight, loss_type)
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
    **cond,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """The prediction: `iters` corrections to a guess that starts at zero.

    No noise anywhere, so the same input always gives the same answer. With a
    displacement target, a network that outputs zero gives exactly the static
    baseline.

    `visibility` also returns the visibility logits (B, T, N) of the last
    iteration, the one the training weighs most.
    """
    B, N, _ = anchor.shape
    g = _pin(torch.zeros(B, num_frames, N, 3, device=anchor.device), known_x0)
    logits = None
    for i in range(iters):
        out = model(g, _clock(i, iters, B, anchor.device), anchor,
                    visibility=visibility, **cond)
        if visibility:
            out, logits = out
        g = _pin(g + out, known_x0)
    return (g, logits) if visibility else g
