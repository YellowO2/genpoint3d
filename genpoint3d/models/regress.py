"""
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

from genpoint3d.models.flow import masked_error


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
    **cond,
) -> tuple[torch.Tensor, dict]:
    """
    x1, anchor, mask, weight, known_x0, loss_type, cond: as `flow_matching_loss`.
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
              zeroes, clips and steps.

    Frame 0 is scored before it is pinned again: the network is asked for a
    zero correction there, as the flow loss asks for a zero velocity.
    """
    B = x1.shape[0]
    w = [gamma ** (iters - 1 - i) for i in range(iters)]
    w = [v / sum(w) for v in w]

    g = _pin(torch.zeros_like(x1), known_x0)
    total = x1.new_zeros(())
    for i in range(iters):
        g = g.detach() + model(g.detach(), _clock(i, iters, B, x1.device), anchor, **cond)
        loss = w[i] * masked_error(g, x1, mask, weight, loss_type)
        if backward is not None:
            # Out of autocast, where the training loop's own backward runs.
            with torch.autocast(device_type=x1.device.type, enabled=False):
                (loss * backward).backward()
            loss = loss.detach()
        total = total + loss
        g = _pin(g, known_x0)

    return total, {"loss": total.detach()}


@torch.no_grad()
def refine(
    model: nn.Module,
    anchor: torch.Tensor,
    num_frames: int,
    iters: int = 4,
    known_x0: Optional[torch.Tensor] = None,
    **cond,
) -> torch.Tensor:
    """The prediction: `iters` corrections to a guess that starts at zero.

    No noise anywhere, so the same input always gives the same answer. With a
    displacement target, a network that outputs zero gives exactly the static
    baseline.
    """
    B, N, _ = anchor.shape
    g = _pin(torch.zeros(B, num_frames, N, 3, device=anchor.device), known_x0)
    for i in range(iters):
        g = _pin(g + model(g, _clock(i, iters, B, anchor.device), anchor, **cond), known_x0)
    return g
