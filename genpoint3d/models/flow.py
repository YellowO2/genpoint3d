"""
Conditional flow matching -- the training objective and the sampler.

The idea in three lines. Draw noise `x0 ~ N(0, I)` and take the ground-truth
trajectory `x1`. Walk in a straight line between them:

    x_k = (1 - k) * x0 + k * x1          k in [0, 1]

Moving along that line at constant speed has velocity `x1 - x0`, which does not
depend on `k`. Train the network to predict it from `x_k`. At sample time,
start from pure noise and integrate the predicted velocity from k=0 to k=1.

Diffusion works the same way with a curved path and a noise-prediction target.
Flow matching's straight path is why it needs far fewer sampling steps, and is
the post-2024 default (SD3, Flux).
"""

from typing import Optional

import torch
from torch import nn


def sample_k(
    batch: int,
    device: torch.device,
    loc: float = -1.0,
    scale: float = 1.5,
    shape: tuple = (),
) -> torch.Tensor:
    """Noise levels from a logit-normal distribution: `k = sigmoid(N(loc, scale))`.

    NOT uniform. The paper reports this as critical, and SD3 found the same:
    uniform `k` wastes most steps on the easy extremes (`k~0` is nearly pure
    noise, `k~1` is nearly clean), while the hard, informative work happens in
    the middle. `loc = -1` skews toward the noisier half.
    """
    return torch.sigmoid(torch.randn(batch, *shape, device=device) * scale + loc)


def flow_matching_loss(
    model: nn.Module,
    x1: torch.Tensor,
    anchor: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    per_frame_k: bool = False,
    **cond,
) -> tuple[torch.Tensor, dict]:
    """
    x1:     (B, T, N, 3) ground-truth trajectory, in model units
    anchor: (B, N, 3) conditioning
    mask:   (B, T, N) bool, True = include in the loss (e.g. visibility)
    weight: (B, T, N) per-point weight. Used to measure error in units of the
            metric's threshold rather than metres -- see `apd_weight` in
            scripts/train.py. None weights every point equally.
    cond:   optional `context` / `visual_mask` / `id_card`, passed straight to
            the model. Absent means the model sees no images.

    `per_frame_k` gives every frame its own independent noise level -- diffusion
    forcing. Off for now; turn it on when we add autoregressive rollout.
    """
    B, T = x1.shape[:2]
    k = sample_k(B, x1.device, shape=(T,) if per_frame_k else ())

    x0 = torch.randn_like(x1)
    k_b = k[..., None, None] if per_frame_k else k[:, None, None, None]
    x_k = (1.0 - k_b) * x0 + k_b * x1
    target = x1 - x0

    pred = model(x_k, k, anchor, **cond)
    err = (pred - target).pow(2).mean(dim=-1)  # (B, T, N)

    w = mask.float() if mask is not None else torch.ones_like(err)
    if weight is not None:
        w = w * weight
    loss = (err * w).sum() / w.sum().clamp(min=1e-8)

    return loss, {"loss": loss.detach(), "k_mean": k.mean().detach()}


@torch.no_grad()
def sample(
    model: nn.Module,
    anchor: torch.Tensor,
    num_frames: int,
    steps: int = 50,
    generator: Optional[torch.Generator] = None,
    **cond,
) -> torch.Tensor:
    """Integrate the velocity field from noise (k=0) to data (k=1).

    Plain Euler. The path is straight by construction, so a first-order solver
    is already close to exact -- this is flow matching's main practical win.
    """
    B, N, _ = anchor.shape
    x = torch.randn(B, num_frames, N, 3, device=anchor.device, generator=generator)

    dk = 1.0 / steps
    for i in range(steps):
        k = torch.full((B,), i * dk, device=anchor.device)
        x = x + model(x, k, anchor, **cond) * dk
    return x
