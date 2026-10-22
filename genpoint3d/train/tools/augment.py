"""Change how a training clip looks, never where anything is.

TAPIP3D trains with colour jitter and blur among its augmentations
(`kubric_base.yaml`); we had none, and our precision on moving points falls
off as their look drifts from frame 0. Only photometric changes here: the
tracks, depth and cameras stay exactly true, so no label has to follow.

Per clip, a brightness, contrast and saturation change, and sometimes a blur;
per frame, a smaller change on top, so a clip's look also drifts over time.
"""

import torch
import torch.nn.functional as F


def augment_frames(frames: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 -> the same, recoloured and sometimes blurred."""
    B, T = frames.shape[:2]
    dev = frames.device
    x = frames.float() / 255.0
    u = lambda lo, hi, *shape: torch.empty(*shape, device=dev).uniform_(lo, hi)
    # A clip-wide change, and a smaller one per frame on top of it.
    bright = u(0.75, 1.25, B, 1) * u(0.92, 1.08, B, T)
    contrast = u(0.75, 1.25, B, 1) * u(0.92, 1.08, B, T)
    sat = u(0.7, 1.3, B, 1) * u(0.9, 1.1, B, T)
    shape = lambda v: v[:, :, None, None, None]
    grey = x.mean(dim=-1, keepdim=True)
    x = grey + (x - grey) * shape(sat)
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    x = mean + (x - mean) * shape(contrast)
    x = x * shape(bright)
    # A Gaussian blur on about a third of the clips.
    sigma = torch.where(torch.rand(B, device=dev) < 0.3, u(0.5, 1.5, B), torch.zeros(B, device=dev))
    if (sigma > 0).any():
        r = torch.arange(-3, 4, device=dev, dtype=x.dtype)
        k = torch.exp(-r[None] ** 2 / (2 * sigma.clamp_min(1e-3)[:, None] ** 2))
        k = torch.where((sigma > 0)[:, None], k / k.sum(-1, keepdim=True),
                        (r[None] == 0).to(x.dtype))                          # (B, 7)
        img = x.permute(0, 1, 4, 2, 3).reshape(B, T * 3, *x.shape[2:4])
        kx = k[:, None, None, :].repeat_interleave(T * 3, 0)                 # (B*T*3, 1, 1, 7)
        img = img.reshape(1, B * T * 3, *img.shape[2:])
        img = F.conv2d(F.pad(img, (3, 3, 0, 0), mode="replicate"), kx, groups=B * T * 3)
        img = F.conv2d(F.pad(img, (0, 0, 3, 3), mode="replicate"), kx.transpose(2, 3), groups=B * T * 3)
        x = img.reshape(B, T, 3, *img.shape[2:]).permute(0, 1, 3, 4, 2)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)
