"""Read a feature grid at pixel positions, the way the cache does."""

import torch
import torch.nn.functional as F


def sample_at(feat: torch.Tensor, uv: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Bilinear lookup at pixel coords -- the query's ID card.

    feat: (T, dim, g, g)      uv: (T, N, 2) pixels      hw: source (H, W)
    returns (T, N, dim)

    `uv` is in original-image pixels, so it is mapped to the [-1, 1] range
    `grid_sample` expects using the ORIGINAL size, not the resized one.

    The frame is resized with `align_corners=False`, so patch j is centred
    on native pixel `(j + 0.5) * W / g - 0.5`, and the lookup has to use the
    same convention: pixel u sits at `(u + 0.5) / W` of the way across.
    Until 2026-10-09 this used `u / (W - 1)` with `align_corners=True`,
    which puts patch 0's centre on pixel 0 and read up to half a patch away
    from the query near the image edges. `border` because a query in the
    outer half-patch has no patch centre beyond it to interpolate towards,
    and the default would blend it with zeros.

    Shared so the trainer can redo the lookup on caches written before then.
    """
    H, W = hw
    norm = (uv + 0.5) / torch.tensor([W, H], dtype=uv.dtype, device=uv.device)
    norm = norm * 2.0 - 1.0
    out = F.grid_sample(feat, norm[:, :, None], padding_mode="border",
                        align_corners=False)                         # (T, dim, N, 1)
    return out[..., 0].transpose(1, 2)
