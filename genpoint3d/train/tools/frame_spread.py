"""The centre and spread of a clip's frame-0 cell positions."""

import torch


def frame_spread(first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(b, 3, h, w) the clip's frame-0 cell positions -> their mean (b, 3) and
    spread (b,), the scale TAPIP3D brings each clip to (`norm_mode:
    isotropic`). Frame 0 alone, which every clip sees: TAPIP3D uses the whole
    window, which a forecast would leak through."""
    first = first.float()
    mean = first.mean(dim=(2, 3))
    return mean, (first - mean[:, :, None, None]).std(dim=(1, 2, 3)).clamp_min(1e-6)
