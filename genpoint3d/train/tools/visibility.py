"""What the model should say it can see, and how wrong it is about it."""

import torch
import torch.nn.functional as F


def visible_target(vis: torch.Tensor, visual_mask: torch.Tensor | None) -> torch.Tensor:
    """(B, T, N) True where the model can see the point: the point is in view
    (not occluded, not out of the frame) and its frame keeps its image. A
    forecast frame has no image, so nothing in it is visible to the model."""
    if visual_mask is None:
        return vis
    return vis & visual_mask[:, :, None]


def visibility_error(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The mean binary cross-entropy of (B, T, N) logits against `target`,
    over every (point, frame). In fp32: the logits may come out of autocast."""
    return F.binary_cross_entropy_with_logits(logits.float(), target.float())
