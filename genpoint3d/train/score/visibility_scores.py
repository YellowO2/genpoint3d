"""Check the visibility head: does it know when it can see the point?"""

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.tools.visibility import visible_target


def visibility_scores(prediction: Prediction) -> dict:
    """Per clip (B,) values, by name; empty for a model with no head.

    vis_acc          how often the head is right, over every (point, frame).
                     With a cutoff, a forecast frame counts as not visible.
    vis_acc_hidden   over the entries where the point is truly hidden,
                     occluded or out of the frame, on the frames with images
    vis_acc_visible  over the entries where the point is in view, likewise

    A logit above 0 says visible. A clip with no entry of a kind is NaN.
    """
    logits = prediction.visibility
    if logits is None:
        return {}
    vis = prediction.batch["visibility"]
    observed = prediction.visual_mask.flatten(2).all(-1)[:, :, None]      # whole frames: scoring's masks
    target = visible_target(vis, observed)
    right = ((logits > 0) == target).float()
    return {"vis_acc": right.flatten(1).mean(1),
            "vis_acc_hidden": per_clip(right, ~vis & observed),
            "vis_acc_visible": per_clip(right, vis & observed)}


def per_clip(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B, T, N) values -> (B,) mean over each clip's valid entries; NaN for none."""
    return (x * valid).flatten(1).sum(1) / valid.flatten(1).sum(1)
