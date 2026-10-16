"""Check the visibility head: does it know when it can see the point?"""

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.tools.matching import per_clip
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
    vis, vm = prediction.batch["visibility"], prediction.visual_mask
    target = visible_target(vis, vm)
    right = ((logits > 0) == target).float()
    observed = vm[:, :, None] if vm is not None else vis.new_ones(())
    return {"vis_acc": right.flatten(1).mean(1),
            "vis_acc_hidden": per_clip(right, ~vis & observed),
            "vis_acc_visible": per_clip(right, vis & observed)}
