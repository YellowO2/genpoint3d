"""How often the patch that looks most like each point is the one it is in."""

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.tools.batch import match_targets, to_metres
from genpoint3d.train.tools.matching import (
    match_accuracy, match_accuracy_topk, match_ce, per_clip,
)
from genpoint3d.train.tools.metrics import motion_px


def match_scores(model, prediction: Prediction) -> dict:
    """Per clip (B,) values, by name; empty for a model with no match.

    A property of the features and, with --match-learn, the head; the trunk
    and the sampler have no part in it.
    """
    b, vm = prediction.batch, prediction.visual_mask
    out = {}
    mt = match_targets(model, b, vm)
    if mt is None:
        return out
    gt_m = to_metres(b["traj"], b)
    scores, target, valid = mt
    # The same rule as the MOVING score: more than 4px, ever.
    far = valid & (motion_px(gt_m, b["visibility"], b["intrinsics_256"], b["extrinsics"]) > 4)[:, None]
    for name, mask in (("", valid), ("_moving", far)):
        acc, near = match_accuracy(scores, target, mask)
        out["match_acc" + name] = acc
        out["match_acc_near" + name] = near
        # Always 5, whatever --match-topk is: it says what more
        # candidates would be worth before a run is spent on them.
        out["match_acc_top5_near" + name] = match_accuracy_topk(scores, target, mask, 5)
    if model.match_head is not None:
        out["match_loss"] = per_clip(
            match_ce(scores, model.match_head.tau(), target, valid), valid)
    return out
