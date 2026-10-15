"""Learn from one micro-batch: the losses' gradients, left in the model."""

import torch

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Losses import Losses
from genpoint3d.train.tools.batch import (
    apd_weight, fine_inputs, known_frame0, match_targets, to_device,
)
from genpoint3d.train.tools.flow import flow_matching_loss
from genpoint3d.train.tools.matching import match_ce
from genpoint3d.train.tools.regress import regress_loss


def learn(model, batch: dict, settings, hardware: Hardware) -> Losses:
    """Forward and backward for one micro-batch, by `--method`; the caller
    steps the optimiser once `--accum` of these have added up."""
    amp = hardware.amp
    b = to_device(batch, hardware.device, amp)
    traj, anchor, vis, ctx, idc, pxyz = (b["traj"], b["anchor"], b["visibility"],
                                         b["context"], b["id_card"], b["patch_xyz"])
    # All-True: pure tracking. Every frame keeps its image.
    vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=hardware.device) if ctx is not None else None
    w = apd_weight(b, traj) if settings.depth_scaled_loss else None
    kx0 = known_frame0(b) if settings.anchor_frame0 else None
    # Before any backward pass of the step, and shared by all of them.
    fine = fine_inputs(model, b, amp, grad=True)
    match = 0.0
    if settings.match_learn:
        # Its own forward and backward, once per step and ahead of the
        # method: the scores do not depend on the sample, so regress
        # would otherwise compute the same loss --refine-iters times,
        # and one path serves both methods. The gradient adds to the
        # method's in the same buffers. Logged apart from `loss`, so
        # train_loss stays comparable with runs that do not have it.
        scores, target, valid = match_targets(model, b, vm)
        ml = (match_ce(scores, model.match_head.tau(), target, valid).sum()
              / valid.sum().clamp(min=1))
        (ml * settings.match_loss_weight / settings.accum).backward()
        match = ml.item() / settings.accum
        del scores, ml
    # Divided by accum so the accumulated gradient is the mean over
    # the effective batch, not its sum -- otherwise the gradient norm
    # scales with accum and clipping would bite differently.
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        if settings.method == "regress":
            # Back-propagated inside, one iteration at a time, so the
            # step holds one forward pass in memory rather than
            # --refine-iters of them. The loss comes back detached.
            loss, _ = regress_loss(model, traj, anchor, mask=vis, weight=w,
                                   known_x0=kx0, loss_type=settings.loss_type,
                                   iters=settings.refine_iters, backward=1 / settings.accum,
                                   context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz,
                                   **fine)
        else:
            loss, _ = flow_matching_loss(model, traj, anchor, mask=vis, weight=w,
                                         known_x0=kx0, loss_type=settings.loss_type,
                                         context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz,
                                         **fine)
    if settings.method == "flow":
        (loss / settings.accum).backward()
    if fine:
        # Every backward pass above stopped at the feature map. This
        # takes what they left there the rest of the way into the CNN.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            model.fine.replay(b["frames"], fine["fine_map"])
        del fine
    return Losses(method=loss.item() / settings.accum, match=match)
