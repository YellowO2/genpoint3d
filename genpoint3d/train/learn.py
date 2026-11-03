"""Run one micro-batch forward and backward, leaving its gradients."""

import torch

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Losses import Losses
from genpoint3d.train.tools.batch import (
    apd_weight, fine_inputs, hidden_weight, random_cutoff, rigid_groups, rigid_pairs,
    to_device, visual_mask,
)
from genpoint3d.train.tools.regress import regress_loss
from genpoint3d.train.tools.visibility import visible_target


def learn(model, batch: dict, settings, hardware: Hardware) -> Losses:
    """Forward and backward for one micro-batch; the caller steps the
    optimiser once `--accum` of these have added up."""
    amp = hardware.amp
    b = to_device(batch, hardware.device)
    traj, anchor, vis = b["traj"], b["anchor"], b["visibility"]
    # Each clip sees its frames before its own random cutoff and forecasts the rest.
    vm = visual_mask(b, random_cutoff(b, settings.min_context))
    mask, w = hidden_weight(vis, apd_weight(b, traj), settings.hidden_loss)
    extra = dict(vel_weight=settings.vel_loss)
    if settings.rigid_loss:
        extra.update(pairs=rigid_pairs(b), rigid_weight=settings.rigid_loss)
    if settings.group_loss:
        extra.update(groups=rigid_groups(b), group_weight=settings.group_loss)
    if settings.vis_loss:
        extra.update(visible=visible_target(vis, vm), vis_weight=settings.vis_loss)
    # The CNN's map, shared by every backward pass of the step.
    fine = fine_inputs(model, b, amp, grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        # Back-propagated a round at a time, divided by accum so the
        # accumulated gradient is the effective batch's mean.
        loss, parts = regress_loss(model, traj, anchor, mask=mask, weight=w,
                                   iters=settings.refine_iters, backward=1 / settings.accum,
                                   window=settings.window, overlap=settings.overlap,
                                   schedule=settings.schedule, **extra,
                                   context=b["context"], visual_mask=vm, id_card=b["id_card"],
                                   patch_xyz=b["patch_xyz"], **fine)
        # What the backward passes left on the map, the rest of the way into the CNN.
        model.fine.replay(b["frames"], fine["fine_map"])
    return Losses(method=loss.item() / settings.accum, visibility=parts["visibility"].item() / settings.accum)
