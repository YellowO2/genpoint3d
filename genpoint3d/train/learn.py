"""Run one micro-batch forward and backward, leaving its gradients."""

import torch

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Losses import Losses
from genpoint3d.train.tools.batch import (
    apd_weight, black_out, fine_inputs, group_pairs, hidden_weight, random_visual_mask, rigid_pairs,
    seen_at, to_device,
)
from genpoint3d.train.tools.camera import project
from genpoint3d.train.tools.regress import regress_loss
from genpoint3d.train.tools.visibility import visible_target


def learn(model, batch: dict, settings, hardware: Hardware) -> Losses:
    """Forward and backward for one micro-batch; the caller steps the
    optimiser once `--accum` of these have added up."""
    amp = hardware.amp
    b = to_device(batch, hardware.device)
    traj, anchor, vis = b["traj"], b["anchor"], b["visibility"]
    # Each clip loses part or all of its frames from a random cutoff, or over
    # a random gap and is seen again after; the CNN sees only what is left.
    vm = random_visual_mask(b, settings.min_context, settings.max_gap)
    b["frames"] = black_out(b["frames"], vm)
    mask, w = hidden_weight(vis, apd_weight(b, traj), settings.hidden_loss)
    extra = dict(vel_weight=settings.vel_loss)
    if settings.rigid_loss:
        extra.update(pairs=rigid_pairs(b), rigid_weight=settings.rigid_loss)
    if settings.group_loss:
        # Each window's pairs, judged on the frames it sees.
        by_object = settings.group_labels == "object"
        extra.update(groups=lambda start, end: group_pairs(b, by_object, frames=(start, end)),
                     group_weight=settings.group_loss)
    # The CNN's map, shared by every backward pass of the step.
    fine = fine_inputs(model, b, amp, grad=True)
    # Some clips are not told which way is down, as real video often is not.
    truly = fine["down_known"]
    told = torch.rand(traj.shape[0]) >= settings.down_hidden
    fine["down_known"] = truly & told.to(hardware.device)
    # What the model says of them is scored; its motion state everywhere.
    extra.update(down_true=fine["down"], down_scored=truly & ~fine["down_known"], down_weight=settings.down_loss,
                 fit_weight=settings.fit_loss)
    if settings.vis_loss:
        # Where each point truly is in the image, against the quarters seen.
        uv, _ = project(model.lookup_pos(traj.float(), anchor.float()), fine["fine_proj"].float())
        extra.update(visible=visible_target(vis, seen_at(vm, uv)), vis_weight=settings.vis_loss)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        # Back-propagated a round at a time, divided by accum so the
        # accumulated gradient is the effective batch's mean.
        loss, parts = regress_loss(model, traj, anchor, mask=mask, weight=w,
                                   backward=1 / settings.accum,
                                   window=settings.window, overlap=settings.overlap,
                                   schedule=settings.schedule, **extra,
                                   context=b["context"], visual_mask=vm, id_card=b["id_card"],
                                   patch_xyz=b["patch_xyz"], **fine)
        # What the backward passes left on the map, the rest of the way into the CNN.
        model.fine.replay(b["frames"], fine["fine_map"])
    return Losses(method=loss.item() / settings.accum, visibility=parts["visibility"].item() / settings.accum)
