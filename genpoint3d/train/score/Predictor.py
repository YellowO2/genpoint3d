"""Predict a batch's tracks and its training loss, without learning."""

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import fine_inputs, known_frame0, to_device, visual_mask
from genpoint3d.train.tools.flow import flow_matching_loss, sample
from genpoint3d.train.tools.regress import refine, regress_loss

# Scoring starts every sample from the same noise, so a checkpoint has one
# score rather than a slightly different one each time it is asked.
EVAL_SEED = 0


class Predictor:
    """Holds the model in eval mode, and the noise every sample starts from,
    until `done` hands the model back for training."""

    def __init__(self, model, judging: ScoreSettings) -> None:
        self.model, self.judging = model, judging
        model.eval()
        self.gen = torch.Generator(device=judging.device).manual_seed(EVAL_SEED)
        # The extra best-of-k samples draw from their own noise, so asking for
        # more of them never changes the first sample, or its score.
        self.more_gen = torch.Generator(device=judging.device).manual_seed(EVAL_SEED + 1)

    def __call__(self, batch: dict) -> Prediction:
        model, j = self.model, self.judging
        b = to_device(batch, j.device, j.amp)
        traj, anchor, vis, ctx, idc, pxyz = (b["traj"], b["anchor"], b["visibility"],
                                             b["context"], b["id_card"], b["patch_xyz"])
        # One cutoff for every clip, so the future scores are all the same
        # forecast; without one, every frame keeps its image.
        cutoff = None
        if j.cutoff:
            if j.cutoff >= traj.shape[1]:
                raise ValueError(f"cutoff {j.cutoff} leaves none of {traj.shape[1]} frames to forecast")
            cutoff = torch.full((traj.shape[0],), j.cutoff, device=j.device)
        vm = visual_mask(b, cutoff)
        # What the model is shown. The scores still use `vm`: hiding is a
        # change to the input, not to which frames count.
        shown = vm
        if j.hide_hidden and vm is not None:
            # A diagnostic that reads the truth, set only by `scripts/evaluate.py
            # --ablate hide-hidden`: each point's image is blanked wherever it is
            # hidden, as a forecast frame is, except frame 0, the query.
            in_view = vis.bool().clone()
            in_view[:, 0] = True
            shown = vm[..., None] & in_view

        kx0 = known_frame0(b) if j.anchor_frame0 else None
        # The same autocast as the training step: a model trained in bf16 and
        # scored in fp32 is being scored on numerics it never saw.
        cond = dict(context=ctx, visual_mask=shown, id_card=idc, patch_xyz=pxyz,
                    **fine_inputs(model, b, j.amp))
        # Regress is deterministic: k samples would be k copies of one answer.
        others = ()
        seen, logits = model.visibility, None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=j.amp):
            if j.method == "regress":
                l, _ = regress_loss(model, traj, anchor, mask=vis, known_x0=kx0,
                                    loss_type=j.loss_type, iters=j.refine_iters,
                                    param=j.param, carry=j.carry, **cond)
                pred = refine(model, anchor, num_frames=traj.shape[1], iters=j.refine_iters,
                              known_x0=kx0, visibility=seen, param=j.param,
                              carry=j.carry, **cond)
            else:
                l, _ = flow_matching_loss(model, traj, anchor, mask=vis, known_x0=kx0,
                                          loss_type=j.loss_type, **cond)
                pred = sample(model, anchor, num_frames=traj.shape[1], steps=j.steps,
                              generator=self.gen, known_x0=kx0, visibility=seen, **cond)
                others = tuple(
                    sample(model, anchor, num_frames=traj.shape[1], steps=j.steps,
                           generator=self.more_gen, known_x0=kx0, **cond)
                    for _ in range(j.samples - 1 if cutoff is not None else 0))
        if seen:
            pred, logits = pred
        return Prediction(batch=b, visual_mask=vm, tracks=pred, loss=l.item(),
                          others=others, visibility=logits)

    def done(self) -> None:
        self.model.train()
