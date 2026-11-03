"""Predict a batch's tracks and its training loss, without learning."""

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import fine_inputs, known_frame0, to_device, visual_mask
from genpoint3d.train.tools.regress import refine, regress_loss


class Predictor:
    """Holds the model in eval mode until `done` hands it back for training."""

    def __init__(self, model, judging: ScoreSettings) -> None:
        self.model, self.judging = model, judging
        model.eval()

    def __call__(self, batch: dict) -> Prediction:
        model, j = self.model, self.judging
        b = to_device(batch, j.device)
        traj, anchor, vis = b["traj"], b["anchor"], b["visibility"]
        # One cutoff for every clip; without one, every frame keeps its image.
        cutoff = None
        if j.cutoff:
            if j.cutoff >= traj.shape[1]:
                raise ValueError(f"cutoff {j.cutoff} leaves none of {traj.shape[1]} frames to forecast")
            cutoff = torch.full((traj.shape[0],), j.cutoff, device=j.device)
        vm = visual_mask(b, cutoff)
        shown = vm
        if j.hide_hidden:
            # A diagnostic that reads the truth (`evaluate.py --ablate
            # hide-hidden`): each point blanked where it is truly hidden.
            in_view = vis.bool().clone()
            in_view[:, 0] = True
            shown = vm[..., None] & in_view
        kx0 = known_frame0(b)
        cond = dict(context=b["context"], visual_mask=shown, id_card=b["id_card"], patch_xyz=b["patch_xyz"],
                    **fine_inputs(model, b, j.amp))
        how = dict(iters=j.refine_iters, window=j.window, overlap=j.overlap, schedule=j.schedule)
        # The training step's autocast: a bf16 model is scored in bf16.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=j.amp):
            loss, _ = regress_loss(model, traj, anchor, kx0, mask=vis, **how, **cond)
            tracks, logits = refine(model, anchor, traj.shape[1], kx0, **how, **cond)
        return Prediction(batch=b, visual_mask=vm, tracks=tracks, loss=loss.item(), visibility=logits)

    def done(self) -> None:
        self.model.train()
