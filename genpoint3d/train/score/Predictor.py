"""Predict a batch's tracks and its training loss, without learning."""

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import fine_inputs, to_device, visual_mask
from genpoint3d.train.tools.regress import regress_loss


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
        cond = dict(context=b["context"], visual_mask=vm, id_card=b["id_card"], patch_xyz=b["patch_xyz"],
                    **fine_inputs(model, b, j.amp))
        # The training step's autocast: a bf16 model is scored in bf16.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=j.amp):
            loss, out = regress_loss(model, traj, anchor, mask=vis, iters=j.refine_iters, window=j.window,
                                     overlap=j.overlap, schedule=j.schedule, **cond)
        return Prediction(batch=b, visual_mask=vm, tracks=out["tracks"], loss=loss.item(),
                          visibility=out["logits"])

    def done(self) -> None:
        self.model.train()
