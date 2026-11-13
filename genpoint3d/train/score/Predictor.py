"""Predict a batch's tracks and its training loss, without learning."""

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.score.spawn_points import spawn_all
from genpoint3d.train.tools.batch import fine_inputs, to_device, visual_mask
from genpoint3d.train.tools.regress import regress_loss


class Predictor:
    """Holds the model in eval mode until `done` hands it back for training."""

    def __init__(self, model, judging: ScoreSettings) -> None:
        self.model, self.judging = model, judging
        # Clips served so far: every other one is not told which way is down.
        self.served = 0
        model.eval()

    def __call__(self, batch: dict) -> Prediction:
        model, j = self.model, self.judging
        b = to_device(batch, j.device)
        traj = b["traj"]
        # One cutoff for every clip; without one, every frame keeps its image.
        cutoff = None
        if j.cutoff:
            if j.cutoff >= traj.shape[1]:
                raise ValueError(f"cutoff {j.cutoff} leaves none of {traj.shape[1]} frames to forecast")
            cutoff = torch.full((traj.shape[0],), j.cutoff, device=j.device)
        vm = visual_mask(b, cutoff)
        fine = fine_inputs(model, b, j.amp)
        # Every other clip not told which way is down; what the model says of it is scored.
        hide = (self.served + torch.arange(traj.shape[0], device=j.device)) % 2 == 1
        self.served += traj.shape[0]
        truly = fine["down_known"]
        fine["down_known"] = truly & ~hide
        loss, out = self._track(b, fine, vm)
        if j.spawn:
            # Again with points added where none covers the frame; only the clip's own are scored.
            more = spawn_all(model, b, fine, vm, j.spawn, j.spawn_gap,
                             lambda bb: (lambda out: (out["tracks"], out["logits"]))(self._track(bb, {**fine, "query_t": bb["query_t"]}, vm)[1]))
            _, out = self._track(more, {**fine, "query_t": more["query_t"]}, vm)
            N = traj.shape[2]
            out = {k: None if v is None else v[:, :N] if k == "groups" else v[:, :, :N] if k in ("tracks", "logits") else v
                   for k, v in out.items()}
        return Prediction(batch=b, visual_mask=vm, tracks=out["tracks"], loss=loss.item(),
                          visibility=out["logits"], groups=out["groups"], down=out["down"],
                          down_scored=truly & hide)

    def _track(self, b: dict, fine: dict, vm: torch.Tensor):
        j = self.judging
        cond = dict(context=b["context"], visual_mask=vm, id_card=b["id_card"], patch_xyz=b["patch_xyz"], **fine)
        # The training step's autocast: a bf16 model is scored in bf16.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=j.amp):
            return regress_loss(self.model, b["traj"], b["anchor"], mask=b["visibility"], window=j.window,
                                overlap=j.overlap, schedule=j.schedule, **cond)

    def done(self) -> None:
        self.model.train()
