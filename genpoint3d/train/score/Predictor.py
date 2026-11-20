"""Predict a batch's tracks and its training loss, without learning."""

import torch

from genpoint3d.train.score.Adder import Adder
from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.score.shape_scores import shape_scores
from genpoint3d.train.score.spawn_points import spawn_points
from genpoint3d.train.tools.batch import fine_inputs, to_device, visual_mask
from genpoint3d.train.tools.regress import regress_loss
from genpoint3d.train.tools.track import track_loss


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
        # A batch that does not say which points are asked has asked them all.
        b.setdefault("asked", torch.ones(traj.shape[0], traj.shape[2], dtype=torch.bool, device=j.device))
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
        add = None
        if j.fill and j.rounds:
            # The stored world adds its points as the clip goes.
            add = Adder(b, fine, vm, j.fill, j.spawn_gap)
        elif j.fill:
            b = spawn_points(model, b, fine, vm, j.fill, j.spawn_gap,
                             lambda bb: (lambda out: (out["tracks"], out["logits"]))(self._track(bb, fine, vm)[1]))
        loss, out = self._track(b, fine, vm, add)
        # The stored world's shapes, on every point it has: before the helpers are dropped.
        shape = shape_scores(b, out["world"], vm) if "world" in out and "mesh" in b else {}
        b, out = _asked_only(b, out)
        return Prediction(batch=b, visual_mask=vm, tracks=out["tracks"], loss=loss.item(),
                          visibility=out["logits"], groups=out["groups"], down=out["down"],
                          down_scored=truly & hide, shape=shape)

    def _track(self, b: dict, fine: dict, vm: torch.Tensor, add=None):
        j = self.judging
        # The batch's own query frames: it may hold points added since `fine` was made.
        cond = dict(context=b["context"], visual_mask=vm, id_card=b["id_card"], patch_xyz=b["patch_xyz"],
                    **{**fine, "query_t": b["query_t"]})
        # The training step's autocast: a bf16 model is scored in bf16.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=j.amp):
            # The loss, as every score, on the asked points only.
            how = dict(rounds=j.rounds, add=add) if j.rounds else dict(window=j.window, overlap=j.overlap, schedule=j.schedule)
            return (track_loss if j.rounds else regress_loss)(
                self.model, b["traj"], b["anchor"], mask=b["visibility"] & b["asked"][:, None], **how, **cond)

    def done(self) -> None:
        self.model.train()


# Each per-point entry of a batch, by the dimension its points are along.
POINT_DIM = {"traj": 2, "visibility": 2, "centre": 2, "norm_offset": 2, "anchor": 1, "query_t": 1, "id_card": 1,
             "segment": 1, "centre_known": 1, "asked": 1}


def _asked_only(b: dict, out: dict) -> tuple[dict, dict]:
    """The batch and the tracking with only the asked points: the helpers the
    clip was filled with are tracked beside them but never scored."""
    asked = b["asked"]
    if asked.all():
        return b, out
    if asked.shape[0] != 1:
        raise ValueError("helper points are scored one clip at a time (--batch 1)")
    keep = asked[0]
    pick = lambda v, dim: v[:, :, keep] if dim == 2 else v[:, keep]
    b = {k: pick(v, POINT_DIM[k]) if k in POINT_DIM and v.dim() > POINT_DIM[k] else v for k, v in b.items()}
    out = {k: None if v is None else pick(v, 1 if k == "groups" else 2) if k in ("tracks", "logits", "groups") else v
           for k, v in out.items()}
    return b, out
