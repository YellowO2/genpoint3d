"""Gather per-clip scores across batches, and average them once at the end."""

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.metrics import clip_mean


class Tally:
    """Every score is a mean over CLIPS, and a clip with no point to score (no
    moving point, when the score is restricted to those) is left out of it, as
    scripts/ref_tapip3d.py does. `moving_clips` is how many were left in.

    Per-clip values, averaged once at the end: a mean of batch means would give
    each clip of a short last batch several times its share.
    """

    def __init__(self) -> None:
        self.loss_sum = self.loss_n = 0.0
        self.vals = {}

    def count(self, prediction: Prediction) -> None:
        """The batch's loss is a mean over it, so it counts once per clip in it."""
        n = prediction.tracks.shape[0]
        self.loss_sum += prediction.loss * n
        self.loss_n += n

    def add(self, values: dict) -> None:
        for k, v in values.items():
            self.vals.setdefault(k, []).append(v.float().cpu())

    def mean(self, judging: ScoreSettings) -> dict:
        out = {"val_loss": self.loss_sum / max(self.loss_n, 1),
               **{k: clip_mean(v) for k, v in self.vals.items()}}
        moving_key = "average_pts_within_thresh" if judging.min_motion_px else "apd_moving"
        if moving_key in self.vals:
            out["moving_clips"] = int((~torch.cat(self.vals[moving_key]).isnan()).sum())
        return out
