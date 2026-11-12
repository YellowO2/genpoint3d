"""Measure a model on clips it does not learn from."""

import torch

from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.score.Tally import Tally
from genpoint3d.train.score.track_scores import track_scores
from genpoint3d.train.score.visibility_scores import visibility_scores


@torch.no_grad()
def score(model, loader, judging: ScoreSettings, split: str, gpus=None) -> dict | None:
    """The mean scores over `loader`'s clips, logged under `split`: val_loss
    (the position error over visible frames, as training measures it), APD in metres (TAPVid-3D's
    `average_pts_within_thresh`), its static baseline, and AJ and OA with a
    visibility head. With `gpus`, each process scores its own share of the
    clips and rank 0 gets the scores of all (the others None)."""
    predictor = Predictor(model, judging)
    tally = Tally(split)
    for batch in loader:
        prediction = predictor(batch)
        tally.count(prediction)
        tally.add(track_scores(prediction, judging))
        tally.add(visibility_scores(prediction))
    predictor.done()
    if gpus is not None and gpus.world > 1:
        tallies = gpus.gather(tally)
        if not gpus.lead:
            return None
        tally.merge(tallies[1:])
    return tally.mean(judging)
