"""Score a model on clips it does not learn from: val loss, APD and the match."""

import torch

from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.score.Tally import Tally
from genpoint3d.train.score.match_scores import match_scores
from genpoint3d.train.score.track_scores import track_scores


@torch.no_grad()
def score(model, loader, judging: ScoreSettings) -> dict:
    """Two numbers, both standard -- no homemade units.

    val_loss  the SAME objective as training (`judging.method`), on held-out
              clips. Plot it against training loss: the gap is overfitting.
    APD       `average_pts_within_thresh` from the TAP-Vid-3D benchmark, scored
              in METRES -- see `tools/metrics.py`. The number other papers
              report. AJ and OA need a visibility prediction, so they appear
              only once the visibility head exists.

    `apd_static` is the benchmark's own Static Baseline (TAPVid-3D, table 3):
    take the query point's 3D position and assume it never moves. That is a
    real bar rather than a formality -- it scores 9.4 there, above TAPIR-3D's
    5.9 and not far below SpatialTracker's 15.5.
    """
    predictor = Predictor(model, judging)
    tally = Tally()
    for batch in loader:
        prediction = predictor(batch)
        tally.count(prediction)
        tally.add(match_scores(model, prediction))
        tally.add(track_scores(prediction, judging))
    predictor.done()
    scores = tally.mean(judging)
    return scores
