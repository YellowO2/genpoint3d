"""Put a validation's val and train-split scores into one report."""

from genpoint3d.train.Interval import Interval
from genpoint3d.train.Report import Report


def merge_scores(interval: Interval, scores: dict, fit_scores: dict | None) -> Report:
    """Every train-split score goes in under its val name with `train_` in
    front, and the interval's match loss as `train_match_loss`."""
    merged = dict(scores)
    if fit_scores is not None:
        merged.update({f"train_{k}": v for k, v in fit_scores.items()})
    if interval.match_loss is not None:
        merged["train_match_loss"] = interval.match_loss
    return Report(step=interval.step, train_loss=interval.train_loss, scores=merged)
