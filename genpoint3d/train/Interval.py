"""The training between two validations: where it ends and its losses."""

from dataclasses import dataclass


@dataclass
class Interval:
    step: int
    # The mean training loss since the last validation, on the same steps as
    # the scores: the gap between it and val_loss IS the overfitting
    # measurement.
    train_loss: float
    # The mean matching loss over the same steps, with --match-learn; None
    # without it.
    match_loss: float | None
    # The mean visibility loss over the same steps, with --vis-loss; None
    # without it.
    vis_loss: float | None
    # The mean depth loss over the same steps, with --depth-loss; None
    # without it.
    depth_loss: float | None = None
