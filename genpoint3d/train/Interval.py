"""The training between two validations: where it ends and its losses."""

from dataclasses import dataclass


@dataclass
class Interval:
    step: int
    train_loss: float         # the mean since the last validation, against val_loss
    vis_loss: float | None    # likewise for the visibility head; None without it
