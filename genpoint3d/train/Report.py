"""One validation's verdict: the step, the training loss so far, the scores."""

from dataclasses import dataclass


@dataclass
class Report:
    step: int
    # The mean training loss over the validation interval, on the same steps
    # as the scores: the gap between it and val_loss IS the overfitting
    # measurement.
    train_loss: float
    scores: dict
