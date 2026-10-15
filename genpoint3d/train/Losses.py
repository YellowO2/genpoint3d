"""What one micro-batch cost, each loss already divided by `--accum`."""

from dataclasses import dataclass


@dataclass
class Losses:
    method: float       # flow matching's or regression's own loss
    match: float = 0.0  # the matching loss, with --match-learn
