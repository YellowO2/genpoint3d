"""What one micro-batch cost, each loss already divided by `--accum`."""

from dataclasses import dataclass


@dataclass
class Losses:
    method: float            # the tracking loss
    visibility: float = 0.0  # the visibility head's loss, with --vis-loss
