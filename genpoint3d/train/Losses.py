"""What one micro-batch cost, each loss already divided by `--accum`."""

from dataclasses import dataclass


@dataclass
class Losses:
    method: float            # the tracking loss
    visibility: float = 0.0  # the visibility head's loss, with --vis-loss
    centre: float = 0.0      # the arrows' to the centres of mass, with --centre-loss (in `method`)
