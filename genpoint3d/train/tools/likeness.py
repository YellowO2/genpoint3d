"""How alike two points' group vectors say they are: one piece or not."""

import torch

# Over this likeness, two points are on one piece.
SAME = 0.5


def likeness(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2 sigmoid(-|a - b|^2) over the last axis, `a` and `b` broadcast against
    each other: 1 for the same vector, falling to 0 as they part."""
    return 2 * torch.sigmoid(-(a - b).pow(2).sum(-1))
