"""One batch on the device, the model's tracks for it, and their loss."""

from dataclasses import dataclass

import torch


@dataclass
class Prediction:
    batch: dict                          # the batch, on the device
    visual_mask: torch.Tensor | None     # (B, T) which frames had their image
    tracks: torch.Tensor                 # (B, T, N, 3) in model units
    loss: float                          # the method's objective on the batch
    others: tuple = ()                   # more samples like `tracks`, for best-of-k
