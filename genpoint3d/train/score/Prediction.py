"""One batch on the device, the model's tracks for it, and their loss."""

from dataclasses import dataclass

import torch


@dataclass
class Prediction:
    batch: dict                          # the batch, on the device
    visual_mask: torch.Tensor            # (B, T, 2, 2) which quarters of each frame were seen
    tracks: torch.Tensor                 # (B, T, N, 3) in model units
    loss: float                          # the training objective on the batch
    visibility: torch.Tensor | None = None  # (B, T, N) the visibility head's logits
    groups: torch.Tensor | None = None      # (B, N, E) each point's last group vector
