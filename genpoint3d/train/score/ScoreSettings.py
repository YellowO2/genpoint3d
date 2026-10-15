"""How a model is scored: how it predicts, and which points count."""

from dataclasses import dataclass

import torch


@dataclass
class ScoreSettings:
    device: torch.device
    amp: bool = False
    method: str = "flow"          # "flow" samples, "regress" refines
    steps: int = 50               # flow: Euler steps from noise to tracks
    refine_iters: int = 4         # regress: corrections to the first guess
    anchor_frame0: bool = False   # hold frame 0 at its known position
    loss_type: str = "l2"         # the objective val_loss is measured by
    space: str = "3d"             # "2d" scores the projections in pixels
    oracle_axis: str = ""         # "x", "y" or "z": that channel set to the truth
    scaling: str = "median"       # the protocol's global rescale, or "none"
    min_motion_px: float = 0.0    # score only points that move more than this
    cutoff: int = 0               # frames from here on get no image; 0 = none
