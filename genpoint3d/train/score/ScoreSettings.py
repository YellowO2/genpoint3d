"""How a model is scored: how it predicts, and which points count."""

from dataclasses import dataclass

import torch


@dataclass
class ScoreSettings:
    device: torch.device
    amp: bool = False
    window: int = 0               # frames tracked at a time, 0 = the whole clip
    overlap: int = 0              # frames each window shares with the one before
    schedule: str = "ffbb"        # the sweeps over the windows
    space: str = "3d"             # "2d" scores the projections in pixels
    oracle_axis: str = ""         # "x", "y" or "z": that channel set to the truth
    scaling: str = "median"       # the protocol's global rescale, or "none"
    min_motion_px: float = 0.0    # score only points that move more than this
    cutoff: int = 0               # frames from here on get no image; 0 = none
    spawn: int = 0                # points added on new surface after the first frame, at most; 0 = none
    spawn_first: int = 0          # points added in the first frame's gaps, at most
    spawn_gap: float = 0.05       # a gap that wide (a share of the frame's width) gets a point
