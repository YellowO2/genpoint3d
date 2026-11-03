"""The clips a run trains and validates on, and what the cache holds."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Clips:
    # One cached clip per entry: a path, loaded when drawn (see `ClipDataset`),
    # or the clip's dict itself from a legacy single-file cache.
    train: list[Path | dict]
    val: list[Path | dict]
    feat_dim: int            # the cache's DINOv3 width
    frames: Path | None = None   # the RGB and depth cache for these clips
