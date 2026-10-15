"""The clips a run trains and validates on, and what the cache holds."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Clips:
    # One cached clip per entry: a path, loaded when drawn (see `ClipDataset`),
    # or the clip's dict itself from a legacy single-file cache.
    train: list[Path | dict]
    val: list[Path | dict]
    # The cache's feature width. The encoder chooses its own width, and it need
    # not equal the model's.
    feat_dim: int | None
    # Whether the model is given DINOv3 features at all.
    has_feats: bool
    # The directory of RGB frames for these clips, for the relative finder;
    # None without it.
    frames: Path | None = None
