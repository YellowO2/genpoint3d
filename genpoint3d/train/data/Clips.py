"""The clips a run trains and validates on, and what the caches hold."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Clips:
    train: list[Path]
    val: list[Path]
    feat_dim: int                        # the caches' DINOv3 width
    weights: list[float] | None = None   # how often each train clip is drawn; None: alike
    frames: dict[Path, Path] | None = None   # each cache's RGB and depth
