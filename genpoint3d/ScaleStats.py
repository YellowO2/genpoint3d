"""Every candidate scale of one clip's scene, measured once."""

from dataclasses import dataclass
from typing import Optional

import torch

from genpoint3d.NormStats import NormStats
from genpoint3d.tools.traj_scale import TRAJ_SCALE


@dataclass
class ScaleStats:
    """Every candidate normalisation statistic, measured once at preprocess time.

    All of them are kept because they cost four floats and computing any one of
    them needs the depth maps -- the single expensive part of preprocessing.
    Storing only the one currently in favour is what forced a re-run last time
    the choice changed.

    `centroid` and `max_from_centroid` describe the MotionForesight scheme
    (centre the scene, divide by its bounding radius). `mean_dist` and
    `median_dist` are distances to the camera for the DUSt3R scheme, which
    divides but never centres -- see `compute_norm_stats`.
    """

    centroid: torch.Tensor          # (3,) metric centre of the observed scene
    max_from_centroid: torch.Tensor  # () furthest point from that centre
    mean_dist: torch.Tensor          # () mean distance to the camera
    median_dist: torch.Tensor        # () median distance to the camera

    def norm(self, mode: str = "median", traj_scale: float = TRAJ_SCALE) -> NormStats:
        """Build the `NormStats` for one scheme. `mode` matches `compute_norm_stats`."""
        ts = torch.as_tensor(traj_scale, dtype=torch.float32)
        zero = torch.zeros(3)
        if mode == "median":
            return NormStats(zero, self.median_dist, ts)
        if mode == "mean":
            return NormStats(zero, self.mean_dist, ts)
        if mode == "centroid_max":   # the pre-2026-09-18 behaviour
            return NormStats(self.centroid, self.max_from_centroid, ts)
        raise ValueError(f"bad norm mode {mode!r}")

    def to_dict(self) -> dict:
        return {f"stats_{k}": getattr(self, k) for k in
                ("centroid", "max_from_centroid", "mean_dist", "median_dist")}

    @classmethod
    def from_dict(cls, d: dict) -> "Optional[ScaleStats]":
        if "stats_median_dist" not in d:
            return None
        return cls(**{k: d[f"stats_{k}"] for k in
                      ("centroid", "max_from_centroid", "mean_dist", "median_dist")})

    @classmethod
    def measure(cls, pointmap: torch.Tensor) -> "ScaleStats":
        """Measure every candidate statistic from an observed pointmap.

        `pointmap`: (T_obs, 3, H, W) unprojected scene points, frame-0 frame.
        Computed once because it needs the depth maps, which are the expensive
        part of preprocessing; picking a scheme afterwards is then free.
        """
        pts = pointmap.permute(0, 2, 3, 1).reshape(-1, 3)
        pts = pts[torch.isfinite(pts).all(dim=-1)]
        if pts.numel() == 0:
            one = torch.ones(())
            return cls(torch.zeros(3), one, one, one)

        centroid = pts.mean(dim=0)
        dist = pts.norm(dim=-1)                 # to the camera, which is the origin
        return cls(
            centroid=centroid,
            max_from_centroid=(pts - centroid).norm(dim=-1).max(),
            mean_dist=dist.mean(),
            median_dist=dist.median(),
        )
