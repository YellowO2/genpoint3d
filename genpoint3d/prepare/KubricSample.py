"""One raw Kubric clip as read from disk: frames, depths, tracks, cameras."""

from dataclasses import dataclass

import numpy as np


@dataclass
class KubricSample:
    seq_id: str
    frames: np.ndarray        # (T, H, W, 3) uint8 RGB
    depths: np.ndarray        # (T, H, W) float32 z-depth, meters
    traj_3d: np.ndarray       # (N, T, 3) float32 -- world-space 3D trajectory
    coords_2d: np.ndarray     # (N, T, 2) float32 -- 2D pixel projections
    visibility: np.ndarray    # (N, T) bool -- True = visible (NOT occluded)
    intrinsics: np.ndarray    # (T, 3, 3) float32, in pixel units
    extrinsics: np.ndarray    # (T, 4, 4) float32, world-to-camera, OpenCV convention
    query_points_3d: np.ndarray  # (N, 3) float32 -- 3D position at each point's first frame
