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
    query_points_3d: np.ndarray  # (N, 3) float32 -- 3D position at each point's query frame
    query_t: np.ndarray = None   # (N,) int64 -- each point's query frame; None is frame 0
    segment: np.ndarray = None   # (N,) int -- each point's object, 0 the background; None if unknown
    centre_3d: np.ndarray = None  # like traj_3d -- each point's object's centre of mass, NaN off objects; None if unknown
