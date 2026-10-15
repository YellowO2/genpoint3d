"""One raw clip, reframed and normalised: what the cache is made from."""

from dataclasses import dataclass

import torch

from genpoint3d.NormStats import NormStats


@dataclass
class ModelInputs:
    """Everything downstream of step 1. All torch, all frame-0-camera frame."""

    seq_id: str

    # --- conditioning signal (stage 1 of the map) ---
    frames: torch.Tensor      # (T, H, W, 3) uint8 RGB
    depths: torch.Tensor      # (T, H, W) float32 z-depth, metres
    intrinsics: torch.Tensor  # (T, 3, 3) pixel units
    extrinsics: torch.Tensor  # (T, 4, 4) frame-0-camera -> frame-t camera

    # --- what the model denoises (stage 2) ---
    traj: torch.Tensor        # (T, N, 3) normalised absolute position <- TARGET
    anchor: torch.Tensor      # (N, 3) scene-normalised frame-0 position
    visibility: torch.Tensor  # (T, N) bool, True = visible

    # --- query pointers (stage 3) ---
    query_uv: torch.Tensor    # (N, 2) pixel location in frame 0

    num_context_frames: int   # T_C -- frames with visual conditioning
    norm: NormStats

    @property
    def traj_metric(self) -> torch.Tensor:
        """(T, N, 3) absolute metric positions in the frame-0 camera frame.

        The full inverse of the transform, and exactly how model output is
        turned back into metres for visualisation and metrics.
        """
        return self.norm.invert_traj(self.traj)
