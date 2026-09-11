"""
Step 1 -- input preparation.

Turns a raw `KubricSample` into the tensors the model actually eats. Three
operations, in this order (see `docs/MAP.md`):

1. **Reframe** -- move everything from Kubric world coordinates into the
   *frozen frame-0 camera* frame. Origin sits at frame 0's camera and never
   moves again, so the frame is static (ego-motion removed) but constructible
   from relative poses alone -- which is all you get on real video.

2. **Normalise** -- scale/centre using statistics from the *observed* frames
   only, so nothing leaks from the future.

3. **Scale the target** -- one further global constant so the trajectory the
   model denoises has unit variance, as flow matching requires. Positions stay
   absolute (what the paper does); the frame-0 anchor is kept separately as
   conditioning rather than being subtracted out.

The inverse (`denormalise`) turns model output back into metric coordinates
for visualisation and metrics.

Convention note: `KubricSample` stores time first -- `(T, N, ...)` -- despite
what its dataclass comments say.
"""

from dataclasses import dataclass

import torch

from genpoint3d.data.kubric import KubricSample
from genpoint3d.geometry import batch_project, batch_unproject


# Dataset-level constant: the spread of scene-normalised trajectories.
# Scene normalisation makes *geometry* O(1), but leaves trajectories at ~0.26 --
# and flow matching mixes them with `x0 ~ N(0, I)`, so the target must be near
# unit variance or the interpolant is mostly noise. One global number, not
# per-clip, so no sample's own motion leaks into its normalisation (and so a
# fast clip stays genuinely faster than a slow one). Recalibrate with
# `scripts/calibrate_motion_scale.py` whenever the training set changes.
TRAJ_SCALE = 0.2136  # calibrated on 2 clips -- redo on the full training set


@dataclass
class NormStats:
    """Affine normalisation for one clip.

    `mean`/`scale` are per-clip and come from the observed scene, so a tabletop
    and a street both land in the same box. `traj_scale` is a single global
    constant applied on top, only to the trajectory, to bring the denoising
    target to unit variance.
    """

    mean: torch.Tensor        # (3,) metric centre of the observed scene
    scale: torch.Tensor       # () scalar, metres per normalised unit
    traj_scale: torch.Tensor  # () scalar, normalised units per model unit

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """(..., 3) metric -> (..., 3) scene-normalised. For geometry."""
        return (pts - self.mean) / self.scale

    def invert(self, pts: torch.Tensor) -> torch.Tensor:
        """(..., 3) scene-normalised -> (..., 3) metric."""
        return pts * self.scale + self.mean

    def apply_traj(self, pts: torch.Tensor) -> torch.Tensor:
        """(..., 3) metric -> (..., 3) model units. For the denoising target."""
        return self.apply(pts) / self.traj_scale

    def invert_traj(self, pts: torch.Tensor) -> torch.Tensor:
        """(..., 3) model units -> (..., 3) metric."""
        return self.invert(pts * self.traj_scale)


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


def relative_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """(T, 4, 4) world->cam_t  ==>  (T, 4, 4) cam_0->cam_t.

    `rel[t] = E[t] @ inv(E[0])`, so `rel[t] @ X_cam0 == X_cam_t`. Note `rel[0]`
    is the identity -- frame 0 is the origin by construction. Only *relative*
    pose is used, which is exactly what pose estimation gives you on real video.
    """
    return extrinsics @ torch.linalg.inv(extrinsics[0])[None]


def to_frame0(points_world: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """(..., 3) world points -> (..., 3) in the frame-0 camera frame."""
    e0 = extrinsics[0]
    return points_world @ e0[:3, :3].T + e0[:3, 3]


def compute_norm_stats(
    pointmap: torch.Tensor,
    traj_scale: float = TRAJ_SCALE,
    lo: float = 2.0,
    hi: float = 98.0,
) -> NormStats:
    """Normalisation statistics from an observed pointmap. (MotionForesight.)

    `pointmap`: (T_obs, 3, H, W) unprojected scene points, frame-0 frame.

    Trims the far/near tails by depth percentile before measuring, so a few
    skybox pixels at 10000 m cannot dominate the scale.
    """
    ts = torch.as_tensor(traj_scale, dtype=torch.float32)
    pts = pointmap.permute(0, 2, 3, 1).reshape(-1, 3)
    pts = pts[torch.isfinite(pts).all(dim=-1)]
    if pts.numel() == 0:
        return NormStats(torch.zeros(3), torch.ones(()), ts)

    z = pts[:, 2]
    z_lo, z_hi = torch.quantile(z, torch.tensor([lo / 100.0, hi / 100.0], dtype=z.dtype))
    inliers = pts[(z >= z_lo) & (z <= z_hi)]
    if inliers.numel() == 0:
        inliers = pts

    mean = inliers.mean(dim=0)
    scale = (inliers - mean).norm(dim=-1).max()
    if not torch.isfinite(scale) or scale < 1e-6:
        scale = torch.ones_like(scale)
    return NormStats(mean=mean, scale=scale, traj_scale=ts)


def transform(
    sample: KubricSample,
    num_context_frames: int,
    traj_scale: float = TRAJ_SCALE,
) -> ModelInputs:
    """`KubricSample` -> `ModelInputs`. The whole of step 1.

    `num_context_frames` (T_C) is the tracking/forecasting cutoff: frames
    `[0, T_C)` keep their visual conditioning, the rest are masked. Only these
    frames may contribute to the normalisation statistics.
    """
    T = sample.frames.shape[0]
    if not 1 <= num_context_frames <= T:
        raise ValueError(f"num_context_frames must be in [1, {T}], got {num_context_frames}")

    frames = torch.from_numpy(sample.frames)
    depths = torch.from_numpy(sample.depths).float()
    intrinsics = torch.from_numpy(sample.intrinsics).float()
    extrinsics_world = torch.from_numpy(sample.extrinsics).float()
    traj_world = torch.from_numpy(sample.traj_3d).float()          # (T, N, 3)
    visibility = torch.from_numpy(sample.visibility).bool()        # (T, N)
    query_uv = torch.from_numpy(sample.coords_2d[0]).float()       # (N, 2)

    # 1. reframe -----------------------------------------------------------
    extrinsics = relative_extrinsics(extrinsics_world)
    traj_cam0 = to_frame0(traj_world, extrinsics_world)

    # 2. normalise (observed frames only) ----------------------------------
    obs_pointmap = batch_unproject(
        depths[:num_context_frames],
        intrinsics[:num_context_frames],
        extrinsics[:num_context_frames],
    )
    norm = compute_norm_stats(obs_pointmap, traj_scale=traj_scale)

    # 3. scale the target to unit variance ---------------------------------
    traj = norm.apply_traj(traj_cam0)   # (T, N, 3) absolute, in model units
    anchor = norm.apply(traj_cam0[0])   # (N, 3) frame-0 position, for conditioning

    return ModelInputs(
        seq_id=sample.seq_id,
        frames=frames,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        traj=traj,
        anchor=anchor,
        visibility=visibility,
        query_uv=query_uv,
        num_context_frames=num_context_frames,
        norm=norm,
    )


def scene_pointmap(inputs: ModelInputs, normalise: bool = True) -> torch.Tensor:
    """(T, 3, H, W) per-pixel scene geometry in the frame-0 camera frame.

    Feeds the stage-1 "feature cloud" in step 3. Kept out of `ModelInputs`
    because it is large and cheap to recompute.
    """
    pts = batch_unproject(inputs.depths, inputs.intrinsics, inputs.extrinsics)
    if normalise:
        pts = inputs.norm.apply(pts.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return pts


def project_to_frames(inputs: ModelInputs, traj_metric: torch.Tensor) -> torch.Tensor:
    """(T, N, 3) metric frame-0-frame points -> (T, N, 2) pixels in their own frame.

    Used in step 3 to decide where each point token samples DINOv3 features.
    """
    T = traj_metric.shape[0]
    return batch_project(
        traj_metric,
        inputs.intrinsics[:T, None],
        inputs.extrinsics[:T, None],
    )
