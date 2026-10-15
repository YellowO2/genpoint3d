"""Reframe and normalise one raw Kubric clip.

Step 1 -- input preparation.

Turns a raw `KubricSample` into the tensors the model actually eats. Three
operations, in this order (see `docs/map.html`):

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

import torch

from genpoint3d.NormStats import NormStats
from genpoint3d.prepare.KubricSample import KubricSample
from genpoint3d.prepare.ModelInputs import ModelInputs
from genpoint3d.prepare.tools.geometry import batch_unproject
from genpoint3d.tools.traj_scale import TRAJ_SCALE


def _relative_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """(T, 4, 4) world->cam_t  ==>  (T, 4, 4) cam_0->cam_t.

    `rel[t] = E[t] @ inv(E[0])`, so `rel[t] @ X_cam0 == X_cam_t`. Note `rel[0]`
    is the identity -- frame 0 is the origin by construction. Only *relative*
    pose is used, which is exactly what pose estimation gives you on real video.
    """
    return extrinsics @ torch.linalg.inv(extrinsics[0])[None]


def _to_frame0(points_world: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """(..., 3) world points -> (..., 3) in the frame-0 camera frame."""
    e0 = extrinsics[0]
    return points_world @ e0[:3, :3].T + e0[:3, 3]


def _compute_norm_stats(
    pointmap: torch.Tensor,
    traj_scale: float = TRAJ_SCALE,
    mode: str = "median",
) -> NormStats:
    """Normalisation statistics from an observed pointmap. (DUSt3R.)

    `pointmap`: (T_obs, 3, H, W) unprojected scene points, frame-0 frame.

    Scale only -- the origin stays on the frame-0 camera and is NOT moved to the
    scene centroid. That matters because depth is measured from the camera and
    projection needs it; shifting the origin throws that away. This follows
    DUSt3R's `normalize_pointcloud` (`dust3r/utils/geometry.py`, `norm_mode`
    `avg_dis`), which divides by the distance to the origin and never centres.
    MotionForesight centres as well; we deliberately do not.

    `mode` picks the statistic, as DUSt3R's does:
      median  robust to a skybox at 10 km with no threshold to tune (default)
      mean    DUSt3R's own default; survives a long tail but is dragged by it
      max     the previous behaviour -- ONE far pixel sets the scale for the
              whole scene, which is why it needed a percentile trim to work
    """
    ts = torch.as_tensor(traj_scale, dtype=torch.float32)
    pts = pointmap.permute(0, 2, 3, 1).reshape(-1, 3)
    pts = pts[torch.isfinite(pts).all(dim=-1)]
    mean = torch.zeros(3)                      # origin stays on the camera
    if pts.numel() == 0:
        return NormStats(mean, torch.ones(()), ts)

    dist = pts.norm(dim=-1)                    # distance to the camera
    if mode == "median":
        scale = dist.median()
    elif mode == "mean":
        scale = dist.mean()
    elif mode == "max":
        scale = dist.max()
    else:
        raise ValueError(f"bad norm mode {mode!r}")

    if not torch.isfinite(scale) or scale < 1e-6:
        scale = torch.ones(())
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
    extrinsics = _relative_extrinsics(extrinsics_world)
    traj_cam0 = _to_frame0(traj_world, extrinsics_world)

    # 2. normalise (observed frames only) ----------------------------------
    obs_pointmap = batch_unproject(
        depths[:num_context_frames],
        intrinsics[:num_context_frames],
        extrinsics[:num_context_frames],
    )
    norm = _compute_norm_stats(obs_pointmap, traj_scale=traj_scale)

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
