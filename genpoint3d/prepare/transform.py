"""Reframe and normalise one raw Kubric clip: world coordinates into the
fixed frame-0 camera (relative poses only, as real video gives), and a scale
from the observed frames' pointmap (`_compute_norm_stats`). `KubricSample`
stores time first, (T, N, ...).
"""

import torch

from genpoint3d.NormStats import NormStats
from genpoint3d.prepare.KubricSample import KubricSample
from genpoint3d.prepare.ModelInputs import ModelInputs
from genpoint3d.prepare.tools.geometry import batch_unproject
from genpoint3d.tools.traj_scale import TRAJ_SCALE


def _relative_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """(T, 4, 4) world->cam_t -> cam_0->cam_t: `E[t] @ inv(E[0])`."""
    return extrinsics @ torch.linalg.inv(extrinsics[0])[None]


def up_in_frame0(extrinsics_world: torch.Tensor) -> torch.Tensor:
    """(3,) Kubric's up, world +z, in the frame-0 camera frame: (T, 4, 4)
    world->cam_t extrinsics, OpenCV axes."""
    return extrinsics_world[0, :3, :3] @ torch.tensor([0.0, 0.0, 1.0])


def _to_frame0(points_world: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """(..., 3) world points -> (..., 3) in the frame-0 camera frame."""
    e0 = extrinsics[0]
    return points_world @ e0[:3, :3].T + e0[:3, 3]


def _compute_norm_stats(
    pointmap: torch.Tensor,
    traj_scale: float = TRAJ_SCALE,
    mode: str = "median",
) -> NormStats:
    """Normalisation from an observed pointmap (T_obs, 3, H, W) in the frame-0
    camera, as DUSt3R's `normalize_pointcloud`: a scale only (the median,
    mean or max distance to the camera), the origin left on the camera."""
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
    N = sample.traj_3d.shape[1]
    query_t = (torch.zeros(N, dtype=torch.long) if sample.query_t is None
               else torch.from_numpy(sample.query_t).long())               # (N,)
    query_uv = torch.from_numpy(sample.coords_2d).float()[query_t, torch.arange(N)]   # (N, 2)

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
    anchor = norm.apply(traj_cam0[query_t, torch.arange(N)])   # (N, 3) at each query, for conditioning

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
        query_t=query_t,
        up=up_in_frame0(extrinsics_world),
        num_context_frames=num_context_frames,
        norm=norm,
        segment=None if sample.segment is None else torch.from_numpy(sample.segment).long(),
    )
