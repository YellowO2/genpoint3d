"""Write down a clip's tracks, cameras and scale, in metres."""

from genpoint3d.CachedClip import CachedClip
from genpoint3d.ScaleStats import ScaleStats
from genpoint3d.prepare.ModelInputs import ModelInputs


def describe_clip(inputs: ModelInputs, pointmap, settings) -> CachedClip:
    """The clip as the cache holds it, without features."""
    x = inputs
    return CachedClip(
        seq_id=x.seq_id,
        traj=x.traj_metric.clone(),       # (T, N, 3) metres <- TARGET
        anchor=x.traj_metric[0].clone(),  # (N, 3) frame-0 position, metres
        visibility=x.visibility.clone(),  # (T, N) bool
        query_uv=x.query_uv.clone(),      # (N, 2) where to read the ID card
        hw=tuple(x.frames.shape[1:3]),    # needed to map uv into the grid
        intrinsics=x.intrinsics.clone(),  # (T, 3, 3) pixel units
        extrinsics=x.extrinsics.clone(),  # (T, 4, 4) cam_0 -> cam_t
        stats=ScaleStats.measure(pointmap),  # every candidate scale, measured once
        points=settings.points,
    )
