"""
Rerun logging helpers, adapted from TAPIP3D's utils/rerun_visualizer.py
(log_video, log_trajectory, setup_visualizer, save_recording) so we don't
depend on cloning their whole repo. Only difference: accepts channel-last
(T, H, W, 3) rgb/frames, matching dataset.py's KubricSample, instead of
their channel-first (T, 3, H, W).
"""

import uuid
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import rerun
import torch
from scipy.spatial.transform import Rotation as R

from genpoint3d.geometry import batch_project, batch_unproject


def setup_visualizer(
    app_name: str = "Gen-Point3D",
    web_port: int = 9091,
    ws_port: int = 9878,
    open_browser: bool = False,
    serve: bool = True,
) -> None:
    rerun.init(app_name, spawn=False, recording_id=uuid.uuid4())
    if serve:
        rerun.serve(open_browser=open_browser, web_port=web_port, ws_port=ws_port, default_blueprint=None)


def save_recording(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rerun.save(str(path))


def log_video(
    entity_name: str,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth: np.ndarray,
    active_mask: Optional[np.ndarray] = None,
) -> None:
    """rgb: (T, H, W, 3) uint8. intrinsics: (T, 3, 3). extrinsics: (T, 4, 4) world-to-camera. depth: (T, H, W).
    active_mask: (T,) bool, frames where False skip the pcd/image (no visual conditioning) but still log camera pose.
    """
    depth_t, intrinsics_t, extrinsics_t = torch.from_numpy(depth), torch.from_numpy(intrinsics), torch.from_numpy(extrinsics)
    pcd = batch_unproject(depth_t, intrinsics_t, extrinsics_t).numpy()  # (T, 3, H, W)

    num_frames, height, width, _ = rgb.shape

    world_from_cam0 = np.linalg.inv(extrinsics[0])
    rerun.log("/", rerun.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rerun.log(
        f"/{entity_name}",
        rerun.Transform3D(translation=world_from_cam0[:3, 3], rotation=rerun.Quaternion(xyzw=R.from_matrix(world_from_cam0[:3, :3]).as_quat())),
        static=True,
    )
    for i in range(num_frames):
        rerun.set_time("frameid", sequence=i)

        world_from_cam = np.linalg.inv(extrinsics[i])
        rerun.log(
            f"/{entity_name}/world/camera",
            rerun.Transform3D(translation=world_from_cam[:3, 3], rotation=rerun.Quaternion(xyzw=R.from_matrix(world_from_cam[:3, :3]).as_quat())),
        )

        # Pinhole (camera pose + intrinsics) is ground truth regardless of masking -- only the
        # actual pixels (image, unprojected point cloud) are what "no visual conditioning" hides.
        rerun.log(
            f"/{entity_name}/world/camera/image",
            rerun.Pinhole(image_from_camera=intrinsics[i], resolution=[width, height], camera_xyz=rerun.ViewCoordinates.RDF),
        )

        # rgb is already zeroed out for masked frames by masking.py, so the image itself needs no
        # special-casing here -- only the point cloud (built from real, unmasked depth) must be hidden.
        if active_mask is None or active_mask[i]:
            rerun.log(f"/{entity_name}/world/pcd", rerun.Points3D(pcd[i].transpose(1, 2, 0).reshape(-1, 3), colors=rgb[i].reshape(-1, 3)))
        else:
            rerun.log(f"/{entity_name}/world/pcd", rerun.Clear(recursive=True))
        rerun.log(f"/{entity_name}/world/camera/image", rerun.Image(rgb[i]))


def log_trajectory(
    entity_name: str,
    track_name: str,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    trajs: np.ndarray,
    visibs: np.ndarray,
    track_len: int = 8,
    cmap_name: str = "rainbow",
) -> None:
    """trajs: (T, N, 3) world points. visibs: (T, N) bool. intrinsics/extrinsics: (T, 3, 3) / (T, 4, 4)."""
    num_frames, num_points, _ = trajs.shape

    intrinsics_t, extrinsics_t, trajs_t = torch.from_numpy(intrinsics), torch.from_numpy(extrinsics), torch.from_numpy(trajs)
    trajs_2d = batch_project(
        trajs_t.reshape(num_frames * num_points, 3),
        torch.repeat_interleave(intrinsics_t, num_points, dim=0),
        torch.repeat_interleave(extrinsics_t, num_points, dim=0),
    ).reshape(num_frames, num_points, 2).numpy()

    cmap = matplotlib.colormaps[cmap_name]
    colors = cmap(matplotlib.colors.Normalize()(trajs[0, :, 1]))  # color by initial height

    for i in range(num_frames):
        rerun.set_time("frameid", sequence=i)

        colors_rgba = colors.copy()
        colors_rgba[~visibs[i], :3] = colors_rgba[~visibs[i], :3] * 0.4  # dim occluded points

        rerun.log(f"/{entity_name}/world/{track_name}/points_3d", rerun.Points3D(trajs[i], colors=colors_rgba, radii=0.005))
        rerun.log(f"/{entity_name}/world/camera/image/{track_name}/points_2d", rerun.Points2D(trajs_2d[i], colors=colors_rgba, radii=1))

        if i >= 1:
            rerun.log(
                f"/{entity_name}/world/camera/image/{track_name}/tracks_2d_{i}",
                rerun.LineStrips2D(trajs_2d[i - 1:i + 1].transpose(1, 0, 2), colors=colors_rgba),
            )
            rerun.log(
                f"/{entity_name}/world/{track_name}/tracks_3d_{i}",
                rerun.LineStrips3D(trajs[i - 1:i + 1].transpose(1, 0, 2), colors=colors_rgba),
            )

        if i >= track_len and track_len > 0:
            rerun.log(f"/{entity_name}/world/camera/image/{track_name}/tracks_2d_{i - track_len}", rerun.Clear(recursive=True))
            rerun.log(f"/{entity_name}/world/{track_name}/tracks_3d_{i - track_len}", rerun.Clear(recursive=True))
