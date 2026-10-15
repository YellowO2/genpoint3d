"""
3D <-> 2D projection helpers, adapted from TAPIP3D's utils/common_utils.py
(batch_unproject / batch_project), trimmed of their training-only decorators
and third_party dependencies -- kept only what visualize.py needs.

Convention (matches dataset.py): intrinsics in pixel units, extrinsics is
world-to-camera (OpenCV axes: X right, Y down, Z forward).
"""

import torch


def batch_unproject(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """depth: (T, H, W), intrinsics: (T, 3, 3), extrinsics: (T, 4, 4) -> world points (T, 3, H, W)."""
    t, h, w = depth.shape
    v, u = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing="ij")
    # Follow the intrinsics' dtype rather than hardcoding float32: callers that
    # project float64 trajectories pass float64 intrinsics, and einsum will not
    # mix the two.
    uv_homogeneous = torch.stack((u, v, torch.ones_like(u)), dim=-1).to(intrinsics.dtype)  # (h, w, 3)

    K_inv = torch.linalg.inv(intrinsics)
    camera_coords = torch.einsum("nij,xyj->nxyi", K_inv, uv_homogeneous)
    camera_coords = camera_coords * depth[..., None]
    camera_coords = torch.cat((camera_coords, torch.ones_like(camera_coords[..., :1])), dim=-1)

    inv_extrinsics = torch.linalg.inv(extrinsics)
    world_coords = torch.einsum("nij,nxyj->nxyi", inv_extrinsics, camera_coords)
    return world_coords[..., :3].permute(0, 3, 1, 2)


def batch_project(pts3d: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """pts3d: (..., 3) world points -> (..., 2) pixel coords, using per-point intrinsics/extrinsics."""
    pts3d_h = torch.cat((pts3d, torch.ones_like(pts3d[..., :1])), dim=-1)
    pts_camera_h = torch.einsum("...ij,...j->...i", extrinsics, pts3d_h)
    pts_camera = pts_camera_h[..., :3] / pts_camera_h[..., 3:]
    pts_image_h = torch.einsum("...ij,...j->...i", intrinsics, pts_camera)
    return pts_image_h[..., :2] / pts_image_h[..., 2:]
