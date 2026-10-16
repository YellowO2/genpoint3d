"""Where a scene-normalised point is drawn in each frame of its clip."""

import torch


def image_projection(intrinsics: torch.Tensor, extrinsics: torch.Tensor, hw: torch.Tensor,
                     scale: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """(B, T, 3, 4): scene-normalised position -> where it is drawn, per frame.

    intrinsics  (B, T, 3, 3) pixels of the NATIVE frame
    extrinsics  (B, T, 4, 4) cam_0 -> cam_t
    hw          (B, 2) native frame size (H, W)
    scale, mean (B,), (B, 3) the clip's normalisation: metres = p * scale + mean

    One matrix does normalisation, camera pose and intrinsics, in that order,
    and its output is `(u * z, v * z, z)` with u, v in [0, 1] across the frame
    -- see `project`. Dividing by the frame size here is what makes the result
    independent of the resolution anything was resized to.
    """
    K = intrinsics.float().clone()
    K[..., 0, :] /= hw[:, 1, None, None].float()
    K[..., 1, :] /= hw[:, 0, None, None].float()
    A = K @ extrinsics.float()[..., :3, :]                            # metres -> image
    lin = A[..., :3] * scale.float()[:, None, None, None]
    off = A[..., :3] @ mean.float()[:, None, :, None] + A[..., 3:]
    return torch.cat([lin, off], dim=-1)


def project(pos: torch.Tensor, proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pos (B, T, N, 3) scene-normalised, proj (B, T, 3, 4), both fp32 ->
    (B, T, N, 2) `(u, v)` in [0, 1] across the frame, and (B, T, N) depth.

    A point behind the camera has no pixel. Its depth is floored and the
    result held to within one frame-width of the frame, so it reads a window
    of zeros rather than an infinity.
    """
    q = torch.einsum("btij,btnj->btni", proj[..., :3], pos) + proj[:, :, None, :, 3]
    z = q[..., 2]
    uv = q[..., :2] / z.clamp_min(1e-4)[..., None]
    return uv.clamp(-1.0, 2.0), z


def unproject(uv: torch.Tensor, z: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """The inverse of `project`: uv (B, T, N, 2) in [0, 1] across the frame,
    z (B, T, N) depth in metres, proj (B, T, 3, 4), all fp32 -> (B, T, N, 3)
    scene-normalised positions.

    Solved in float64: the matrix carries the clip's scale and the frame size,
    and a point that comes back a millimetre off is the error this exists to
    remove.
    """
    q = torch.cat([uv * z[..., None], z[..., None]], dim=-1).double()
    inv = torch.linalg.inv(proj[..., :3].double())                     # (B, T, 3, 3)
    pos = torch.einsum("btij,btnj->btni", inv, q - proj[:, :, None, :, 3].double())
    return pos.to(uv.dtype)
