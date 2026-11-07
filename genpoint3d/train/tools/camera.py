"""Where a scene-normalised point is drawn in each frame of its clip."""

import torch


def image_projection(intrinsics: torch.Tensor, extrinsics: torch.Tensor, hw: torch.Tensor,
                     scale: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """(B, T, 3, 4): scene-normalised position -> where it is drawn, per frame.

    intrinsics  (B, T, 3, 3) pixels of the NATIVE frame
    extrinsics  (B, T, 4, 4) cam_0 -> cam_t
    hw          (B, 2) native frame size (H, W)
    scale, mean (B,), (B, 3) the clip's normalisation: metres = p * scale + mean

    One matrix for normalisation, pose and intrinsics, giving `(u z, v z, z)`
    with u, v in [0, 1] across the frame, whatever it was resized to.
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

    Behind the camera the depth is floored and uv held within a frame-width
    of the frame, so it reads zeros rather than an infinity.
    """
    q = torch.einsum("btij,btnj->btni", proj[..., :3], pos) + proj[:, :, None, :, 3]
    z = q[..., 2]
    uv = q[..., :2] / z.clamp_min(1e-4)[..., None]
    return uv.clamp(-1.0, 2.0), z


def unproject(uv: torch.Tensor, z: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """The inverse of `project`: uv (B, T, N, 2) in [0, 1] across the frame,
    z (B, T, N) depth in metres, proj (B, T, 3, 4), all fp32 -> (B, T, N, 3)
    scene-normalised positions, solved in float64 to stay sub-millimetre.
    """
    q = torch.cat([uv * z[..., None], z[..., None]], dim=-1).double()
    inv = torch.linalg.inv(proj[..., :3].double())                     # (B, T, 3, 3)
    pos = torch.einsum("btij,btnj->btni", inv, q - proj[:, :, None, :, 3].double())
    return pos.to(uv.dtype)


def query_start(depth: torch.Tensor, uv: torch.Tensor, q: torch.Tensor, intrinsics: torch.Tensor,
                extrinsics: torch.Tensor, hw: tuple[int, int], radius: float) -> torch.Tensor:
    """Each query's start, read off the depth map the model is given: its pixel
    `uv` (N, 2) of the native frame in its frame `q` (N,), at the depth of the
    CNN cell under it in `depth` (T, h, w) metres, held within `radius` as the
    model's cells are -> (N, 3) metres in the frame-0 camera. Only the pixel
    is the query's; the depth is the map's, true or estimated."""
    H, W = hw
    h, w = depth.shape[1:]
    col = (uv[:, 0] / W * w).long().clamp(0, w - 1)
    row = (uv[:, 1] / H * h).long().clamp(0, h - 1)
    z = depth[q, row, col].float().clamp(max=radius)
    K = intrinsics[q].float()                                            # (N, 3, 3)
    x = (uv[:, 0].float() - K[:, 0, 2]) / K[:, 0, 0] * z
    y = (uv[:, 1].float() - K[:, 1, 2]) / K[:, 1, 1] * z
    cam = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)             # (N, 4) in camera q
    return torch.linalg.solve(extrinsics[q].float(), cam[..., None])[:, :3, 0]
