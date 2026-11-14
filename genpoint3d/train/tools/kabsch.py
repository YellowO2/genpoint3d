"""The weighted Kabsch fit, its refit, and turns as vectors.

The fit is the turn and shift that best carry one set of points onto another."""

import torch

# A pose needs at least this many points behind it (three not in a line fix a turn).
MIN_POSE_POINTS = 3


def kabsch(w: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> tuple:
    """The fit carrying `src` onto `dst` (..., K, 3, broadcast against the
    weights `w` (..., K)): its turn R (..., 3, 3), never a reflection, the
    centres ms and md (..., 3) and the weight s (...) behind it. A turn
    whatever the weight: what too little means is the caller's."""
    s = w.sum(-1)
    src, dst = src.expand(*w.shape, 3), dst.expand(*w.shape, 3)
    mean = lambda x: (w[..., None] * x).sum(-2) / s.clamp_min(1e-6)[..., None]
    ms, md = mean(src), mean(dst)
    H = torch.einsum("...k,...kx,...ky->...xy", w, src - ms[..., None, :], dst - md[..., None, :])
    U, _, Vh = torch.linalg.svd(H)
    V, Ut = Vh.transpose(-1, -2), U.transpose(-1, -2)
    flip = torch.ones_like(H[..., 0])
    flip[..., 2] = torch.linalg.det(V @ Ut).sign()
    return V @ torch.diag_embed(flip) @ Ut, ms, md, s


def turn_vector(R: torch.Tensor) -> torch.Tensor:
    """(..., 3) the turn R (..., 3, 3) as its axis times its angle (radians),
    for a turn under half a revolution."""
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    angle = torch.acos(cos)
    axis2 = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    # axis2 is the axis times 2 sin(angle); angle / (2 sin(angle)) goes to 1/2 at no turn.
    scale = torch.where(angle > 1e-4, angle / (2 * torch.sin(angle)).clamp_min(1e-8), torch.full_like(angle, 0.5))
    return axis2 * scale[..., None]


def turn_matrix(v: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) the turn about the axis of `v` (..., 3) by its length
    (radians): Rodrigues' formula, `turn_vector`'s inverse."""
    angle = v.norm(dim=-1, keepdim=True)
    k = v / angle.clamp_min(1e-12)
    x, y, z = k.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).view(*v.shape[:-1], 3, 3)
    s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
    return eye + s * K + (1 - c) * (K @ K)


def misfit(R: torch.Tensor, ms: torch.Tensor, md: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """(..., K) how far each point of `dst` sits from where the fit puts its `src`."""
    a = src - ms[..., None, :]
    return ((R[..., None, :, :] @ a[..., None])[..., 0] + md[..., None, :] - dst).norm(dim=-1)


def robust_kabsch(w0: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, refits: int, fit=kabsch) -> tuple:
    """`fit` (`kabsch` or one built on it) refit `refits` times, each point's
    weight `w0` cut by how far it sat from the last fit against the typical
    point, so a few wrong points cannot drag it."""
    R, ms, md, s = fit(w0, src, dst)
    w = w0
    for _ in range(refits):
        # The refit's weights are constants, as reweighted least squares takes them: a point the
        # fit puts exactly in place sits at distance 0, where a distance's gradient is NaN.
        left = misfit(R, ms, md, src, dst).detach()
        typical = (w * left).sum(-1, keepdim=True) / s.clamp_min(1e-6)[..., None] + 1e-6
        w = w0 / (1 + (left / typical).pow(2))
        R, ms, md, s = fit(w, src, dst)
    return R, ms, md, s
