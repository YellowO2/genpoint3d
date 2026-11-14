"""The weighted Kabsch fit, and its robust refit.

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
