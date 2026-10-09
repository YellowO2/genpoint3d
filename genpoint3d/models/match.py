"""
Supervising the match: "which patch looks like me" as a trained answer.

`PointDiT` scores every patch of every frame against each point's frame-0
feature and hands the trunk the winner. On the raw backbone features that
winner is the right patch about half the time, and nothing trains it: the
winner is an argmax, and the features are cached frozen.

Three pieces, all method-agnostic -- none of them sees the trajectory being
denoised or refined, only the image features and the ground truth:

    MatchHead        a small residual on the features, for the comparison only
    true_patch       which patch each ground-truth point really falls in
    match_ce,        cross-entropy of the scores against that patch, and how
    match_accuracy   often the argmax is that patch
"""

import torch
import torch.nn.functional as F
from torch import nn

from genpoint3d.eval.metrics import _to_camera_t
from genpoint3d.models.layers import bounded_exp, log_param


def raw_cosine(feats: torch.Tensor, id_card: torch.Tensor) -> torch.Tensor:
    """(B, T, P, C) patches and (B, N, C) templates -> (B, T, N, P) cosine.

    The patches are divided by their norm after the product rather than
    normalised first: a normalised copy of `feats` is a second full-size tensor.
    """
    q = F.normalize(id_card, dim=-1)
    corr = torch.einsum("bnc,btpc->btnp", q, feats)
    return corr / torch.linalg.vector_norm(feats, dim=-1).clamp_min(1e-6)[:, :, None]


class MatchHead(nn.Module):
    """Cosine between `f' = normalize(f + MLP(f))` for template and patches.

    One MLP, shared by both sides, `Linear(C, h) -> GELU -> Linear(h, C)`. The
    last layer starts at exactly zero, so at step 0 `f' = normalize(f)` and the
    scores are the raw cosine, bit for bit.

    `f'` is never built for the patches. They are (B, T, P, C) with P up to
    2304 -- 1.4 GB in fp32 at batch 16 -- and the obvious code holds three more
    of those for the backward pass. With `d = W2 h` the residual and `h` the
    hidden activation, everything the cosine needs is an h-wide product:

        q' . f'    = q' . f  +  (q' W2) . h
        |f'|^2     = |f|^2  +  (2 f W2 + h W2^T W2) . h

    so the largest new tensors are (B, T, P, h), a sixth of the size at h = 64.
    The last layer has no bias for this reason: a bias is a full-width term.
    """

    def __init__(self, feat_dim: int, hidden: int = 64, tau: float = 0.07) -> None:
        super().__init__()
        self.inp = nn.Linear(feat_dim, hidden)
        self.out = nn.Linear(hidden, feat_dim, bias=False)
        nn.init.zeros_(self.out.weight)
        # Temperature of the matching loss only. The trunk reads the scores
        # through its own (`match_log_tau`, `corr_log_scale`).
        self.loss_log_tau = log_param(tau)

    def tau(self) -> torch.Tensor:
        return bounded_exp(self.loss_log_tau, 0.01, 1.0)

    def forward(self, feats: torch.Tensor, id_card: torch.Tensor) -> torch.Tensor:
        """feats (B, T, P, C), id_card (B, N, C), fp32 -> (B, T, N, P) cosine."""
        W2 = self.out.weight                                          # (C, h)
        q = F.normalize(id_card + F.gelu(self.inp(id_card)) @ W2.T, dim=-1)
        h = F.gelu(self.inp(feats))                                   # (B, T, P, h)
        corr = (torch.einsum("bnc,btpc->btnp", q, feats)
                + torch.einsum("bnh,btph->btnp", q @ W2, h))
        # |f'| as |f| times a factor that is exactly 1 while W2 is zero.
        norm = torch.linalg.vector_norm(feats, dim=-1)                # (B, T, P)
        extra = ((2 * (feats @ W2) + h @ (W2.T @ W2)) * h).sum(-1)
        norm = norm * (1 + extra / norm.pow(2).clamp_min(1e-12)).clamp_min(1e-12).sqrt()
        return corr / norm.clamp_min(1e-6)[:, :, None]


@torch.no_grad()
def true_patch(gt: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor,
               hw: torch.Tensor, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Which patch of its own frame each ground-truth point is drawn in.

    gt          (B, T, N, 3) metres, frame-0 camera
    intrinsics  (B, T, 3, 3) pixels of the NATIVE frame
    extrinsics  (B, T, 4, 4) cam_0 -> cam_t
    hw          (B, 2) native frame size (H, W)
    grid        patches per side

    returns     (B, T, N) long, row-major patch index, and (B, T, N) bool:
                the point is in front of the camera and inside the frame.

    Projected, not looked up as the nearest `patch_xyz`: the patch a point is
    DRAWN in is a fact about the camera, and nearest-in-3D picks a neighbour
    whenever the point sits near a patch border or the patch centre lies on a
    different surface. The frame is squashed to a square before encoding, so
    patch j covers `[j, j+1) * W / grid` of the native width whatever the
    aspect ratio, and row and column are scaled separately.

    Pixel coordinates are the intrinsics' own: the principal point is W / 2,
    so the image spans [0, W) and pixel i covers [i, i+1).

    The index of a point outside the frame is clamped into range so it can be
    gathered with; the mask is what says not to use it.
    """
    cam = _to_camera_t(gt.float(), extrinsics.float())
    z = cam[..., 2]
    K = intrinsics.float()
    u = K[..., 0, 0, None] * cam[..., 0] / z.clamp_min(1e-6) + K[..., 0, 2, None]
    v = K[..., 1, 1, None] * cam[..., 1] / z.clamp_min(1e-6) + K[..., 1, 2, None]
    H, W = hw[:, 0, None, None].float(), hw[:, 1, None, None].float()
    inside = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    col = (u * grid / W).floor().long().clamp(0, grid - 1)
    row = (v * grid / H).floor().long().clamp(0, grid - 1)
    return row * grid + col, inside


def match_ce(scores: torch.Tensor, tau: torch.Tensor, target: torch.Tensor,
             valid: torch.Tensor) -> torch.Tensor:
    """(B, T, N) cross-entropy of `scores / tau` over the frame's P patches
    against the true patch; zero where `valid` is False.

    Only the valid rows are put through the softmax: its output is as large as
    the scores, and an occluded point has no patch to be right about.
    """
    out = scores.new_zeros(valid.shape)
    if valid.any():
        out[valid] = F.cross_entropy(scores[valid] / tau, target[valid], reduction="none")
    return out


def per_clip(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B, T, N) values -> (B,) mean over each clip's valid entries. NaN for a
    clip with none, which `clip_mean` leaves out."""
    return (x * valid).flatten(1).sum(1) / valid.flatten(1).sum(1)


@torch.no_grad()
def match_accuracy(scores: torch.Tensor, target: torch.Tensor,
                   valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per clip (B,): how often the best-scoring patch is the true one, and how
    often it is within one patch of it (the true patch's 3x3 on the grid)."""
    g = int(round(scores.shape[-1] ** 0.5))
    top = scores.argmax(-1)
    near = (((top // g) - (target // g)).abs() <= 1) & (((top % g) - (target % g)).abs() <= 1)
    return per_clip(top == target, valid), per_clip(near, valid)
