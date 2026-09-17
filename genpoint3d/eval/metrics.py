"""
TAP-Vid-3D metrics.

Adapted from `google-deepmind/tapnet`, `tapnet/tapvid3d/evaluation/metrics.py`
(Apache-2.0) -- the benchmark authors' own implementation, which every 3D point
tracking paper scores against. TAPIP3D ships a copy of the same file. Rewritten
in torch for our tensor layout `(B, T, N, 3)`; the thresholds, the depth
scaling and the formulas are theirs.

Three numbers come out of it:

  APD  `average_pts_within_thresh` -- fraction of visible points landing inside
       the threshold. Needs ground-truth visibility only, so we can report it
       today.
  AJ   `average_jaccard` -- the headline number in every paper. Counts a point
       only if it is both close enough AND predicted visible, so it needs a
       visibility PREDICTION. We have no visibility head yet, so this is None
       until step 4 lands.
  OA   `occlusion_accuracy` -- also needs the prediction. Same story.

The thresholds are the 2D TAP pixel thresholds {1,2,4,8,16} back-projected into
3D, so tolerance grows with distance: a point 20 m away is allowed more
absolute error than one 2 m away, because both correspond to the same pixel
error. That is why intrinsics are required, and why fixed metric thresholds
(`use_fixed_metric_threshold`) are a different, non-default protocol.
"""

from typing import Optional

import torch

# Their pixel thresholds, and the fixed-metric alternative in metres.
THRESHOLDS = (1, 2, 4, 8, 16)
PIXEL_TO_FIXED_METRIC_THRESH = {1: 0.01, 2: 0.04, 4: 0.16, 8: 0.64, 16: 2.56}


def _to_camera_t(pts: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """(B,T,N,3) points in the frame-0 camera -> each point in ITS OWN frame's camera.

    Only the threshold needs this: it scales with how far the point is from the
    camera that is looking at it, which is not the same as its distance from
    frame 0. Error distances are unaffected, because a rigid transform
    preserves them.
    """
    R = extrinsics[..., :3, :3]                     # (B, T, 3, 3)
    t = extrinsics[..., :3, 3]                      # (B, T, 3)
    return torch.einsum("btij,btnj->btni", R, pts) + t[:, :, None]


def _scale_factor(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor,
                  scaling: str) -> torch.Tensor:
    """Per-video rescaling of the prediction, as the benchmark protocol requires.

    Monocular depth is scale-ambiguous, so a method can be perfectly correct up
    to one global factor. `median` removes exactly that freedom and nothing
    more. With Kubric's ground-truth depth `none` is defensible, but then the
    number is no longer the published protocol.
    """
    if scaling == "none":
        return torch.ones(pred.shape[0], 1, 1, 1, device=pred.device)

    pred_n = pred.norm(dim=-1).masked_fill(~valid, float("nan"))
    gt_n = gt.norm(dim=-1).masked_fill(~valid, float("nan"))
    flat = lambda x: x.reshape(x.shape[0], -1)
    if scaling == "median":
        num, den = flat(gt_n).nanmedian(dim=1).values, flat(pred_n).nanmedian(dim=1).values
    elif scaling == "mean":
        num, den = flat(gt_n).nanmean(dim=1), flat(pred_n).nanmean(dim=1)
    else:
        raise ValueError(f"unknown scaling {scaling!r}")
    return (num / den.clamp(min=1e-12))[:, None, None, None]


@torch.no_grad()
def tapvid3d_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    visible: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: Optional[torch.Tensor] = None,
    pred_visible: Optional[torch.Tensor] = None,
    scaling: str = "median",
    use_fixed_metric_threshold: bool = False,
) -> dict:
    """
    pred, gt      (B, T, N, 3) metric positions in the frame-0 camera frame
    visible       (B, T, N) bool, ground truth -- True means visible
    intrinsics    (B, T, 3, 3) pixel units
    extrinsics    (B, T, 4, 4) cam_0 -> cam_t. Without it the threshold uses
                  frame-0 depth, which is wrong for points that move in z.
    pred_visible  (B, T, N) bool. Without it only APD is returned; AJ and OA
                  need a visibility prediction by definition.

    Returns a dict of floats averaged over the batch.
    """
    B = pred.shape[0]
    pred = pred * _scale_factor(pred, gt, visible, scaling)

    # Depth for the threshold: the z of each point in the camera that sees it.
    gt_cam = _to_camera_t(gt, extrinsics) if extrinsics is not None else gt
    depth = gt_cam[..., 2].abs()                                  # (B, T, N)
    focal = (intrinsics[..., 0, 0] * intrinsics[..., 1, 1]).sqrt()  # (B, T)
    multiplier = depth / focal[..., None].clamp(min=1e-12)

    dist_sq = (pred - gt).pow(2).sum(-1)                          # (B, T, N)
    n_visible = visible.flatten(1).sum(1).clamp(min=1)

    out, fracs, jaccards = {}, [], []
    for thresh in THRESHOLDS:
        pw = (torch.full_like(multiplier, PIXEL_TO_FIXED_METRIC_THRESH[thresh])
              if use_fixed_metric_threshold else thresh * multiplier)
        within = dist_sq < pw.pow(2)
        correct = within & visible

        frac = correct.flatten(1).sum(1) / n_visible
        out[f"pts_within_{thresh}"] = frac
        fracs.append(frac)

        if pred_visible is not None:
            tp = (correct & pred_visible).flatten(1).sum(1)
            # true positives + false negatives is just the number of visible
            # ground-truth points, which is cheaper than counting both.
            fp = (((~visible) | (~within)) & pred_visible).flatten(1).sum(1)
            jac = tp / (n_visible + fp).clamp(min=1)
            out[f"jaccard_{thresh}"] = jac
            jaccards.append(jac)

    out["average_pts_within_thresh"] = torch.stack(fracs).mean(0)
    if jaccards:
        out["average_jaccard"] = torch.stack(jaccards).mean(0)
        out["occlusion_accuracy"] = (
            (pred_visible == visible).flatten(1).sum(1) / (pred.shape[1] * pred.shape[2])
        )

    assert all(v.shape == (B,) for v in out.values())
    return {k: v.mean().item() for k, v in out.items()}
