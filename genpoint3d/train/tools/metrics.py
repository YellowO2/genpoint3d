"""Score predicted tracks the way the TAPVid-3D benchmark does.

A torch port of `google-deepmind/tapnet`'s `tapvid3d/evaluation/metrics.py`,
its thresholds and formulas: APD (`average_pts_within_thresh`, the share of
visible points within the threshold), and with a visibility prediction AJ
(`average_jaccard`) and OA (`occlusion_accuracy`). The thresholds are the
pixel thresholds {1, 2, 4, 8, 16} back-projected at each point's depth.
"""

from typing import Optional

import torch

# Their pixel thresholds, and the fixed-metric alternative in metres.
THRESHOLDS = (1, 2, 4, 8, 16)
PIXEL_TO_FIXED_METRIC_THRESH = {1: 0.01, 2: 0.04, 4: 0.16, 8: 0.64, 16: 2.56}

# The hidden-point scores, beside their last-seen baseline.
HIDDEN_KEYS = ("apd_hidden", "apd_hidden_lastseen", "apd_moving_hidden", "apd_moving_hidden_lastseen")


# A point that gets more than this many of the metric's pixels from its query
# position while seen counts as moving (`motion_px`): the moving-point scores.
MOVING_PX = 4.0

def at_query(pts: torch.Tensor, query_t: torch.Tensor) -> torch.Tensor:
    """(B, T, N, 3) -> (B, 1, N, 3): each point at its query frame `query_t` (B, N)."""
    return pts.gather(1, query_t[:, None, :, None].expand(-1, 1, -1, pts.shape[-1]))


def resize_intrinsics(K: torch.Tensor, hw, short_side: int = 256) -> torch.Tensor:
    """(T, 3, 3) pixel intrinsics, rescaled as if the frame's short side were
    `short_side` px: the benchmark's thresholds are for a 256 px short side.
    """
    K = K.clone()
    K[..., :2, :] *= short_side / min(hw)
    return K


def _to_camera_t(pts: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """(B,T,N,3) points in the frame-0 camera -> in each frame's own camera,
    where the threshold's depth is measured."""
    R = extrinsics[..., :3, :3]                     # (B, T, 3, 3)
    t = extrinsics[..., :3, 3]                      # (B, T, 3)
    return torch.einsum("btij,btnj->btni", R, pts) + t[:, :, None]


def _project(pts: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """(B,T,N,3) points in their own frame's camera -> (B,T,N,2) pixels,
    blind to a global scale, so the 2D score leaves depth error out."""
    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    z = pts[..., 2].abs().clamp(min=1e-6)
    return torch.stack([fx[..., None] * pts[..., 0] / z + cx[..., None],
                        fy[..., None] * pts[..., 1] / z + cy[..., None]], dim=-1)


def _scale_factor(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor,
                  scaling: str) -> torch.Tensor:
    """The benchmark's one rescale of the prediction per video ("median" of
    the norms), which monocular depth's unknown scale calls for; "none" off."""
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


def metres_per_px(cam: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """(B, T, N) metres one of the metric's pixels spans at each point, `cam`
    (B, T, N, 3) in the camera that sees it: its depth over the focal length
    (the geometric mean of fx and fy, `intrinsics` (B, T, 3, 3))."""
    focal = (intrinsics[..., 0, 0] * intrinsics[..., 1, 1]).sqrt()
    return (cam[..., 2].abs() / focal[..., None].clamp(min=1e-12)).clamp(min=1e-12)


@torch.no_grad()
def motion_px(gt: torch.Tensor, visible: torch.Tensor, intrinsics: torch.Tensor,
              extrinsics: Optional[torch.Tensor] = None,
              query_t: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(B, N) how far each point ever gets from its query position (frame 0
    without `query_t`) while visible, in the metric's pixels."""
    gt_cam = _to_camera_t(gt, extrinsics) if extrinsics is not None else gt
    mult = metres_per_px(gt_cam, intrinsics)
    start = gt[:, :1] if query_t is None else at_query(gt, query_t)
    disp = (gt - start).norm(dim=-1) / mult                       # (B, T, N)
    return disp.masked_fill(~visible, 0).amax(dim=1)


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
    space: str = "3d",
    per_clip: bool = False,
) -> dict:
    """
    pred, gt      (B, T, N, 3) metres in the frame-0 camera
    visible       (B, T, N) the true visibility: which entries are scored
    intrinsics    (B, T, 3, 3) pixels; extrinsics (B, T, 4, 4) cam_0 -> cam_t
    pred_visible  (B, T, N) the predicted visibility, for AJ and OA
    space         "3d", or "2d": both projected, scored in pixels
    per_clip      (B,) values per metric (NaN for a clip with nothing to
                  score), to average over clips with `clip_mean`; otherwise
                  the batch means as floats
    """
    B = pred.shape[0]
    if space not in ("2d", "3d"):
        raise ValueError(f"unknown space {space!r}")
    pred = pred * _scale_factor(pred, gt, visible, scaling)

    # Depth for the threshold: the z of each point in the camera that sees it.
    gt_cam = _to_camera_t(gt, extrinsics) if extrinsics is not None else gt
    multiplier = metres_per_px(gt_cam, intrinsics)                # (B, T, N)

    if space == "2d":
        pred_cam = _to_camera_t(pred, extrinsics) if extrinsics is not None else pred
        dist_sq = (_project(pred_cam, intrinsics)
                   - _project(gt_cam, intrinsics)).pow(2).sum(-1)
        multiplier = torch.ones_like(multiplier)
    else:
        dist_sq = (pred - gt).pow(2).sum(-1)                      # (B, T, N)
    n_visible = visible.flatten(1).sum(1).clamp(min=1)

    out, fracs, jaccards = {}, [], []
    for thresh in THRESHOLDS:
        pw = (torch.full_like(multiplier, PIXEL_TO_FIXED_METRIC_THRESH[thresh])
              if use_fixed_metric_threshold and space == "3d"
              else thresh * multiplier)
        within = dist_sq < pw.pow(2)
        correct = within & visible

        frac = correct.flatten(1).sum(1) / n_visible
        out[f"pts_within_{thresh}"] = frac
        fracs.append(frac)

        if pred_visible is not None:
            tp = (correct & pred_visible).flatten(1).sum(1)
            # tp + fn is the number of visible truths.
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
    if per_clip:
        # Occlusion accuracy counts every point, so it exists without the mask.
        empty = ~visible.flatten(1).any(1)
        return {k: v if k == "occlusion_accuracy" else v.masked_fill(empty, float("nan"))
                for k, v in out.items()}
    return {k: v.mean().item() for k, v in out.items()}


def clip_mean(values: list[torch.Tensor]) -> float:
    """Mean over clips of per-clip values gathered across batches, skipping the
    NaN of a clip that had nothing to score. NaN if no clip had anything."""
    return torch.cat(values).nanmean().item()


def error_and_motion(pred: torch.Tensor, gt: torch.Tensor, visible: torch.Tensor,
                     intrinsics: torch.Tensor, extrinsics: torch.Tensor, query_t: torch.Tensor):
    """For every visible (point, frame) but its query frame, flat: the error
    and how far the point has moved from its query position, in the metric's
    pixels."""
    mult = metres_per_px(_to_camera_t(gt, extrinsics), intrinsics)
    err = (pred - gt).norm(dim=-1) / mult
    disp = (gt - at_query(gt, query_t)).norm(dim=-1) / mult
    keep = visible.bool() & (torch.arange(gt.shape[1], device=gt.device)[None, :, None] != query_t[:, None])
    return err[keep].float().cpu(), disp[keep].float().cpu()


def motion_table(err: torch.Tensor, disp: torch.Tensor) -> str:
    """Error against distance moved, binned: does the error grow in
    proportion to the motion (a steady error/moved), or level off?"""
    edges = (0, 1, 4, 8, 16, 32, 64, 128, float("inf"))
    rows = [f"    {'moved (px)':>12} {'n':>8} {'median err':>11} {'err/moved':>10} {'<1px':>6} {'<4px':>6}"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (disp >= lo) & (disp < hi)
        if m.sum() < 20:
            continue
        e, d = err[m], disp[m]
        ratio = (e / d.clamp_min(1e-6)).median() if lo > 0 else float("nan")
        rows.append(f"    {f'{lo:g}-{hi:g}':>12} {int(m.sum()):>8} {e.median():>11.2f} {ratio:>10.3f}"
                    f" {(e < 1).float().mean():>6.3f} {(e < 4).float().mean():>6.3f}")
    return "\n".join(rows)
