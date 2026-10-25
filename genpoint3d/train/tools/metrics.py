"""Score predicted tracks the way the TAP-Vid-3D benchmark does.

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
       visibility PREDICTION, so it is scored only for a model trained
       with the visibility head (`--vis-loss`).
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

# The pixel thresholds APD averages over; the moving-points score is logged at
# each of them as well.
MOVING_THRESHOLDS = (1, 2, 4, 8, 16)

# The scores of points nobody can see on the observed frames, all points and
# moving ones: hidden, then split into occluded and out of the frame, each
# beside its two last-seen baselines; and the share of hidden entries that
# are out of the frame.
HIDDEN_KEYS = tuple(f"{name}{suffix}"
                    for name in ("apd_hidden", "apd_moving_hidden", "apd_still_hidden",
                                 "apd_occluded", "apd_moving_occluded", "apd_still_occluded",
                                 "apd_outside", "apd_moving_outside", "apd_still_outside",
                                 "apd_returns", "apd_moving_returns", "apd_still_returns",
                                 "apd_gone", "apd_moving_gone", "apd_still_gone")
                    for suffix in ("", "_lastseen", "_lastcv", "_interp", "_buddy", "_buddypred")) + (
    "hidden_outside_frac",) + tuple(f"apd_moving_seen_{part}{s}" for part in ("early", "mid", "late")
                                    for s in ("", "_static"))


def resize_intrinsics(K: torch.Tensor, hw, short_side: int = 256) -> torch.Tensor:
    """(T, 3, 3) pixel intrinsics, rescaled as if the frame's short side were
    `short_side` px. Focal lengths and principal point scale together.

    The benchmark defines its pixel thresholds on a frame whose short side is
    256 px, and scores with intrinsics resized to match (TAPIP3D
    `evaluation/metrics.py`, following DeepMind's `evaluate_model.py`).
    """
    K = K.clone()
    K[..., :2, :] *= short_side / min(hw)
    return K


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


def _project(pts: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """(B,T,N,3) points in their own frame's camera -> (B,T,N,2) pixels.

    A global scale on the prediction cancels here, because perspective division
    only sees x/z and y/z. That is exactly why the 2D score is the right way to
    ask whether depth is what we are getting wrong: it is blind to the one
    degree of freedom depth is most likely to be wrong by.
    """
    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    z = pts[..., 2].abs().clamp(min=1e-6)
    return torch.stack([fx[..., None] * pts[..., 0] / z + cx[..., None],
                        fy[..., None] * pts[..., 1] / z + cy[..., None]], dim=-1)


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


def in_image(gt: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor,
             hw: torch.Tensor) -> torch.Tensor:
    """(B, T, N) bool: is each true position inside its frame's image?

    gt          (B, T, N, 3) in the frame-0 camera
    intrinsics  (B, T, 3, 3) native pixels, to match `hw`
    hw          (B, 2) the native frame's height and width

    A point behind the camera is outside, wherever its projection lands.
    """
    gt_cam = _to_camera_t(gt, extrinsics)
    uv = _project(gt_cam, intrinsics)
    h, w = hw[:, 0, None, None], hw[:, 1, None, None]
    return ((gt_cam[..., 2] > 0)
            & (uv[..., 0] >= 0) & (uv[..., 0] < w)
            & (uv[..., 1] >= 0) & (uv[..., 1] < h))


@torch.no_grad()
def motion_px(gt: torch.Tensor, visible: torch.Tensor, intrinsics: torch.Tensor,
              extrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(B, N): how far each point ever gets from its frame-0 position, in the
    metric's own units -- pixels back-projected at the point's depth.

    A point that never moves is tracked perfectly by predicting "it stays where
    it started". If most query points are like that, the average says little
    about tracking; this is what lets a score be restricted to the rest.
    """
    gt_cam = _to_camera_t(gt, extrinsics) if extrinsics is not None else gt
    focal = (intrinsics[..., 0, 0] * intrinsics[..., 1, 1]).sqrt()
    mult = (gt_cam[..., 2].abs() / focal[..., None].clamp(min=1e-12)).clamp(min=1e-12)
    disp = (gt - gt[:, :1]).norm(dim=-1) / mult                   # (B, T, N)
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
    pred, gt      (B, T, N, 3) metric positions in the frame-0 camera frame
    visible       (B, T, N) bool, ground truth -- True means visible
    intrinsics    (B, T, 3, 3) pixel units
    extrinsics    (B, T, 4, 4) cam_0 -> cam_t. Without it the threshold uses
                  frame-0 depth, which is wrong for points that move in z.
    pred_visible  (B, T, N) bool. Without it only APD is returned; AJ and OA
                  need a visibility prediction by definition.
    space         "3d" for APD3D, or "2d" to project both tracks into the image
                  and score the original TAP-Vid pixel thresholds. Same points,
                  same clips, same thresholds -- the only difference is whether
                  depth error counts. The pair separates "the model cannot find
                  the point" from "the model finds it but misplaces it in z".

    per_clip      return a (B,) tensor per metric instead of the batch mean,
                  NaN for a clip whose `visible` is empty. A score averaged
                  across batches has to be built from these: a mean of batch
                  means over-weights a short last batch, and an empty clip has
                  no score rather than a score of 0. Average with `clip_mean`.

    Returns a dict of floats averaged over the batch.
    """
    B = pred.shape[0]
    if space not in ("2d", "3d"):
        raise ValueError(f"unknown space {space!r}")
    pred = pred * _scale_factor(pred, gt, visible, scaling)

    # Depth for the threshold: the z of each point in the camera that sees it.
    gt_cam = _to_camera_t(gt, extrinsics) if extrinsics is not None else gt
    depth = gt_cam[..., 2].abs()                                  # (B, T, N)
    focal = (intrinsics[..., 0, 0] * intrinsics[..., 1, 1]).sqrt()  # (B, T)
    multiplier = depth / focal[..., None].clamp(min=1e-12)

    if space == "2d":
        # In pixels the thresholds ARE the thresholds -- no back-projection,
        # which is the whole point of the 3D multiplier above.
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
                     intrinsics: torch.Tensor, extrinsics: torch.Tensor):
    """For every visible (point, frame) after frame 0: the error and how far the
    point has moved since frame 0, both in the metric's own pixels (metres
    divided by depth / focal, at the point's own depth). Flat (M,) tensors, for
    `motion_table`."""
    gt_cam = _to_camera_t(gt, extrinsics)
    focal = (intrinsics[..., 0, 0] * intrinsics[..., 1, 1]).sqrt()
    mult = (gt_cam[..., 2].abs() / focal[..., None].clamp(min=1e-12)).clamp(min=1e-12)
    err = (pred - gt).norm(dim=-1) / mult
    disp = (gt - gt[:, :1]).norm(dim=-1) / mult
    keep = visible.bool().clone()
    keep[:, 0] = False
    return err[keep].float().cpu(), disp[keep].float().cpu()


def appearance_change(gt: torch.Tensor, visible: torch.Tensor, extrinsics: torch.Tensor,
                      tol: float = 0.02, min_buddies: int = 4):
    """For every visible (point, frame) after frame 0, as `error_and_motion`
    picks them: how far the point has turned towards or away from the camera
    since frame 0, in degrees, and how much nearer or further it is, as
    |log(depth_t / depth_0)|. Flat (M,) tensors; the angle is NaN where it
    cannot be told.

    The cache keeps no object ids, so the turn is the rotation that best maps
    the point's rigid companions -- points whose distance to it never strays
    more than `tol` metres -- from frame 0 onto frame t, both in the camera
    looking at them. Still background points are companions of each other,
    so a camera that turns turns them too. Fewer than `min_buddies`
    companions gives NaN.
    """
    cam = _to_camera_t(gt, extrinsics)                                    # (B, T, N, 3)
    d = (gt[:, :, :, None] - gt[:, :, None]).norm(dim=-1)
    w = ((d - d[:, :1]).abs().amax(dim=1) < tol).float()                  # (B, N, N)
    w = w * (1 - torch.eye(gt.shape[2], device=gt.device))
    a = cam[:, :1, None] - cam[:, :1, :, None]                            # (B, 1, N, N, 3)
    b = cam[:, :, None] - cam[:, :, :, None]                              # (B, T, N, N, 3)
    H = torch.einsum("bnj,bsnjx,btnjy->btnxy", w, a, b).float()
    U, _, Vh = torch.linalg.svd(H)
    flip = torch.ones_like(U[..., 0])
    flip[..., 2] = torch.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2)).sign()
    R = Vh.transpose(-1, -2) @ torch.diag_embed(flip) @ U.transpose(-1, -2)
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cos))
    angle = angle.masked_fill((w.sum(-1) < min_buddies)[:, None], float("nan"))
    depth = cam[..., 2].abs().clamp_min(1e-6)
    scale = (depth / depth[:, :1]).log().abs()
    keep = visible.bool().clone()
    keep[:, 0] = False
    return angle[keep].cpu(), scale[keep].float().cpu()


def change_table(err: torch.Tensor, disp: torch.Tensor, change: torch.Tensor,
                 edges: tuple, name: str) -> str:
    """Error / distance moved for points that moved about as far, split by how
    much they changed in another way: does that change, at a fixed distance,
    make the error worse?"""
    moved = ((16, 32), (32, 64), (64, 128), (128, float("inf")))
    cuts = list(zip(edges[:-1], edges[1:]))
    rows = [f"    {'moved (px)':>12} " + " ".join(f"{f'{name} {lo:g}-{hi:g}':>22}" for lo, hi in cuts)]
    for mlo, mhi in moved:
        cells = []
        for lo, hi in cuts:
            m = (disp >= mlo) & (disp < mhi) & (change >= lo) & (change < hi)
            r = (err[m] / disp[m]).median().item() if m.sum() >= 20 else float("nan")
            cells.append(f"{r:>12.3f} n={int(m.sum()):<7}")
        rows.append(f"    {f'{mlo:g}-{mhi:g}':>12} " + " ".join(f"{c:>22}" for c in cells))
    return "\n".join(rows)


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
