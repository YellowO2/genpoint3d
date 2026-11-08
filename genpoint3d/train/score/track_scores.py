"""Score predicted tracks in metres against the truth and baselines.

Every reference is each point's query frame, where it was handed over: the
static baseline holds it there, and a point is moving if it gets more than 4
px from there while visible.
"""

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import rigid_groups, to_metres
from genpoint3d.train.tools.metrics import at_query, motion_px, tapvid3d_metrics


def track_scores(prediction: Prediction, judging: ScoreSettings) -> dict:
    """Per clip (B,) values, by name:

    average_pts_within_thresh, pts_within_{t}   APD over visible points
    apd_static                                  each point held at its query position
    apd_moving, apd_moving_static               the moving points, no rescale
    average_jaccard, occlusion_accuracy         with the visibility head
    apd_hidden, apd_moving_hidden (+ _lastseen) where the point is hidden, beside
                                                the true position it was last seen at
    with a cutoff C, on the observed (_obs) and forecast (_fut) frames apart,
    beside two baselines on the forecast frames from the true positions at
    C-1 (_stop) and C-2, C-1 (_cv, constant speed)
    group_right, group_found                    with group vectors: of the points the
                                                model puts with each point, the share
                                                truly on its piece, and of those truly
                                                on it, the share the model puts with it
    """
    b, j = prediction.batch, judging
    out = {}
    vis, q = b["visibility"], b["query_t"]
    gt_m, pred_m = to_metres(b["traj"], b), to_metres(prediction.tracks, b)
    static = at_query(gt_m, q).expand_as(gt_m)
    if j.min_motion_px:
        # Only the points that move: what the metric counts, not the loss.
        moving = motion_px(gt_m, vis, b["intrinsics_256"], b["extrinsics"], q) > j.min_motion_px
        out["moving_frac"] = moving.float().mean(1)
        vis = vis & moving[:, None]
        # How much of the true motion is predicted (ratio), and in which direction (cos).
        d_gt, d_pr = gt_m - static, pred_m - static
        on_vis = lambda x: (x * vis).flatten(1).sum(1)
        out["disp_ratio"] = on_vis(d_pr.norm(dim=-1)) / on_vis(d_gt.norm(dim=-1))
        out["disp_cos"] = (on_vis(torch.nn.functional.cosine_similarity(d_pr, d_gt, dim=-1))
                           / vis.flatten(1).sum(1))
    if j.oracle_axis:
        # One channel of the answer replaced by the truth: which axis the error is in.
        pred_m = pred_m.clone()
        a = "xyz".index(j.oracle_axis)
        pred_m[..., a] = gt_m[..., a]
    m = tapvid3d_metrics(pred_m, gt_m, vis, b["intrinsics_256"], b["extrinsics"], space=j.space,
                         scaling=j.scaling, per_clip=True)
    out.update({k: v for k, v in m.items() if k.startswith("pts_within_") or k == "average_pts_within_thresh"})
    out["apd_static"] = _apd(static, gt_m, vis, b, j.scaling, j.space)
    if prediction.visibility is not None:
        out.update(_jaccard_scores(pred_m, gt_m, prediction, j))
    if prediction.groups is not None:
        out.update(_group_scores(prediction.groups, b))
    if j.space == "3d" and not j.min_motion_px:
        moving = motion_px(gt_m, vis, b["intrinsics_256"], b["extrinsics"], q) > 4
        mv = vis & moving[:, None]
        out["apd_moving"] = _apd(pred_m, gt_m, mv, b, "none")
        out["apd_moving_static"] = _apd(static, gt_m, mv, b, "none")
        out.update(_hidden_scores(pred_m, gt_m, vis, moving, q, b, j))
        if j.cutoff:
            out.update(_split_scores(pred_m, gt_m, vis, mv, b, j))
    return out


def _group_scores(groups: torch.Tensor, b: dict) -> dict:
    """The group vectors against the training labels' pieces (`rigid_groups`):
    an affinity 2 sigmoid(-|v_i - v_j|^2) above 0.5 says "same piece". Pairs
    of a point with itself are left out."""
    v = groups.float()
    said = 2 * torch.sigmoid(-(v[:, :, None] - v[:, None]).pow(2).sum(-1)) > 0.5
    true = rigid_groups(b)
    off = ~torch.eye(v.shape[1], dtype=torch.bool, device=v.device)
    said, true = said & off, true & off
    both = (said & true).flatten(1).sum(1).float()
    return {"group_right": both / said.flatten(1).sum(1).clamp_min(1),
            "group_found": both / true.flatten(1).sum(1).clamp_min(1)}


def _jaccard_scores(pred_m, gt_m, prediction: Prediction, j: ScoreSettings) -> dict:
    """AJ and OA, with a logit above 0 saying visible, over every point on the
    frames that have their image (before the cutoff)."""
    b = prediction.batch
    T = j.cutoff or gt_m.shape[1]
    m = tapvid3d_metrics(pred_m[:, :T], gt_m[:, :T], b["visibility"][:, :T],
                         b["intrinsics_256"][:, :T], b["extrinsics"][:, :T],
                         pred_visible=prediction.visibility[:, :T] > 0,
                         space=j.space, scaling=j.scaling, per_clip=True)
    return {k: m[k] for k in ("average_jaccard", "occlusion_accuracy")}


def _split_scores(pred_m, gt_m, vis, moving, b: dict, j: ScoreSettings) -> dict:
    """With a cutoff C: all points and moving points on the observed (< C) and
    forecast (>= C) frames apart, and on the forecast frames the two
    generous baselines from the true past: held at C-1, and constant speed
    from C-2 and C-1."""
    C, T = j.cutoff, gt_m.shape[1]
    last, before = gt_m[:, C - 1:C], gt_m[:, C - 2:C - 1]
    steps = torch.arange(T, device=gt_m.device, dtype=gt_m.dtype) - (C - 1)
    stop = last.expand_as(gt_m)
    velocity = last + steps[None, :, None, None] * (last - before)
    future = torch.arange(T, device=gt_m.device) >= C
    out = {}
    for name, points, scaling in (("apd", vis, j.scaling), ("apd_moving", moving, "none")):
        for part, frames in (("_obs", ~future), ("_fut", future)):
            out[name + part] = _apd(pred_m, gt_m, points & frames[None, :, None], b, scaling)
        mask = points & future[None, :, None]
        out[name + "_stop_fut"] = _apd(stop, gt_m, mask, b, scaling)
        out[name + "_cv_fut"] = _apd(velocity, gt_m, mask, b, scaling)
    return out


def _hidden_scores(pred_m, gt_m, vis, moving, q, b: dict, j: ScoreSettings) -> dict:
    """Where the point is hidden on the observed frames, all points and moving
    ones, beside the true position where it was last seen (its query position
    before it was first seen): a generous baseline, since a tracker has only
    its own estimate there."""
    T = gt_m.shape[1]
    frame = torch.arange(T, device=gt_m.device)[None, :, None]
    observed = frame < j.cutoff if j.cutoff else torch.ones_like(frame, dtype=torch.bool)
    hidden = ~vis & observed
    seen = torch.where(vis, frame, -1).cummax(1).values                    # last frame seen, -1 none
    last = torch.where(seen >= 0, seen, q[:, None])
    lastseen = gt_m.gather(1, last[..., None].expand_as(gt_m))
    out = {}
    for name, points, scaling in (("apd", torch.ones_like(moving), j.scaling), ("apd_moving", moving, "none")):
        mask = hidden & points[:, None]
        out[name + "_hidden"] = _apd(pred_m, gt_m, mask, b, scaling)
        out[name + "_hidden_lastseen"] = _apd(lastseen, gt_m, mask, b, scaling)
    return out


def _apd(pred_m, gt_m, mask, b: dict, scaling: str, space: str = "3d"):
    """(B,) APD of `pred_m` on the (point, frame) entries `mask` keeps."""
    return tapvid3d_metrics(pred_m, gt_m, mask, b["intrinsics_256"], b["extrinsics"], space=space,
                            scaling=scaling, per_clip=True)["average_pts_within_thresh"]
