"""Score predicted tracks in metres against the truth and baselines."""

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import to_metres
from genpoint3d.train.tools.metrics import MOVING_THRESHOLDS, in_image, motion_px, tapvid3d_metrics


def track_scores(prediction: Prediction, judging: ScoreSettings) -> dict:
    """Per clip (B,) values, by name.

    Scored in metres, per clip -- a threshold in model units would mean a
    different physical distance in every clip.
    """
    b, j = prediction.batch, judging
    out = {}
    vis = b["visibility"]
    gt_m = to_metres(b["traj"], b)
    if j.min_motion_px:
        # Score only points that actually move. `vis` keeps its meaning for
        # the loss; this narrows what the METRIC counts.
        moving = motion_px(gt_m, vis, b["intrinsics_256"], b["extrinsics"]) > j.min_motion_px
        out["moving_frac"] = moving.float().mean(1)
        vis = vis & moving[:, None]
    score = lambda p: tapvid3d_metrics(p, gt_m, vis, b["intrinsics_256"],
                                       b["extrinsics"], space=j.space,
                                       scaling=j.scaling, per_clip=True)
    pred_m = to_metres(prediction.tracks, b)
    if j.min_motion_px:
        # On the moving points: how much motion does the model predict, and
        # in the right direction? ratio ~0 means it predicts "stays put";
        # cos ~0 means the motion it does predict is unrelated to the truth.
        # Per clip; one with no moving point is 0/0 = NaN and is left out.
        d_gt, d_pr = gt_m - gt_m[:, :1], pred_m - gt_m[:, :1]
        on_vis = lambda x: (x * vis).flatten(1).sum(1)
        out["disp_ratio"] = on_vis(d_pr.norm(dim=-1)) / on_vis(d_gt.norm(dim=-1))
        out["disp_cos"] = (on_vis(torch.nn.functional.cosine_similarity(d_pr, d_gt, dim=-1))
                           / vis.flatten(1).sum(1))
    if j.oracle_axis:
        # Attribution, not an input: the model has already produced its
        # answer, and this only overwrites one channel of that answer before
        # scoring. Nothing is fed back, so it cannot tell the model where to
        # look.
        #
        # Run all three axes. Correcting ANY of the three removes a third of
        # the error budget, so a jump on z alone proves nothing -- z only
        # matters if it buys MORE than x and y do.
        a = "xyz".index(j.oracle_axis)
        pred_m = pred_m.clone()
        pred_m[..., a] = gt_m[..., a]
    m = score(pred_m)
    # Static Baseline: each point stays where it started. Free -- no
    # sampling -- and the number a reader will ask for first.
    m["apd_static"] = score(
        gt_m[:, :1].expand_as(gt_m)
    )["average_pts_within_thresh"]
    if j.space == "3d":
        # The same prediction, scored in the image plane. Costs no extra
        # sampling, and the pair separates "cannot find the point" from
        # "finds it and misplaces it in depth" on every validation.
        s2 = lambda p: tapvid3d_metrics(p, gt_m, vis, b["intrinsics_256"],
                                        b["extrinsics"], space="2d",
                                        scaling=j.scaling, per_clip=True)
        m["apd_2d"] = s2(pred_m)["average_pts_within_thresh"]
        m["apd_2d_static"] = s2(gt_m[:, :1].expand_as(gt_m))["average_pts_within_thresh"]
        if not j.min_motion_px:
            # The honest pair: only points that move more than 4px, and no
            # global rescale. About half our query points never move, and
            # "nothing moves" tracks those perfectly, so the plain average
            # can sit well above its baseline without any tracking in it.
            moving = motion_px(gt_m, vis, b["intrinsics_256"], b["extrinsics"]) > 4
            mv = vis & moving[:, None]
            s3 = lambda p: tapvid3d_metrics(p, gt_m, mv, b["intrinsics_256"],
                                            b["extrinsics"], scaling="none",
                                            per_clip=True)
            mv_m = s3(pred_m)
            m["apd_moving"] = mv_m["average_pts_within_thresh"]
            # The average hides WHERE the points land: finding most of them
            # within 16px and none within 1px scores the same as a model
            # with some precision.
            for t in MOVING_THRESHOLDS:
                m[f"moving_within_{t}"] = mv_m[f"pts_within_{t}"]
            m["apd_moving_static"] = s3(gt_m[:, :1].expand_as(gt_m))["average_pts_within_thresh"]
            m.update(_hidden_scores(pred_m, gt_m, vis, moving, b, j))
            if j.cutoff:
                others = [to_metres(o, b) for o in prediction.others]
                m.update(_split_scores(pred_m, others, gt_m, vis, mv, b, j))
    out.update(m)
    return out


def _split_scores(pred_m, others, gt_m, vis, moving, b: dict, j: ScoreSettings) -> dict:
    """With a cutoff C: the same two scores as above, all points and moving
    points, on the observed frames (< C, `_obs`) and the forecast ones
    (>= C, `_fut`) apart, and two baselines on the forecast frames.

    The baselines are generous: they start from the TRUE positions at frames
    C-2 and C-1, where the model has only its own estimate of them. A model
    that beats them is forecasting; one that only matches them may still be
    tracking its observed frames well and extrapolating no better.

        stop      every forecast frame is the true position at frame C-1
        cv        constant velocity from the true positions at C-2 and C-1

    And one that is not generous, as it starts from the model's own estimate:

        ownstop   every forecast frame is the MODEL's position at frame C-1.
                  A forecast that does no better than this adds nothing to
                  freezing the model's own track.

    With `others` (more flow samples, k in all), `_fut_best{k}` is the oracle
    best of k, the usual score of a generative forecaster: each point keeps
    the sample nearest the truth over its visible forecast frames.
    """
    C, T = j.cutoff, gt_m.shape[1]
    last, before = gt_m[:, C - 1:C], gt_m[:, C - 2:C - 1]
    steps = (torch.arange(T, device=gt_m.device, dtype=gt_m.dtype) - (C - 1))
    stop = last.expand_as(gt_m)
    velocity = last + steps[None, :, None, None] * (last - before)
    future = torch.arange(T, device=gt_m.device) >= C
    ownstop = pred_m[:, C - 1:C].expand_as(pred_m)
    best = _best_of(pred_m, others, gt_m, vis & future[None, :, None]) if others else None
    out = {}
    for name, points, scaling in (("apd", vis, j.scaling), ("apd_moving", moving, "none")):
        for part, frames in (("_obs", ~future), ("_fut", future)):
            mask = points & frames[None, :, None]
            score = lambda p: _apd(p, gt_m, mask, b, scaling)
            out[name + part] = score(pred_m)
            out[name + "_static" + part] = score(gt_m[:, :1].expand_as(gt_m))
            if part == "_fut":
                out[name + "_stop_fut"] = score(stop)
                out[name + "_cv_fut"] = score(velocity)
                out[name + "_ownstop_fut"] = score(ownstop)
                if best is not None:
                    out[name + f"_fut_best{len(others) + 1}"] = score(best)
    return out


def _hidden_scores(pred_m, gt_m, vis, moving, b: dict, j: ScoreSettings) -> dict:
    """The point still exists when nobody can see it. Scored where it is
    hidden on the observed frames, `_hidden`, and that split in two:
    `_occluded`, where its true position is inside the image, and `_outside`,
    where it has left the frame. With a cutoff C, also on every forecast
    frame, hidden or not, `_fut_all`. All points with the protocol's rescale,
    and the moving ones without, as above. `moving` (B, N) is the points that
    move, judged on visible frames.

    `hidden_outside_frac` is the share of hidden entries that are outside,
    per clip; NaN for a clip with none hidden.
    """
    T = gt_m.shape[1]
    frame = torch.arange(T, device=gt_m.device)[None, :, None]
    observed = frame < j.cutoff if j.cutoff else torch.ones_like(frame, dtype=torch.bool)
    inside = in_image(gt_m, b["intrinsics"], b["extrinsics"], b["hw"])
    hidden = ~vis & observed
    out = {"hidden_outside_frac": ((hidden & ~inside).flatten(1).sum(1)
                                   / hidden.flatten(1).sum(1))}
    for name, points, scaling in (("apd", torch.ones_like(moving), j.scaling),
                                  ("apd_moving", moving, "none")):
        mine = hidden & points[:, None]
        score = lambda mask: _apd(pred_m, gt_m, mask, b, scaling)
        out[name + "_hidden"] = score(mine)
        out[name + "_occluded"] = score(mine & inside)
        out[name + "_outside"] = score(mine & ~inside)
        if j.cutoff:
            out[name + "_fut_all"] = score(points[:, None] & ~observed)
    return out


def _apd(pred_m, gt_m, mask, b: dict, scaling: str):
    """(B,) APD of `pred_m` on the (point, frame) entries `mask` keeps."""
    return tapvid3d_metrics(pred_m, gt_m, mask, b["intrinsics_256"], b["extrinsics"],
                            scaling=scaling, per_clip=True)["average_pts_within_thresh"]


def _best_of(pred_m, others, gt_m, mask):
    """Per point, the sample with the least mean error (metres) on `mask`'s
    frames. A point with no such frame keeps the first sample."""
    cands = torch.stack([pred_m, *others])                       # (K, B, T, N, 3)
    err = (cands - gt_m).norm(dim=-1)                            # (K, B, T, N)
    err = (err * mask).sum(2) / mask.sum(1).clamp(min=1)         # (K, B, N)
    pick = err.argmin(0)                                         # (B, N)
    idx = pick[None, :, None, :, None].expand(1, *pred_m.shape)
    return cands.gather(0, idx)[0]
