"""Score predicted tracks in metres against the truth and baselines."""

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import to_metres
from genpoint3d.train.tools.metrics import MOVING_THRESHOLDS, motion_px, tapvid3d_metrics


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
            mv = vis & (motion_px(gt_m, vis, b["intrinsics_256"], b["extrinsics"]) > 4)[:, None]
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
    out.update(m)
    return out
