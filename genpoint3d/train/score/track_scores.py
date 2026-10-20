"""Score predicted tracks in metres against the truth and baselines."""

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import rigid_buddies, to_metres
from genpoint3d.train.tools.metrics import MOVING_THRESHOLDS, in_image, motion_px, tapvid3d_metrics

# A point counts as still if its true position never moves more than this
# (metres): saved positions are not bit-exact, but a still object in Kubric
# does not move at all in the frame-0 camera.
STILL_M = 1e-3


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
    if prediction.visibility is not None:
        m.update(_jaccard_scores(pred_m, gt_m, prediction, judging))
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


def _jaccard_scores(pred_m, gt_m, prediction: Prediction, j: ScoreSettings) -> dict:
    """The benchmark's two scores that need the visibility head's answer:
    `average_jaccard` (with `jaccard_{t}` at each threshold) and
    `occlusion_accuracy`. A logit above 0 (sigmoid above 0.5) says visible.

    Only on the frames that have their image: with a cutoff, a forecast frame
    has nothing to see, so AJ and OA are for the observed frames alone.
    Over every query point and the true visibility, even when the other
    scores keep only the moving points: a point left out of the truth but
    called visible would count against the head for no reason. The rescale is
    the protocol's, fitted on those observed frames.
    """
    b = prediction.batch
    T = j.cutoff or gt_m.shape[1]
    m = tapvid3d_metrics(pred_m[:, :T], gt_m[:, :T], b["visibility"][:, :T],
                         b["intrinsics_256"][:, :T], b["extrinsics"][:, :T],
                         pred_visible=prediction.visibility[:, :T] > 0,
                         space=j.space, scaling=j.scaling, per_clip=True)
    return {k: v for k, v in m.items() if k.startswith("jaccard_")
            or k in ("average_jaccard", "occlusion_accuracy")}


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
    the moving ones without, as above, and the still ones (`apd_still_*`) without
    too. `moving` (B, N) is the points that move, judged on visible frames;
    still points are the ones whose true position never changes at all.

    `hidden_outside_frac` is the share of hidden entries that are outside,
    per clip; NaN for a clip with none hidden.

    Beside them, the moving points where they ARE seen, by third of the clip
    (`apd_moving_seen_early`, `_mid`, `_late`, frame 0 left out): whether the
    model gets less precise the further a frame is from frame 0, whose look
    is the template. `_static` after each is "nothing moves" on the same
    entries, how far the points have gone by then.

    Two baselines on the same entries, each with the same suffix after the
    hidden, occluded and outside names:

        _lastseen  the true position at the last frame the point was seen
        _lastcv    constant speed from the true positions at the last two
                   frames it was seen; the last seen one if it was seen once
        _interp    a straight line from the true position where it was last
                   seen to the one where it is next seen; the last seen one
                   if it is never seen again
        _buddy     carried from where it was last seen by the rotation and
                   shift that best move its object's points visible in this
                   frame (`rigid_buddies`, among the scored points) from
                   there to here; the shift alone with fewer than three, the
                   last seen position with none. What knowing which visible
                   points share its rigid object could give.
        _buddypred the same from the PREDICTED positions, of the point where
                   it was last seen and of its buddies: what that rule would
                   make of our own tracks, with the true buddies and the true
                   visibility. Not a baseline but a diagnosis.

And each hidden set split by whether the point is seen again later in the
clip: `_returns`, a gap that a tracker reading the whole clip can fill from
both ends, and `_gone`, which only a guess forward can.

    They are generous: they use the TRUE positions at those frames, where a
    tracker has only its own estimate of them. They show how much of a
    hidden score carrying the last estimate forward could explain.
    """
    T = gt_m.shape[1]
    frame = torch.arange(T, device=gt_m.device)[None, :, None]
    observed = frame < j.cutoff if j.cutoff else torch.ones_like(frame, dtype=torch.bool)
    inside = in_image(gt_m, b["intrinsics"], b["extrinsics"], b["hw"])
    hidden = ~vis & observed
    out = {"hidden_outside_frac": ((hidden & ~inside).flatten(1).sum(1)
                                   / hidden.flatten(1).sum(1))}
    seen = vis & moving[:, None]
    for part, frames in zip(("early", "mid", "late"), torch.arange(1, T).tensor_split(3)):
        mask = seen & torch.isin(frame, frames.to(frame.device))
        out[f"apd_moving_seen_{part}"] = _apd(pred_m, gt_m, mask, b, "none")
        out[f"apd_moving_seen_{part}_static"] = _apd(gt_m[:, :1].expand_as(gt_m), gt_m, mask, b, "none")
    lastseen, lastcv = _last_seen(gt_m, vis)
    interp, returns = _between_sightings(gt_m, vis)
    pairs = rigid_buddies(gt_m)
    buddy = _with_buddies(gt_m, vis, pairs)
    buddy_pred = _with_buddies(pred_m, vis, pairs)
    # `apd_still` is the easy case: points whose true position never changes,
    # hidden frames included, so holding the last seen position is exactly
    # right (`_lastseen` scores 1). Anything below that is the model losing
    # points it cannot see even when nothing happens to them.
    still = (gt_m - gt_m[:, :1]).norm(dim=-1).amax(1) < STILL_M     # (B, N)
    for name, points, scaling in (("apd", torch.ones_like(moving), j.scaling),
                                  ("apd_moving", moving, "none"),
                                  ("apd_still", still, "none")):
        mine = hidden & points[:, None]
        for guess, suffix in ((pred_m, ""), (lastseen, "_lastseen"), (lastcv, "_lastcv"),
                              (interp, "_interp"), (buddy, "_buddy"),
                              (buddy_pred, "_buddypred")):
            score = lambda mask: _apd(guess, gt_m, mask, b, scaling)
            out[name + "_hidden" + suffix] = score(mine)
            out[name + "_occluded" + suffix] = score(mine & inside)
            out[name + "_outside" + suffix] = score(mine & ~inside)
            out[name + "_returns" + suffix] = score(mine & returns)
            out[name + "_gone" + suffix] = score(mine & ~returns)
        if j.cutoff:
            out[name + "_fut_all"] = _apd(pred_m, gt_m, points[:, None] & ~observed, b, scaling)
    return out


def _last_seen(gt_m, vis):
    """Two guesses (B, T, N, 3) for every frame, from the true positions at
    the frames where the point was last seen, up to and including this one:
    the last seen position, and constant speed from the last two. A point
    never seen up to a frame falls back to frame 0, and constant speed falls
    back to the last seen position when there is no earlier sighting."""
    T = gt_m.shape[1]
    frame = torch.arange(T, device=gt_m.device)[None, :, None].expand_as(vis)
    # The last frame seen up to each frame; -1 if none.
    seen = torch.where(vis, frame, -1).cummax(1).values
    last = seen.clamp(min=0)
    # The last frame seen strictly before each frame; -1 if none.
    seen_before = torch.cat([torch.full_like(seen[:, :1], -1), seen[:, :-1]], 1)
    # The sighting before the last one; none if the last is only the fallback.
    before = torch.where(seen >= 0, seen_before.gather(1, last), -1)
    pick = lambda idx: gt_m.gather(1, idx.clamp(min=0)[..., None].expand_as(gt_m))
    at_last, at_before = pick(last), pick(before)
    speed = (at_last - at_before) / (last - before).clamp(min=1)[..., None]
    lastcv = torch.where((before >= 0)[..., None],
                         at_last + speed * (frame - last)[..., None], at_last)
    return at_last, lastcv


def _with_buddies(traj, vis, pairs):
    """A guess (B, T, N, 3) for every frame: each point carried from its last
    sighting by the rigid motion of its buddies `pairs` (B, N, N) visible now,
    all positions read from `traj`. See `_buddy` in `_hidden_scores`."""
    B, T, N, _ = traj.shape
    frame = torch.arange(T, device=traj.device)[None, :, None].expand_as(vis)
    last = torch.where(vis, frame, -1).cummax(1).values.clamp(min=0)   # (B, T, N)
    lastseen = traj.gather(1, last[..., None].expand_as(traj))
    w = (pairs[:, None] & vis[:, :, None, :]).float()                  # (B, T, i, j)
    n = w.sum(-1)                                                       # (B, T, i)
    src = traj[torch.arange(B)[:, None, None], last]                   # (B, T, i, j, 3)
    dst = traj[:, :, None]                                              # (B, T, 1, j, 3)
    mean = lambda x: (w[..., None] * x).sum(-2) / n.clamp(min=1)[..., None]
    cs, cd = mean(src), mean(dst)                                       # (B, T, i, 3)
    H = torch.einsum("btij,btijk,btijl->btikl", w, src - cs[..., None, :],
                     (dst - cd[..., None, :]).expand_as(src))
    U, _, Vh = torch.linalg.svd(H)
    flip = torch.ones_like(cs)
    flip[..., 2] = torch.linalg.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2)).sign()
    R = Vh.transpose(-1, -2) @ torch.diag_embed(flip) @ U.transpose(-1, -2)
    R = torch.where((n >= 3)[..., None, None], R, torch.eye(3, device=traj.device))
    guess = (R @ (lastseen - cs)[..., None])[..., 0] + cd
    return torch.where((n >= 1)[..., None], guess, lastseen)


def _between_sightings(gt_m, vis):
    """A guess (B, T, N, 3) for every frame on a straight line between the
    true positions at the last sighting up to it and the next one from it,
    and (B, T, N) whether there is a next one. With none, the last sighting;
    with no earlier one, frame 0, as `_last_seen` does."""
    T = gt_m.shape[1]
    frame = torch.arange(T, device=gt_m.device)[None, :, None].expand_as(vis)
    last = torch.where(vis, frame, -1).cummax(1).values.clamp(min=0)
    nxt = torch.where(vis, frame, T).flip(1).cummin(1).values.flip(1)
    returns = nxt < T
    pick = lambda idx: gt_m.gather(1, idx.clamp(max=T - 1)[..., None].expand_as(gt_m))
    at_last, at_next = pick(last), pick(nxt)
    w = ((frame - last) / (nxt - last).clamp(min=1))[..., None]
    return torch.where(returns[..., None], at_last + w * (at_next - at_last), at_last), returns


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
