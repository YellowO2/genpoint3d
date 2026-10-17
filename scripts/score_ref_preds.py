"""
Score the TAPIP3D predictions that scripts/ref_tapip3d.py saved, with OUR
scorer, on the CPU and without TAPIP3D's environment.

ref_tapip3d.py --out wrote one .pt per clip, named like the clip:
{"seq_id", "coords", "visibs"}. `coords` is (T, N, 3) in metres in the
frame-0 camera, the same frame and units as the clip's `traj_metric`, for the
clip's own query points (each point's true frame-0 position). `visibs` is
(T, N) bool: TAPIP3D's inference already thresholds it, sigmoid of its logit
>= 0.9 (`utils/inference_utils.py`), and that is used as it is.

The conventions are ref_tapip3d.py's: thresholds on intrinsics resized to a
256 px short side; "moving" is a point that gets more than 4 px from its
start; all points are scored with the protocol's median rescale and moving
points with no rescale; each score is a mean over clips, leaving out a clip
with nothing to score. Out comes:

  APD          all points and moving points (earlier: 0.965 and 0.926)
  AJ, OA       average Jaccard and occlusion accuracy from `visibs`, on all
               points and on the moving ones alone
  hidden APD   where the point is truly hidden, split into occluded (inside
               the image) and out of the frame, all and moving points -- the
               same masks validation uses (`track_scores._hidden_scores`);
               each beside two generous carry-forward baselines from the
               TRUE positions where the point was last seen: `_lastseen`
               stays there, `_lastcv` goes on at its last speed

Run (see scripts/score_ref_preds.pbs):
  python scripts/score_ref_preds.py --preds ~/scratch/ref_preds/tapip3d \\
      --clips ~/scratch/ref_eval/val100
"""

import argparse
from pathlib import Path

import torch

from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.score.track_scores import _hidden_scores
from genpoint3d.train.tools.metrics import (HIDDEN_KEYS, THRESHOLDS, clip_mean, motion_px,
                                            resize_intrinsics, tapvid3d_metrics)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="ref_tapip3d.py's --out directory")
    p.add_argument("--clips", required=True, help="the cached clips it was run on")
    p.add_argument("--min-motion-px", type=float, default=4.0)
    args = p.parse_args()

    preds = sorted(Path(args.preds).expanduser().glob("*.pt"))
    clip_dir = Path(args.clips).expanduser()
    if not preds:
        raise SystemExit(f"no predictions in {args.preds}")
    # The hidden scores read only the cutoff and the rescale from these.
    judging = ScoreSettings(device=torch.device("cpu"), scaling="median")

    vals = {}
    for path in preds:
        r = torch.load(path, weights_only=False)
        c = torch.load(clip_dir / path.name, weights_only=False)
        if r["seq_id"] != c["seq_id"]:
            raise SystemExit(f"{path.name}: prediction is for {r['seq_id']}, clip is {c['seq_id']}")
        pred = r["coords"].float()[None]
        # A run on a subset of the queries (ref_tapip3d.py --max-points) keeps
        # the first ones, so it is scored against those.
        n = pred.shape[2]
        gt = c["traj_metric"].float()[None][:, :, :n]                # (1, T, N, 3)
        if pred.shape != gt.shape:
            raise SystemExit(f"{path.name}: prediction {tuple(pred.shape)} vs truth {tuple(gt.shape)}")
        vis = c["visibility"].bool()[None][:, :, :n]
        pvis = r["visibs"].bool()[None]
        K = c["intrinsics"].float()[None]
        b = {"intrinsics": K, "intrinsics_256": resize_intrinsics(K, c["hw"]),
             "extrinsics": c["extrinsics"].float()[None], "hw": torch.tensor([c["hw"]])}
        K256, E = b["intrinsics_256"], b["extrinsics"]
        moving = motion_px(gt, vis, K256, E) > args.min_motion_px      # (1, N)

        add = lambda k, v: vals.setdefault(k, []).append(v)
        add("moving_frac", moving.float().mean(1))
        # All points with the protocol's rescale; moving points, picked out of
        # the clip's points, with none.
        mv = moving[0]
        for name, keep, scaling in (("all", slice(None), "median"), ("moving", mv, "none")):
            sel = lambda x: x[:, :, keep]
            score = lambda pr, pv=None: tapvid3d_metrics(
                sel(pr), sel(gt), sel(vis), K256, E, pred_visible=pv,
                scaling=scaling, per_clip=True)
            m = score(pred, sel(pvis))
            if not mv.any() and name == "moving":
                # No moving point: nothing to score, OA included.
                m = {k: torch.full((1,), float("nan")) for k in m}
            for k, v in m.items():
                add(f"{name}/{k}", v)
            add(f"{name}/static", score(gt[:, :1].expand_as(gt))["average_pts_within_thresh"])
        for k, v in _hidden_scores(pred, gt, vis, moving, b, judging).items():
            add(k, v)

    m = {k: clip_mean(v) for k, v in vals.items()}
    n_moving = int((~torch.cat(vals["moving/average_pts_within_thresh"]).isnan()).sum())
    print(f"{len(preds)} clips ({n_moving} with a moving point) | {m['moving_frac']:.0%} of"
          f" points move > {args.min_motion_px:g}px | visibs: TAPIP3D's own sigmoid >= 0.9\n")
    head = "".join(f"{'<' + str(t) + 'px':>8}" for t in THRESHOLDS)
    for name, label in (("all", "all points, protocol rescale"),
                        ("moving", "moving points, no rescale")):
        print(f"--- {label} ---")
        print(f"{'':10}{head}{'APD':>8}{'AJ':>8}{'OA':>8}")
        print(f"{'static':10}{'':>{8 * len(THRESHOLDS)}}{m[f'{name}/static']:>8.3f}")
        apd = [m[f"{name}/pts_within_{t}"] for t in THRESHOLDS]
        print(f"{'TAPIP3D':10}" + "".join(f"{x:>8.3f}" for x in apd)
              + f"{m[f'{name}/average_pts_within_thresh']:>8.3f}"
              + f"{m[f'{name}/average_jaccard']:>8.3f}{m[f'{name}/occlusion_accuracy']:>8.3f}")
        jac = [m[f"{name}/jaccard_{t}"] for t in THRESHOLDS]
        print(f"{'  Jaccard':10}" + "".join(f"{x:>8.3f}" for x in jac) + "\n")
    print("--- hidden on every frame (all points, protocol rescale / moving, no rescale) ---")
    for k in HIDDEN_KEYS:
        print(f"  {k:<30} {m[k]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
