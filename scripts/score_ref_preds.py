"""
Score the TAPIP3D predictions that scripts/ref_tapip3d.py saved, with the
very scorer scripts/evaluate.py scores our own model with, on the CPU and
without TAPIP3D's environment.

ref_tapip3d.py --out wrote one .pt per clip, named like the clip:
{"seq_id", "coords", "visibs"}. `coords` is (T, N, 3) in metres in the
frame-0 camera, the same frame and units as the clip's `traj_metric`, for the
clip's own query points. `visibs` is (T, N) bool: TAPIP3D's inference already
thresholds it, sigmoid of its logit >= 0.9 (`utils/inference_utils.py`), and
that is used as it is.

Each clip is served by the same ClipDataset as in evaluate.py (with
--points, its first that many queries, as a checkpoint trained with that many
is scored), and the
prediction goes through the same `track_scores` and `visibility_scores`,
averaged by the same Tally: the protocol's median rescale on all points, no
rescale on the moving ones (more than 4 px), thresholds on intrinsics resized
to a 256 px short side, AJ and OA from `visibs`, hidden-point scores with
their carry-forward baselines. So the numbers are evaluate.py's, key for key.

Run (see scripts/score_ref_preds.pbs and scripts/tapvid3d_ref.pbs):
  python scripts/score_ref_preds.py --preds ~/scratch/ref_preds/tapip3d \\
      --clips ~/scratch/ref_eval/val100
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.score.Tally import Tally
from genpoint3d.train.score.track_scores import track_scores
from genpoint3d.train.score.visibility_scores import visibility_scores
from genpoint3d.train.tools.batch import to_device
from genpoint3d.train.tools.metrics import HIDDEN_KEYS


def to_model_units(pts_m: torch.Tensor, b: dict) -> torch.Tensor:
    """(B, T, N, 3) metres -> the units `to_metres` reads, each clip by its own
    normalisation: the scorer takes tracks as a model hands them over."""
    scene = (pts_m - b["norm_mean"][:, None, None, :]) / b["norm_scale"][:, None, None, None]
    return (scene - b["norm_offset"]) / b["norm_traj_scale"][:, None, None, None]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="ref_tapip3d.py's --out directory")
    p.add_argument("--clips", required=True, help="the cached clips it was run on")
    p.add_argument("--points", type=int, default=0,
                   help="score the first this many queries of a clip, as evaluate.py does"
                        " for a checkpoint trained with that many (0 = all)")
    p.add_argument("--out", default=None, help="write the metrics as JSON here")
    args = p.parse_args()

    clips = sorted(Path(args.clips).expanduser().glob("*.pt"))
    pred_dir = Path(args.preds).expanduser()
    missing = [c.name for c in clips if not (pred_dir / c.name).exists()]
    if not clips or missing:
        raise SystemExit(f"{len(clips)} clips, {len(missing)} without a prediction in {pred_dir}"
                         f" ({' '.join(missing[:5])})")
    loader = DataLoader(ClipDataset(clips, args.points or 2**31, resample=False, target="absolute",
                                    fix_idcard=False, features=False),
                        batch_size=1, shuffle=False, num_workers=4)
    # evaluate.py's scoring rules at their defaults, as tapvid3d_score.pbs ran it.
    device = torch.device("cpu")
    judging = ScoreSettings(device=device, scaling="median")

    tally = Tally("ref")
    for path, batch in zip(clips, loader):
        b = to_device(batch, device)
        r = torch.load(pred_dir / path.name, weights_only=False)
        n = b["traj"].shape[2]
        coords, visibs = r["coords"].float()[None], r["visibs"].bool()[None]
        if coords.shape[2] < n or coords.shape[:2] != b["traj"].shape[:2]:
            raise SystemExit(f"{path.name}: prediction {tuple(coords.shape)} for a clip of"
                             f" {tuple(b['traj'].shape)}")
        coords, visibs = coords[:, :, :n], visibs[:, :, :n]
        prediction = Prediction(batch=b, visual_mask=None, tracks=to_model_units(coords, b),
                                loss=0.0, visibility=visibs.float() * 2 - 1)
        tally.add(track_scores(prediction, judging))
        tally.add(visibility_scores(prediction))
    m = tally.mean(judging)

    print(f"{len(clips)} clips | TAPIP3D | visibs: its own sigmoid >= 0.9\n")
    for k in ("average_pts_within_thresh", "apd_static", "apd_moving", "apd_moving_static",
              "apd_2d", "apd_2d_static", "vis_acc", "vis_acc_hidden", "vis_acc_visible",
              "average_jaccard", "occlusion_accuracy", *HIDDEN_KEYS,
              *(f"pts_within_{t}" for t in (1, 2, 4, 8, 16))):
        if k in m:
            print(f"  {k:<30} {m[k]:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"preds": str(pred_dir), "clips": len(clips), **m},
                                             indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
