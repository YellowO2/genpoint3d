"""Gather per-clip scores across batches, and average them once at the end."""

import logging
import time

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.matching import MATCH_KEYS
from genpoint3d.train.tools.metrics import clip_mean

log = logging.getLogger(__name__)


class Tally:
    """Every score is a mean over CLIPS, and a clip with no point to score (no
    moving point, when the score is restricted to those) is left out of it, as
    scripts/ref_tapip3d.py does. `moving_clips` is how many were left in.

    Per-clip values, averaged once at the end: a mean of batch means would give
    each clip of a short last batch several times its share.

    `split` names the clips in the logged line ("val", "train").
    """

    def __init__(self, split: str) -> None:
        self.split = split
        self.t0 = time.time()
        self.loss_sum = self.loss_n = 0.0
        self.vals = {}

    def count(self, prediction: Prediction) -> None:
        """The batch's loss is a mean over it, so it counts once per clip in it."""
        n = prediction.tracks.shape[0]
        self.loss_sum += prediction.loss * n
        self.loss_n += n

    def add(self, values: dict) -> None:
        for k, v in values.items():
            self.vals.setdefault(k, []).append(v.float().cpu())

    def mean(self, judging: ScoreSettings) -> dict:
        out = {"val_loss": self.loss_sum / max(self.loss_n, 1),
               **{k: clip_mean(v) for k, v in self.vals.items()}}
        moving_key = "average_pts_within_thresh" if judging.min_motion_px else "apd_moving"
        if moving_key in self.vals:
            out["moving_clips"] = int((~torch.cat(self.vals[moving_key]).isnan()).sum())
        log.info(f"    {self.split:<5}" + _line(out) + f"  ({time.time() - self.t0:.0f}s)")
        return out


def _line(m: dict) -> str:
    """The scores worth a glance, each group only if it was scored: loss and
    APD beside the static baseline; in 2D and on moving points; hidden points;
    observed and forecast frames; the matcher; the visibility head."""
    line = (f"  loss {m['val_loss']:.4f}"
            f"  APD {m['average_pts_within_thresh']:.3f} (static {m['apd_static']:.3f})")
    if "apd_2d" in m:
        line += f"  2D {m['apd_2d']:.3f} (static {m['apd_2d_static']:.3f})"
    if "apd_moving" in m:
        line += f"  MOVING {m['apd_moving']:.3f} (static {m['apd_moving_static']:.3f})"
    return line + _hidden(m) + _forecast(m) + _match(m) + _visible(m)


def _hidden(m: dict) -> str:
    """APD where the point is hidden on the observed frames, all/moving
    points, the moving one beside its last-seen and constant-speed baselines;
    then split into occluded and out of the frame, all/moving; then the share
    of hidden entries out of the frame."""
    if "apd_hidden" not in m:
        return ""
    return (f"  hid {m['apd_hidden']:.3f} mv {m['apd_moving_hidden']:.3f}"
            f" (last {m['apd_moving_hidden_lastseen']:.3f}"
            f" cv {m['apd_moving_hidden_lastcv']:.3f})"
            f" (occ {m['apd_occluded']:.3f}/{m['apd_moving_occluded']:.3f}"
            f" out {m['apd_outside']:.3f}/{m['apd_moving_outside']:.3f}"
            f" outFrac {m['hidden_outside_frac']:.2f})")


def _forecast(m: dict) -> str:
    """With a cutoff: APD on the observed frames, then on the forecast ones
    beside the stop, constant-velocity and own-stop baselines, all points and
    then moving points."""
    if "apd_fut" not in m:
        return ""
    return (f"  OBS {m['apd_obs']:.3f} mv {m['apd_moving_obs']:.3f}"
            f"  FUT {m['apd_fut']:.3f} (stop {m['apd_stop_fut']:.3f}"
            f" cv {m['apd_cv_fut']:.3f} own {m['apd_ownstop_fut']:.3f})"
            f" mv {m['apd_moving_fut']:.3f} (stop {m['apd_moving_stop_fut']:.3f}"
            f" cv {m['apd_moving_cv_fut']:.3f} own {m['apd_moving_ownstop_fut']:.3f})")


def _match(m: dict) -> str:
    """The match accuracies: exact/within-one-patch over all visible points,
    then over the moving ones; then how often the truth is within one patch
    of any of the 5 best distinct peaks, all/moving."""
    if "match_acc" not in m:
        return ""
    v = [m[k] for k in MATCH_KEYS[:4]]
    return (f"  match {v[0]:.2f}/{v[1]:.2f} mv {v[2]:.2f}/{v[3]:.2f}"
            f" top5 {m['match_acc_top5_near']:.2f}/{m['match_acc_top5_near_moving']:.2f}")


def _visible(m: dict) -> str:
    """The visibility head's accuracy: over every entry, then where the
    point is hidden, then where it is in view; then the benchmark's average
    Jaccard and occlusion accuracy, on the observed frames."""
    if "vis_acc" not in m:
        return ""
    return (f"  vis {m['vis_acc']:.2f} (hid {m['vis_acc_hidden']:.2f}"
            f" seen {m['vis_acc_visible']:.2f})"
            f"  AJ {m['average_jaccard']:.3f} OA {m['occlusion_accuracy']:.3f}")
