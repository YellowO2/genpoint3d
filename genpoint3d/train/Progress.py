"""Keep the running losses and print how training is going."""

import logging
import time

from genpoint3d.train.Losses import Losses
from genpoint3d.train.Report import Report
from genpoint3d.train.tools.matching import MATCH_KEYS
from genpoint3d.train.tools.metrics import HIDDEN_KEYS

log = logging.getLogger(__name__)


class Progress:
    """`running` is the 100-step print window; `since_val` spans a whole
    validation interval so the log holds train and val loss on the same steps.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self.t0 = self.tv = time.time()
        self.running = 0.0
        self.since_val, self.since_val_n = 0.0, 0
        self.micro_loss = 0.0
        self.match_sum = 0.0

    def add(self, losses: Losses) -> None:
        """One micro-batch's losses."""
        self.micro_loss += losses.method
        self.match_sum += losses.match

    def step(self, step: int, lr: float) -> None:
        """The optimiser has stepped; every 100 steps, say how it went."""
        self.running += self.micro_loss
        self.since_val += self.micro_loss
        self.since_val_n += 1
        self.micro_loss = 0.0
        if step % 100 == 0:
            log.info(f"  step {step:>6}  loss {self.running / 100:.5f}"
                     f"  lr {lr:.2e}"
                     f"  {(time.time() - self.t0) / step:.3f}s/it")
            self.running = 0.0
        self.tv = time.time()

    def report(self, step: int, scores: dict, fit_scores: dict | None) -> Report:
        """The validation's line, with the train-split scores folded in."""
        m = scores
        if fit_scores is not None:
            # Every train-split score, under the same name as its val one.
            m.update({f"train_{k}": v for k, v in fit_scores.items()})
        if self.settings.match_learn:
            m["train_match_loss"] = self.match_sum / max(self.since_val_n, 1)
        self.match_sum = 0.0
        train_loss = self.since_val / max(self.since_val_n, 1)
        log.info(f"  VAL step {step}  train_loss {train_loss:.4f}"
                 f"  val_loss {m['val_loss']:.4f}"
                 f"  APD {m['average_pts_within_thresh']:.3f}"
                 f"  (static {m['apd_static']:.3f})"
                 f"  2D {m['apd_2d']:.3f} (static {m['apd_2d_static']:.3f})"
                 f"  MOVING {m['apd_moving']:.3f} (static {m['apd_moving_static']:.3f})"
                 + (f"  trainAPD {m['train_average_pts_within_thresh']:.3f}"
                    f" (static {m['train_apd_static']:.3f})"
                    f" 2D {m['train_apd_2d']:.3f}"
                    f" MOVING {m['train_apd_moving']:.3f}"
                    f" (static {m['train_apd_moving_static']:.3f})"
                    if fit_scores is not None else "")
                 + _hidden_line(m) + _hidden_line(m, "train_")
                 + _forecast_line(m)
                 + _match_line(m) + _match_line(m, "train_")
                 + f"  ({time.time() - self.tv:.0f}s)")
        self.since_val, self.since_val_n = 0.0, 0
        return Report(step=step, train_loss=train_loss, scores=m)

    def finish(self) -> None:
        log.info(f"\ndone in {(time.time() - self.t0) / 60:.1f} min -> {self.settings.out}")


def _hidden_line(m: dict, prefix: str = "") -> str:
    """APD where the point is hidden on the observed frames, all/moving
    points; then split into occluded and out of the frame, all/moving; then
    the share of hidden entries out of the frame. Empty if it was not scored."""
    if prefix + "apd_hidden" not in m:
        return ""
    v = [m[prefix + k] for k in HIDDEN_KEYS]
    return (f"  {'trainHid' if prefix else 'hid'} {v[0]:.3f} mv {v[1]:.3f}"
            f" (occ {v[2]:.3f}/{v[3]:.3f} out {v[4]:.3f}/{v[5]:.3f}"
            f" outFrac {v[6]:.2f})")


def _forecast_line(m: dict) -> str:
    """With --eval-cutoff: APD on the observed frames, then on the forecast
    ones beside the stop, constant-velocity and own-stop baselines, all points and then
    moving points. Empty without a cutoff."""
    if "apd_fut" not in m:
        return ""
    return (f"  OBS {m['apd_obs']:.3f} mv {m['apd_moving_obs']:.3f}"
            f"  FUT {m['apd_fut']:.3f} (stop {m['apd_stop_fut']:.3f}"
            f" cv {m['apd_cv_fut']:.3f} own {m['apd_ownstop_fut']:.3f})"
            f" mv {m['apd_moving_fut']:.3f} (stop {m['apd_moving_stop_fut']:.3f}"
            f" cv {m['apd_moving_cv_fut']:.3f} own {m['apd_moving_ownstop_fut']:.3f})")


def _match_line(m: dict, prefix: str = "") -> str:
    """The match accuracies for the VAL line: exact/within-one-patch over all
    visible points, then over the moving ones; then how often the truth is
    within one patch of any of the 5 best distinct peaks, all/moving. Empty if
    there is no match."""
    if prefix + "match_acc" not in m:
        return ""
    v = [m[prefix + k] for k in MATCH_KEYS[:4]]
    return (f"  {'trainMatch' if prefix else 'match'} {v[0]:.2f}/{v[1]:.2f}"
            f" mv {v[2]:.2f}/{v[3]:.2f}"
            f" top5 {m[prefix + 'match_acc_top5_near']:.2f}"
            f"/{m[prefix + 'match_acc_top5_near_moving']:.2f}")
