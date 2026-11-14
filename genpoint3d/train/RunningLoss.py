"""Keep the training losses across steps, and log them every 100 steps."""

import logging
import math
import time

from genpoint3d.train.Interval import Interval
from genpoint3d.train.Losses import Losses

log = logging.getLogger(__name__)


class RunningLoss:
    """`running` is the 100-step window; `since_val` spans a whole validation
    interval so the log holds train and val loss on the same steps.
    """

    def __init__(self, settings) -> None:
        self.vis_learn = settings.vis_loss > 0
        self.t0 = time.time()
        self.running = 0.0
        self.since_val, self.since_val_n = 0.0, 0
        self.micro_loss = 0.0
        self.vis_sum = 0.0
        self.centre_learn = settings.centre_loss > 0
        self.centre_sum = self.micro_centre = 0.0
        self.clip = settings.clip
        self.norm_sum, self.clipped = 0.0, 0

    def add(self, losses: Losses) -> None:
        """One micro-batch's losses."""
        self.micro_loss += losses.method
        self.vis_sum += losses.visibility
        self.micro_centre += losses.centre

    def step(self, step: int, lr: float, norm: float) -> None:
        """The optimiser has stepped; every 100 steps, say how it went, and at
        steps 1 and 10 how long a step takes, so a hang or a slow step shows in
        minutes. `norm` is the gradient's norm before the clip. A loss or
        gradient that is not finite stops the run: the weights it reaches are
        lost."""
        if not math.isfinite(self.micro_loss) or not math.isfinite(norm):
            raise RuntimeError(f"step {step}: the loss ({self.micro_loss}) or the gradient's norm ({norm}) "
                               "is not finite; stopping before the weights are")
        if step in (1, 10):
            log.info(f"  step {step:>6}  {(time.time() - self.t0) / step:.1f} s a step so far")
        self.norm_sum += norm
        self.clipped += bool(self.clip) and norm > self.clip
        self.running += self.micro_loss
        self.since_val += self.micro_loss
        self.since_val_n += 1
        self.centre_sum += self.micro_centre
        self.micro_loss = self.micro_centre = 0.0
        if step % 100 == 0:
            centre = f" (centre {self.centre_sum / 100:.5f})" if self.centre_learn else ""
            log.info(f"  step {step:>6}  loss {self.running / 100:.5f}{centre}"
                     f"  lr {lr:.2e}"
                     f"  grad {self.norm_sum / 100:.2f} (clipped {self.clipped}%)"
                     f"  {(time.time() - self.t0) / step:.3f}s/it")
            self.running = self.centre_sum = 0.0
            self.norm_sum, self.clipped = 0.0, 0

    def interval(self, step: int) -> Interval:
        """The mean losses since the last validation, and a fresh start."""
        n = max(self.since_val_n, 1)
        interval = Interval(step=step, train_loss=self.since_val / n,
                            vis_loss=self.vis_sum / n if self.vis_learn else None)
        self.since_val, self.since_val_n = 0.0, 0
        self.vis_sum = 0.0
        log.info(f"  VAL step {step}  train_loss {interval.train_loss:.4f}"
                 + (f"  vis_loss {interval.vis_loss:.4f}" if interval.vis_loss is not None else ""))
        return interval
