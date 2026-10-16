"""Keep the training losses across steps, and log them every 100 steps."""

import logging
import time

from genpoint3d.train.Interval import Interval
from genpoint3d.train.Losses import Losses

log = logging.getLogger(__name__)


class RunningLoss:
    """`running` is the 100-step window; `since_val` spans a whole validation
    interval so the log holds train and val loss on the same steps.
    """

    def __init__(self, settings) -> None:
        self.match_learn = settings.match_learn
        self.t0 = time.time()
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

    def interval(self, step: int) -> Interval:
        """The mean losses since the last validation, and a fresh start."""
        n = max(self.since_val_n, 1)
        interval = Interval(step=step, train_loss=self.since_val / n,
                            match_loss=self.match_sum / n if self.match_learn else None)
        self.since_val, self.since_val_n = 0.0, 0
        self.match_sum = 0.0
        log.info(f"  VAL step {step}  train_loss {interval.train_loss:.4f}"
                 + (f"  match_loss {interval.match_loss:.4f}"
                    if interval.match_loss is not None else ""))
        return interval
