"""How a training run scores itself at each validation."""

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.score.ScoreSettings import ScoreSettings


def validation_settings(settings, hardware: Hardware) -> ScoreSettings:
    """The run's own method and loss, the scoring defaults otherwise."""
    return ScoreSettings(device=hardware.device, amp=hardware.amp, method=settings.method,
                         refine_iters=settings.refine_iters,
                         anchor_frame0=settings.anchor_frame0, loss_type=settings.loss_type)
