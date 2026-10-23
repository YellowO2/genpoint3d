"""How a training run scores itself at each validation."""

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.score.ScoreSettings import ScoreSettings


def validation_settings(settings, hardware: Hardware) -> ScoreSettings:
    """The run's own method, loss and cutoff, the scoring defaults otherwise."""
    return ScoreSettings(device=hardware.device, amp=hardware.amp, method=settings.method,
                         refine_iters=settings.refine_iters, param=settings.param,
                         carry=bool(settings.carry), window=settings.window,
                         overlap=settings.overlap,
                         anchor_frame0=settings.anchor_frame0, loss_type=settings.loss_type,
                         cutoff=settings.eval_cutoff)
