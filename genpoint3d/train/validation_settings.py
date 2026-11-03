"""How a training run scores itself at each validation."""

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.score.ScoreSettings import ScoreSettings


def validation_settings(settings, hardware: Hardware) -> ScoreSettings:
    """The run's own refinement and cutoff, the scoring defaults otherwise."""
    return ScoreSettings(device=hardware.device, amp=hardware.amp, refine_iters=settings.refine_iters,
                         window=settings.window, overlap=settings.overlap, schedule=settings.schedule,
                         cutoff=settings.eval_cutoff)
