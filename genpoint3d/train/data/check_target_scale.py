"""Stop a run whose target paths are far from unit size for the diffusion method. Checked once before training.
"""

import logging

log = logging.getLogger(__name__)


def check_target_scale(loaders, settings) -> None:
    """The denoising target must sit near unit variance: flow matching mixes it
    with x0 ~ N(0, I), and a target far below 1 teaches the model to output
    -x0 while the loss still falls. That failure is silent, so check it once
    against a real batch rather than trusting the constant.
    """
    probe = next(iter(loaders.train))
    tstd = probe["traj"][probe["visibility"]].std().item()
    log.info(f"target std {tstd:.3f} ({settings.target}, TRAJ_SCALE"
             f" {loaders.train.dataset.traj_scale})")
    if not 0.3 < tstd < 3.0:
        raise SystemExit(
            f"target std {tstd:.3f} is outside [0.3, 3.0] -- recalibrate with\n"
            f"  python scripts/calibrate_traj_scale.py --cache {settings.cache}"
            f" --target {settings.target}")
