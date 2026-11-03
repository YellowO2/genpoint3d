"""Apply this run's settings to the train/val split, and log the run line."""

import logging
from dataclasses import replace

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.data.Clips import Clips

log = logging.getLogger(__name__)


def apply_run_settings(clips: Clips, settings, hardware: Hardware) -> Clips:
    """`--overfit` (train and val the same first N clips), and the frames cache checked."""
    train, val = clips.train, clips.val
    if settings.overfit:
        train = val = (train + val)[: settings.overfit]
    frames = ClipDataset.frames_dir(settings.frames_cache, train + val, settings.fine_res)
    log.info(f"device {hardware.device} | {len(train)} train clips, {len(val)} val clips"
             f"{' | OVERFIT (train == val)' if settings.overfit else ''} | frames at {settings.fine_res}px")
    return replace(clips, train=train, val=val, frames=frames)
