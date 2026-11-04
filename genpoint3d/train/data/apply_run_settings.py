"""Apply this run's settings to the train/val split, and log the run line."""

import logging
from dataclasses import replace
from pathlib import Path

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.data.Clips import Clips

log = logging.getLogger(__name__)


def apply_run_settings(clips: Clips, settings, hardware: Hardware) -> Clips:
    """`--overfit` (train and val the same first N clips, drawn alike), and
    each cache's frames checked."""
    if len(settings.frames_cache) != len(settings.cache):
        raise SystemExit("--frames-cache gives one directory per --cache")
    train, val, weights = clips.train, clips.val, clips.weights
    if settings.overfit:
        train = val = (train + val)[: settings.overfit]
        weights = None
    frames = {Path(c): ClipDataset.frames_dir(f, [p for p in train + val if p.parent == Path(c)], settings.fine_res)
              for c, f in zip(settings.cache, settings.frames_cache)}
    log.info(f"device {hardware.device} | {len(train)} train clips from {len(settings.cache)} caches, "
             f"{len(val)} val clips{' | OVERFIT (train == val)' if settings.overfit else ''}"
             f" | frames at {settings.fine_res}px")
    return replace(clips, train=train, val=val, weights=weights, frames=frames)
