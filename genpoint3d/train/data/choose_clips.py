"""Narrow a split to what this run uses, and say what that is."""

import logging
from dataclasses import replace

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.data.Clips import Clips

log = logging.getLogger(__name__)


def choose_clips(clips: Clips, settings, hardware: Hardware) -> Clips:
    """`--overfit`, `--dino 0` and `--fine 1` applied to the split."""
    train, val = clips.train, clips.val
    if settings.overfit:
        # Diagnostic, not a run: the same clips in both halves, so val measures
        # memorisation rather than generalisation. Overfitting a handful of
        # clips is the easiest thing a network does, so failing here means a
        # bug in the pipeline -- no data or capacity excuse is available.
        train = val = (train + val)[: settings.overfit]
    has_feats = clips.has_feats and bool(settings.dino)
    frames = (ClipDataset.frames_dir(settings.frames_cache, train + val, settings.fine_res)
              if settings.fine else None)
    log.info(f"device {hardware.device} | {len(train)} train clips, {len(val)} val clips"
             f"{' | OVERFIT (train == val)' if settings.overfit else ''}"
             + (" | TRACKING (with images)" if has_feats
                else " | no DINOv3" if settings.fine else " | no images (step 2)")
             + (f" | fine CNN at {settings.fine_res}px" if settings.fine else ""))
    return replace(clips, train=train, val=val, has_feats=has_feats, frames=frames)
