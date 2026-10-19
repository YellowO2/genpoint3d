"""How many point pairs `rigid_pairs` finds per clip, before a rigidity run.

    python scripts/count_rigid_pairs.py --cache <cache> --points 128 --batch 8

Zero pairs would make --rigid-loss silently do nothing.
"""

import sys

import torch

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.Loaders import Loaders
from genpoint3d.train.data.apply_run_settings import apply_run_settings
from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.tools.batch import rigid_pairs, to_device

settings = Settings.from_command_line(sys.argv[1:] + ["--workers", "0"])
hardware = Hardware.pick(settings)
clips = apply_run_settings(split_clips(settings.cache, settings.val_frac, settings.seed), settings, hardware)
b = to_device(next(iter(Loaders(clips, settings).train)), hardware.device)
pairs = rigid_pairs(b)
print("pairs per clip:", (pairs.sum((1, 2)) // 2).tolist())
print("points with a rigid buddy per clip:", pairs.any(2).sum(1).tolist(), "of", pairs.shape[1])
