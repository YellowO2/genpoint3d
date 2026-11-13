"""Time and memory of training steps: `--steps` steps after `--warm` warm-up
ones, with the run's flags (`--points`, `--patch`, ...), no validation. Prints
the mean time a step and the peak GPU memory, or that it ran out of memory.

Run:  python scripts/profile_step.py --cache ... --frames-cache ... --init CKPT --points 1000 --patch 64
"""

import logging
import sys
import time

import torch

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Optimiser import Optimiser
from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.Loaders import Loaders
from genpoint3d.train.data.apply_run_settings import apply_run_settings
from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.learn import learn
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.model.set_up_model import set_up_model


def main(warm: int = 5, steps: int = 15) -> int:
    settings = Settings.from_command_line()
    torch.manual_seed(settings.seed)
    hardware = Hardware.pick(settings)
    clips = apply_run_settings(split_clips(settings.cache, settings.val_frac, settings.seed, settings.mix),
                               settings, hardware)
    loaders = Loaders(clips, settings)
    model = set_up_model(build_model(settings, clips.feat_dim), settings, hardware)
    optimiser = Optimiser(model, settings, hardware)
    label = f"points {settings.points}, patch {settings.patch}"
    try:
        for step in range(warm + steps):
            if step == warm:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
            for batch in loaders.micro_batches(settings.accum):
                learn(model, batch, settings, hardware)
            optimiser.step(model)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(f"{label}: out of memory", flush=True)
        return 0
    print(f"{label}: {(time.time() - t0) / steps:.2f} s a step, peak memory "
          f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    raise SystemExit(main())
