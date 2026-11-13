"""Where a training step's time goes: the run's flags (`--points`, `--spawn`,
...), a few steps, each part timed with the GPU synchronised around it:
drawing a clip (spawning included), the grouping (`objects`), the object
pose (`object_places`), and the whole step. Prints any call over a second as
it ends, then each part's seconds a step and its share.

Run (--workers 0, so drawing a clip is counted):
  python scripts/time_parts.py --cache ... --frames-cache ... --init CKPT --points 1000 --spawn 500 --workers 0
"""

import logging
import sys
import time
from collections import defaultdict

import torch

import genpoint3d.train.data.ClipDataset as clip_dataset
import genpoint3d.train.tools.regress as regress
from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Optimiser import Optimiser
from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.Loaders import Loaders
from genpoint3d.train.data.apply_run_settings import apply_run_settings
from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.learn import learn
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.model.set_up_model import set_up_model

spent = defaultdict(float)


def timed(name, fn, gpu=True):
    """`fn`, its time added to `spent[name]`."""
    def run(*a, **k):
        if gpu:
            torch.cuda.synchronize()
        t0 = time.time()
        out = fn(*a, **k)
        if gpu:
            torch.cuda.synchronize()
        took = time.time() - t0
        spent[name] += took
        if took > 1:
            print(f"    {name}: {took:.1f} s", flush=True)
        return out
    return run


def main(steps: int = 3) -> int:
    settings = Settings.from_command_line()
    torch.manual_seed(settings.seed)
    hardware = Hardware.pick(settings)
    clips = apply_run_settings(split_clips(settings.cache, settings.val_frac, settings.seed, settings.mix),
                               settings, hardware)
    clip_dataset._spawned = timed("drawing a clip: spawning", clip_dataset._spawned, gpu=False)
    regress.objects = timed("the grouping (objects)", regress.objects)
    regress.object_places = timed("the object pose (object_places, objects in it)", regress.object_places)
    loaders = Loaders(clips, settings)
    model = set_up_model(build_model(settings, clips.feat_dim), settings, hardware)
    optimiser = Optimiser(model, settings, hardware)
    for step in range(steps + 1):
        if step == 1:
            spent.clear()
            t0 = time.time()
        t_load = time.time()
        batches = list(loaders.micro_batches(settings.accum))
        spent["drawing a clip (all)"] += time.time() - t_load
        print(f"  step {step}: clips drawn in {time.time() - t_load:.1f} s", flush=True)
        for batch in batches:
            t_learn = time.time()
            learn(model, batch, settings, hardware)
            torch.cuda.synchronize()
            print(f"  step {step}: learning took {time.time() - t_learn:.1f} s", flush=True)
        optimiser.step(model)
        torch.cuda.synchronize()
    total = (time.time() - t0) / steps
    print(f"points {settings.points}, spawn {settings.spawn}: {total:.1f} s a step", flush=True)
    for name, s in sorted(spent.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<48} {s / steps:>7.2f} s  ({100 * s / steps / total:.0f}%)", flush=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    raise SystemExit(main())
