"""Fit the tracker to cached clips, scoring and saving it as it goes.

Each training clip keeps its images before a random cutoff and forecasts the
rest; validation does the same from `--eval-cutoff`, and reports APD in metres
(TAPVid-3D's `average_pts_within_thresh`) beside the static baseline.

Run:  python -m genpoint3d.train.main --cache CACHE --frames-cache FRAMES --init CKPT
"""

import logging

import torch

from genpoint3d.train.Gpus import Gpus
from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Optimiser import Optimiser
from genpoint3d.train.RunningLoss import RunningLoss
from genpoint3d.train.Saver import Saver
from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.Loaders import Loaders
from genpoint3d.train.data.apply_run_settings import apply_run_settings
from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.learn import learn
from genpoint3d.train.merge_scores import merge_scores
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.model.set_up_model import set_up_model
from genpoint3d.train.score.main import score
from genpoint3d.train.validation_settings import validation_settings


def train(settings: Settings, gpus: Gpus):
    hardware = Hardware.pick(settings)
    saver = Saver(settings)
    clips = split_clips(settings.cache, settings.val_frac, settings.seed, settings.mix)
    clips = apply_run_settings(clips, settings, hardware)
    loaders = Loaders(clips, settings, gpus.rank, gpus.world)
    model = build_model(settings, clips.feat_dim)
    model = set_up_model(model, settings, hardware)
    gpus.same_start(model)
    optimiser = Optimiser(model, settings, hardware)
    first = 1
    if settings.resume:
        # Where an earlier job stopped: its weights, optimiser, schedule and log.
        state = torch.load(settings.resume, map_location=hardware.device, weights_only=False)
        optimiser.restore(model, state["optimiser"])
        saver.resume(state)
        first = state["step"] + 1
        logging.getLogger(__name__).info(f"resumed from {settings.resume} at step {state['step']}")
    judging = validation_settings(settings, hardware)
    running_loss = RunningLoss(settings)
    for step in range(first, settings.steps + 1):
        for batch in loaders.micro_batches(settings.accum):
            losses = learn(model, batch, settings, hardware)
            running_loss.add(losses)
        gpus.average_grads(model)
        norm = optimiser.step(model)
        running_loss.step(step, optimiser.lr(), norm)
        if step % settings.save_every == 0:
            gpus.check_same(model)
        if step % settings.val_every == 0 or step == settings.steps:
            # Every process scores its share; rank 0 logs and saves.
            optimiser.swap_in_average(model)
            interval = running_loss.interval(step)
            scores = score(model, loaders.val, judging, "val", gpus)
            fit_scores = score(model, loaders.fit, judging, "train", gpus) if loaders.fit is not None else None
            if gpus.lead:
                saver.save(merge_scores(interval, scores, fit_scores), model, clips)
            optimiser.swap_out_average(model)
            if gpus.lead:
                saver.keep_resume(step, optimiser.state(model))
            gpus.wait()
        elif step % settings.save_every == 0 and gpus.lead:
            optimiser.swap_in_average(model)
            saver.keep(model, step)
            optimiser.swap_out_average(model)
            saver.keep_resume(step, optimiser.state(model))


if __name__ == "__main__":
    import sys
    import time

    # The package's progress lines, message only, on stdout.
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("genpoint3d").setLevel(logging.INFO)
    settings = Settings.from_command_line()
    gpus = Gpus()
    if not gpus.lead:
        logging.getLogger("genpoint3d").setLevel(logging.WARNING)
    # Each process draws its own clips; the weights start alike (`Gpus.same_start`).
    torch.manual_seed(settings.seed + gpus.rank)
    start = time.time()
    train(settings, gpus)
    gpus.close()
    logging.getLogger("genpoint3d").info(
        f"\ndone in {(time.time() - start) / 60:.1f} min -> {settings.out}")
