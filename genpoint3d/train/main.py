"""Fit the tracker to cached clips, scoring and saving it as it goes.

Each training clip keeps its images before a random cutoff and forecasts the
rest; validation does the same from `--eval-cutoff`, and reports APD in metres
(TAPVid-3D's `average_pts_within_thresh`) beside the static baseline.

Run:  python -m genpoint3d.train.main --cache CACHE --frames-cache FRAMES --init CKPT
"""

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


def train(settings: Settings):
    hardware = Hardware.pick(settings)
    saver = Saver(settings)
    clips = split_clips(settings.cache, settings.val_frac, settings.seed)
    clips = apply_run_settings(clips, settings, hardware)
    loaders = Loaders(clips, settings)
    model = build_model(settings, clips.feat_dim)
    model = set_up_model(model, settings, hardware)
    optimiser = Optimiser(model, settings, hardware)
    judging = validation_settings(settings, hardware)
    running_loss = RunningLoss(settings)
    for step in range(1, settings.steps + 1):
        for batch in loaders.micro_batches(settings.accum):
            losses = learn(model, batch, settings, hardware)
            running_loss.add(losses)
        norm = optimiser.step(model)
        running_loss.step(step, optimiser.lr(), norm)
        if step % settings.val_every == 0 or step == settings.steps:
            optimiser.swap_in_average(model)
            interval = running_loss.interval(step)
            loader, split = loaders.val, "val"
            scores = score(model, loader, judging, split)
            fit_scores = None
            if loaders.fit is not None:
                loader, split = loaders.fit, "train"
                fit_scores = score(model, loader, judging, split)
            report = merge_scores(interval, scores, fit_scores)
            saver.save(report, model, clips)
            optimiser.swap_out_average(model)


if __name__ == "__main__":
    import logging
    import sys
    import time

    import torch

    # The package's progress lines, message only, on stdout.
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("genpoint3d").setLevel(logging.INFO)
    settings = Settings.from_command_line()
    torch.manual_seed(settings.seed)
    start = time.time()
    train(settings)
    logging.getLogger("genpoint3d").info(
        f"\ndone in {(time.time() - start) / 60:.1f} min -> {settings.out}")
