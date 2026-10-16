"""Fit the tracker to cached clips, scoring and saving it as it goes.

Many clips, held-out validation, checkpoints. The question is whether the model
learns anything that transfers to clips it has never seen.

The metric that matters is APD on the VAL split -- `average_pts_within_thresh`
from the TAP-Vid-3D benchmark, scored in metres. `apd_static` is the
benchmark's own Static Baseline -- the query point assumed never to move --
which is the bar a tracker has to clear.

With a feature cache this trains pure TRACKING by default: every frame keeps
its image, so `visual_mask` is all-True and the null embedding is never used.
`--random-cutoff 1` trains FORECASTING as Gen-points does: each clip keeps the
images of its frames before a random cutoff, the rest get the null embedding,
and the model still predicts every frame. `--eval-cutoff C` validates the same
way with one cutoff for every clip, and scores observed and forecast frames
apart (`_obs`, `_fut`).

Without a feature cache this falls back to the step-2 model, which sees no
images at all -- a pure motion prior rather than tracking.

Run:  python -m genpoint3d.prepare.main --root DATA --out local/cache/kubric
      python -m genpoint3d.train.main --cache local/cache/kubric --steps 20000 --batch 8
"""

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Optimiser import Optimiser
from genpoint3d.train.Progress import Progress
from genpoint3d.train.Saver import Saver
from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.Loaders import Loaders
from genpoint3d.train.data.check_target_scale import check_target_scale
from genpoint3d.train.data.apply_run_settings import apply_run_settings
from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.learn import learn
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
    check_target_scale(loaders, settings)
    model = build_model(settings, clips.feat_dim, clips.has_feats)
    model = set_up_model(model, settings, hardware)
    optimiser = Optimiser(model, settings, hardware)
    judging = validation_settings(settings, hardware)
    progress = Progress(settings)
    for step in range(1, settings.steps + 1):
        for batch in loaders.micro_batches(settings.accum):
            losses = learn(model, batch, settings, hardware)
            progress.add(losses)
        optimiser.step(model)
        progress.step(step, optimiser.lr())
        if step % settings.val_every == 0 or step == settings.steps:
            optimiser.swap_in_average(model)
            loader = loaders.val
            scores = score(model, loader, judging)
            fit_scores = None
            if loaders.fit is not None:
                loader = loaders.fit
                fit_scores = score(model, loader, judging)
            report = progress.report(step, scores, fit_scores)
            saver.save(report, model, clips)
            optimiser.swap_out_average(model)
    progress.finish()


if __name__ == "__main__":
    import logging
    import sys

    import torch

    # The package's progress lines, message only, on stdout.
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("genpoint3d").setLevel(logging.INFO)
    settings = Settings.from_command_line()
    torch.manual_seed(settings.seed)
    train(settings)
