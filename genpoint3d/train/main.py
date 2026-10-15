"""Train the tracker on cached clips, scoring and saving it as it goes.

Many clips, held-out validation, checkpoints. The question is whether the model
learns anything that transfers to clips it has never seen.

The metric that matters is APD on the VAL split -- `average_pts_within_thresh`
from the TAP-Vid-3D benchmark, scored in metres. `apd_static` is the
benchmark's own Static Baseline -- the query point assumed never to move --
which is the bar a tracker has to clear.

With a feature cache this trains pure TRACKING: every frame keeps its image, so
`visual_mask` is all-True and the null embedding is never used. Masking is
deliberately left off for the first learning run -- forecasting is far harder
than tracking, and training both at once splits the signal between them. If
tracking does not work, forecasting never would.

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
from genpoint3d.train.data.choose_clips import choose_clips
from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.learn import learn
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.model.set_up_model import set_up_model
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.score.main import score


def train(settings: Settings):
    hardware = Hardware.pick(settings)
    saver = Saver(settings)
    clips = split_clips(settings.cache, settings.val_frac, settings.seed)
    clips = choose_clips(clips, settings, hardware)
    loaders = Loaders(clips, settings)
    check_target_scale(loaders, settings)
    model = build_model(settings, clips.feat_dim, clips.has_feats)
    model = set_up_model(model, settings, hardware)
    optimiser = Optimiser(model, settings, hardware)
    judging = ScoreSettings.for_training(settings, hardware)
    progress = Progress(settings)
    for step in range(1, settings.steps + 1):
        for batch in loaders.micro_batches(settings.accum):
            losses = learn(model, batch, settings, hardware)
            progress.add(losses)
        optimiser.step(model)
        progress.step(step, optimiser.lr())
        if step % settings.val_every == 0 or step == settings.steps:
            optimiser.swap_in_average(model)
            scores = score(model, loaders.val, judging)
            fit_scores = None
            if loaders.fit is not None:
                fit_scores = score(model, loaders.fit, judging)
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
