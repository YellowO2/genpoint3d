"""Everything a training run is told on its command line."""

import argparse


class Settings(argparse.Namespace):
    """The run's flags, one attribute each. `vars(settings)` is what a
    checkpoint saves as its `args`, and `Settings(**ckpt["args"])` is that
    run's settings again. The defaults are the current design."""

    @classmethod
    def from_command_line(cls, argv: list[str] | None = None) -> "Settings":
        p = argparse.ArgumentParser(
            prog="python -m genpoint3d.train.main",
            description="Train the tracker on cached clips and save its checkpoints.")
        # Data and run
        p.add_argument("--cache", required=True, help="cache directory from genpoint3d.prepare")
        p.add_argument("--frames-cache", required=True, help="RGB and depth from scripts/cache_frames.py")
        p.add_argument("--fine-res", type=int, default=384, help="side of the cached frames")
        p.add_argument("--out", default="local/outputs/run")
        p.add_argument("--init", default=None, metavar="CKPT",
                       help="start from this run's weights, by name and shape")
        p.add_argument("--fine-init", default=None,
                       help="TAPIP3D's CNN and comparison (scripts/extract_tapip3d_finder.py)")
        p.add_argument("--steps", type=int, default=1000)
        p.add_argument("--batch", type=int, default=1)
        p.add_argument("--accum", type=int, default=1, help="micro-batches per optimiser step")
        p.add_argument("--points", type=int, default=500, help="points drawn per clip")
        p.add_argument("--level", type=int, default=0, choices=[0, 1],
                       help="turn every clip so up is -y (needs the cache's up)")
        p.add_argument("--val-frac", type=float, default=0.1)
        p.add_argument("--val-every", type=int, default=250)
        p.add_argument("--train-eval", type=int, default=0, metavar="N",
                       help="also score N train clips at each validation")
        p.add_argument("--overfit", type=int, default=0, metavar="N",
                       help="train and validate on the first N clips")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="bf16 autocast")
        p.add_argument("--seed", type=int, default=0)
        # Model
        p.add_argument("--dim", type=int, default=256)
        p.add_argument("--depth", type=int, default=6)
        p.add_argument("--heads", type=int, default=4)
        p.add_argument("--cross-window", type=int, default=49, help="DINOv3 patches each point reads")
        p.add_argument("--fine-neigh", type=int, default=32, help="CNN cells per 3D neighbourhood")
        p.add_argument("--rope-current", type=int, default=0, choices=[0, 1],
                       help="the attention between points encodes each point's current guess")
        p.add_argument("--group-feedback", type=int, default=16, metavar="E",
                       help="width of each point's group vector, fed back each call (0: none)")
        p.add_argument("--feedback-scale", type=float, default=10.0,
                       help="multiplies what the group vector adds")
        # Refinement
        p.add_argument("--window", type=int, default=16, help="frames per window, 0 = the whole clip")
        p.add_argument("--overlap", type=int, default=4, help="frames each window shares")
        p.add_argument("--schedule", default="fbfb", metavar="SWEEPS",
                       help="sweeps over the windows, one round each: f forward, b backward")
        p.add_argument("--refine-iters", type=int, default=4, help="rounds forward, with no --schedule")
        p.add_argument("--min-context", type=int, default=2,
                       help="training cutoffs and gaps start at or after this frame")
        p.add_argument("--max-gap", type=int, default=12,
                       help="half the training clips lose a span of 2 to this many frames "
                            "instead of their end (0: cutoffs only)")
        p.add_argument("--eval-cutoff", type=int, default=24, metavar="C",
                       help="validate with images before frame C only (0: every frame)")
        # Losses
        p.add_argument("--hidden-loss", type=float, default=1.0, help="weight of a hidden frame's error")
        p.add_argument("--vel-loss", type=float, default=1.0, help="frame-to-frame step error")
        p.add_argument("--rigid-loss", type=float, default=1.0, help="rigid pairs' gap error")
        p.add_argument("--group-loss", type=float, default=1.0, help="group vectors against rigid pairs")
        p.add_argument("--vis-loss", type=float, default=1.0, help="the visibility head (0: no head)")
        # Optimiser
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--clip", type=float, default=20.0, help="gradient norm clip")
        p.add_argument("--warmup-frac", type=float, default=0.05)
        p.add_argument("--wdecay", type=float, default=1e-4)
        p.add_argument("--ema", type=float, default=0.999)
        settings = p.parse_args(argv, namespace=cls())
        if settings.eval_cutoff == 1 or settings.eval_cutoff < 0:
            p.error("--eval-cutoff is 0 (off) or at least 2")
        if settings.window and not 0 < settings.overlap < settings.window:
            p.error("--window needs 0 < --overlap < --window")
        if settings.schedule and not settings.window:
            p.error("--schedule sweeps over the windows: needs --window W")
        if bool(settings.group_feedback) != (settings.group_loss > 0):
            p.error("--group-feedback E and --group-loss W go together")
        if min(settings.hidden_loss, settings.vis_loss, settings.vel_loss, settings.rigid_loss) < 0:
            p.error("loss weights are 0 or more")
        if settings.min_context < 1:
            p.error("--min-context is at least 1: frame 0 always keeps its image")
        return settings
