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
        p.add_argument("--cache", nargs="+", required=True,
                       help="cache directories from genpoint3d.prepare; the first's val split picks best.pt")
        p.add_argument("--mix", type=float, nargs="+", default=None,
                       help="each cache's share of the training draws, in --cache's order (default: alike)")
        p.add_argument("--frames-cache", nargs="+", required=True,
                       help="each cache's RGB and depth, from scripts/cache_frames.py")
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
        p.add_argument("--val-every", type=int, default=500)
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
        p.add_argument("--rope-current", type=int, default=1, choices=[0, 1],
                       help="the attention between points encodes each point's current guess")
        p.add_argument("--group-feedback", type=int, default=16, metavar="E",
                       help="width of each point's group vector, fed back each call (0: none)")
        p.add_argument("--rigid-proposal", type=int, default=1, choices=[0, 1],
                       help="each point told where moving with its piece would put it (needs --group-feedback)")
        p.add_argument("--feedback-scale", type=float, default=10.0,
                       help="multiplies what the fed-back visibility and group vector add")
        # Refinement
        p.add_argument("--window", type=int, default=16, help="frames per window, 0 = the whole clip")
        p.add_argument("--overlap", type=int, default=4, help="frames each window shares")
        p.add_argument("--schedule", default="ffbb", metavar="SWEEPS",
                       help="sweeps over the windows, f forward, b backward; a run of one letter is one sweep, "
                            "its length the rounds a window gets before the next: ffbb")
        p.add_argument("--refine-iters", type=int, default=None, help="rounds forward, with no --schedule (4)")
        p.add_argument("--min-context", type=int, default=2,
                       help="training cutoffs and gaps start at or after this frame")
        p.add_argument("--partial-masks", type=int, default=1, choices=[0, 1],
                       help="training masks hide a quarter, half or three quarters of a frame too (0: whole frames)")
        p.add_argument("--max-gap", type=int, default=12,
                       help="half the training clips lose a span of 2 to this many frames "
                            "instead of their end (0: cutoffs only)")
        p.add_argument("--eval-cutoff", type=int, default=12, metavar="C",
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
        if settings.schedule and settings.refine_iters is not None:
            p.error("--refine-iters is the rounds without a schedule; a schedule sets its own")
        settings.refine_iters = settings.refine_iters or 4
        if settings.mix is not None and (len(settings.mix) != len(settings.cache) or min(settings.mix) <= 0):
            p.error("--mix gives each --cache a share above 0")
        if settings.rigid_proposal and not settings.group_feedback:
            p.error("--rigid-proposal is built from the group vectors: needs --group-feedback E")
        if bool(settings.group_feedback) != (settings.group_loss > 0):
            p.error("--group-feedback E and --group-loss W go together")
        if min(settings.hidden_loss, settings.vis_loss, settings.vel_loss, settings.rigid_loss) < 0:
            p.error("loss weights are 0 or more")
        if settings.min_context < 1:
            p.error("--min-context is at least 1: frame 0 always keeps its image")
        return settings
