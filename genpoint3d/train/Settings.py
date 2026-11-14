"""Everything a training run is told on its command line."""

import argparse


# Flags since removed, as a checkpoint may still hold them: each with the value
# that left it off. A loss weight here only weighed its feature's loss.
REMOVED = {"fit_blend": 0, "physics": 0, "blend_hidden": 0, "trust_loss": None, "physics_loss": None}


class Settings(argparse.Namespace):
    """The run's flags, one attribute each. `vars(settings)` is what a
    checkpoint saves as its `args`, and `Settings.saved(ckpt["args"])` is that
    run's settings again. The defaults are the current design."""

    @classmethod
    def saved(cls, args: dict) -> "Settings":
        """A checkpoint's settings, its removed flags dropped where they were
        off; one that was on is an error, not a different model."""
        on = {k: args[k] for k, off in REMOVED.items() if off is not None and args.get(k, off) != off}
        if on:
            raise ValueError(f"the checkpoint was trained with features since removed: {on} "
                             "(check out a commit from before their removal to load it)")
        return cls(**{k: v for k, v in args.items() if k not in REMOVED})

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
        p.add_argument("--spawn", type=int, default=0, metavar="M",
                       help="up to M of each training clip's points start where new surface comes into view, frame "
                            "by frame, the clip's other tracks standing in for them (0: none)")
        p.add_argument("--spawn-gap", type=float, default=0.05,
                       help="a new point starts only this far (a share of the frame's width) from every point "
                            "tracked and seen there")
        p.add_argument("--val-frac", type=float, default=0.1)
        p.add_argument("--val-every", type=int, default=500)
        p.add_argument("--resume", default=None, metavar="RESUME_PT",
                       help="go on from an earlier job of this run: its resume.pt (weights, optimiser, schedule, log)")
        p.add_argument("--save-every", type=int, default=100,
                       help="steps between saves of the latest weights to ckpt.pt, between validations")
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
        p.add_argument("--group-feedback", type=int, default=16, metavar="E",
                       help="width of each point's group vector, fed back each call (0: none)")
        p.add_argument("--patch", type=int, default=0, metavar="P",
                       help="attention between points only within patches of P points near each other (Point "
                            "Transformer V3's serialized attention); 0: every point reads every other")
        p.add_argument("--fit-robust", type=int, default=0,
                       help="refits of the rigid proposal, each mate counting less the further it sat from the last "
                            "fit, so a few wrong mates cannot drag it (0: one plain fit)")
        p.add_argument("--fit-fresh", type=float, default=30.0, metavar="DEG",
                       help="the object's pose counts a mate 1 / (1 + turn / DEG)^2 as much, its turn since given (0: alike)")
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
        p.add_argument("--min-context", type=int, default=2,
                       help="training cutoffs and gaps start at or after this frame")
        p.add_argument("--max-gap", type=int, default=12,
                       help="half the training clips lose a span of 2 to this many frames "
                            "instead of their end (0: cutoffs only)")
        p.add_argument("--down-hidden", type=float, default=0.2,
                       help="share of training clips not told which way is down, as real video without its "
                            "camera's up")
        p.add_argument("--eval-cutoff", type=int, default=12, metavar="C",
                       help="validate with images before frame C only (0: every frame)")
        # Losses
        p.add_argument("--hidden-loss", type=float, default=1.0, help="weight of a hidden frame's error")
        p.add_argument("--vel-loss", type=float, default=1.0, help="frame-to-frame step error")
        p.add_argument("--rigid-loss", type=float, default=1.0, help="rigid pairs' gap error")
        p.add_argument("--group-loss", type=float, default=1.0, help="group vectors against rigid pairs")
        p.add_argument("--fit-loss", type=float, default=1.0,
                       help="the rigid fit's own positions from each round's group vectors against the truth, "
                            "training the groups through the fit (0: off)")
        p.add_argument("--centre-loss", type=float, default=1.0,
                       help="each point's arrow to its object's centre of mass, its tip against the truth where the "
                            "clip says (0: no arrow)")
        p.add_argument("--down-loss", type=float, default=1.0,
                       help="which way is down, said for clips not told it: its angle off the truth")
        p.add_argument("--group-labels", default="motion", choices=["object", "motion"],
                       help="the pairs to put together: on one object where the clip says (else as motion), "
                            "or those whose distance stays within 2 cm")
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
        if not settings.window:
            p.error("--schedule sweeps over the windows: needs --window W")
        if settings.mix is not None and (len(settings.mix) != len(settings.cache) or min(settings.mix) <= 0):
            p.error("--mix gives each --cache a share above 0")
        if settings.rigid_proposal and not settings.group_feedback:
            p.error("--rigid-proposal is built from the group vectors: needs --group-feedback E")
        if bool(settings.group_feedback) != (settings.group_loss > 0):
            p.error("--group-feedback E and --group-loss W go together")
        if min(settings.hidden_loss, settings.vis_loss, settings.vel_loss, settings.rigid_loss, settings.centre_loss) < 0:
            p.error("loss weights are 0 or more")
        if settings.min_context < 1:
            p.error("--min-context is at least 1: frame 0 always keeps its image")
        return settings
