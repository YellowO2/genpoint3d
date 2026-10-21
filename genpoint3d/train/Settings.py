"""Everything a training run is told on its command line."""

import argparse


class Settings(argparse.Namespace):
    """The run's flags, one attribute each, as `argparse` parses them.

    `vars(settings)` is what a checkpoint saves as its `args`, and
    `Settings(**ckpt["args"])` is that run's settings again. A flag missing
    from an older checkpoint was added after it was trained; whoever reads one
    gives the default every model was built with until then.
    """

    @classmethod
    def from_command_line(cls, argv: list[str] | None = None) -> "Settings":
        p = argparse.ArgumentParser(
            prog="python -m genpoint3d.train.main",
            description="Train the tracker on cached clips and save its checkpoints.")
        p.add_argument("--cache", required=True, help="cache DIRECTORY from genpoint3d.prepare")
        p.add_argument("--out", default="local/outputs/run")
        p.add_argument("--steps", type=int, default=20000)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--points", type=int, default=128)
        p.add_argument("--train-eval", type=int, default=0, metavar="N",
                       help="also score N clips from the TRAIN split at every "
                            "validation, logged as train_average_pts_within_thresh. Without it a run "
                            "reports only val APD, which cannot separate bias from "
                            "variance: a model that scores badly on clips it has "
                            "trained on is underfitting, and more data will not "
                            "help it. Costs one extra eval pass per interval.")
        p.add_argument("--method", default="flow", choices=["flow", "regress"],
                       help="flow: flow matching, the network denoises a noisy "
                            "sample. regress: the same network refines its own "
                            "guess from \"nothing moves\", with no noise, as "
                            "TAPIP3D and CoTracker do. The control that tells a "
                            "noise problem from a feature problem.")
        p.add_argument("--refine-iters", type=int, default=4,
                       help="regress only: refinement iterations, in training and "
                            "in evaluation. Each is a full forward and backward "
                            "pass, so a step costs this many times a flow step in "
                            "time, and the same in memory. TAPIP3D trains with 4.")
        p.add_argument("--param", default="position",
                       choices=["position", "velocity", "acceleration"],
                       help="regress with --target displacement only: what each "
                            "per-frame correction the network gives means. "
                            "position: a change to that frame's displacement. "
                            "velocity: a change to the speed from that frame on. "
                            "acceleration: a change to how fast the speed changes "
                            "from that frame on. Summed along time into a "
                            "displacement before anything uses it, so a point the "
                            "network cannot see keeps moving as it was rather than "
                            "staying put. Not for flow: there the output is a "
                            "velocity from the noisy sample to the tracks, not "
                            "along time. See tools/regress.py.")
        p.add_argument("--accum", type=int, default=1,
                       help="micro-batches per optimiser step. A wider model does "
                            "not fit at --batch 16 on a 40GB A100, and halving the "
                            "batch would change the optimisation as well as the "
                            "capacity. accum keeps the effective batch fixed so a "
                            "size comparison isolates size.")
        p.add_argument("--dim", type=int, default=256)
        p.add_argument("--depth", type=int, default=6)
        p.add_argument("--heads", type=int, default=4)
        # 5e-4 is what both references use on Kubric -- CoTracker3's
        # launch_training_kubric_offline.sh and genpt's train_tracker_tapvid_kubric.
        p.add_argument("--lr", type=float, default=5e-4)
        p.add_argument("--clip", type=float, default=10.0,
                       help="gradient-norm clip. 10 is TAPIP3D's. The norm is 4 to 9 "
                            "for most of a run (logged every 100 steps), so 1.0, the "
                            "default until 2026-10-21, shrank every step 4 to 9 times: "
                            "CNN-only went from 0.30 to 0.45 moving APD at step 1000 "
                            "when it was lifted. 0 disables it.")
        # A FRACTION, not a step count. 500 fixed steps was 20% of a 2500-step run
        # and 2.5% of a 20000-step one -- the same flag meaning two different
        # things. CoTracker3 uses OneCycleLR with pct_start=0.05.
        p.add_argument("--warmup-frac", type=float, default=0.05)
        p.add_argument("--wdecay", type=float, default=1e-4,
                       help="CoTracker3 uses 5e-4 on Kubric; ours was 0.01, 20x more")
        p.add_argument("--val-frac", type=float, default=0.1)
        p.add_argument("--val-every", type=int, default=2000)
        # Non-zero by default: clips are loaded lazily now, so with 0 workers the
        # GPU sits idle while the main process reads 7.3 MB/clip off Lustre and
        # converts the fp16 features to fp32. 4 is enough to stay ahead at batch 16.
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                       help="bf16 autocast on CUDA; roughly halves memory and time")
        # On by default: an unweighted loss optimises something the metric does not
        # measure. 0 reproduces every run before 2026-09-18.
        p.add_argument("--depth-scaled-loss", type=int, default=1, choices=[0, 1])
        # Frame 0 is handed to the model as `anchor` and was still its worst-scoring
        # frame in absolute terms (APD 0.028 on run3493), so it is pinned rather
        # than denoised. 0 reproduces the earlier behaviour.
        # Bias cross-attention towards patches near each point's current position.
        # 0 is the model that ignored which video it was given.
        p.add_argument("--adapter-depth", type=int, default=1,
                       help="layers in the feature adapter. The encoder is cached "
                            "frozen, so this map is the ONLY thing that adapts "
                            "DINOv3 to tracking; at depth 1 it is a single linear "
                            "layer. Deepening it is the cheap stand-in for "
                            "unfreezing the backbone, which the cache forbids.")
        p.add_argument("--upsample", type=int, default=1,
                       help="feature-grid upsampling factor per side, the paper's "
                            "nearest-interpolation-plus-conv block. 2 turns a "
                            "48x48 grid into 96x96, and P grows 4x with it.")
        p.add_argument("--legacy-lookup", type=int, default=0, choices=[0, 1],
                       help="1 reproduces the lookups as they were before 2026-10-09, "
                            "centred on the raw sample instead of the point's "
                            "position. Only for a like-for-like control run.")
        p.add_argument("--time-norm", type=int, default=1, choices=[0, 1],
                       help="frame times in [-1, 1] for temporal RoPE. 0 feeds the "
                            "raw frame index, as every run before 2026-10-09 did.")
        p.add_argument("--fix-idcard", type=int, default=1, choices=[0, 1],
                       help="re-read each query's ID card from frame 0's cached "
                            "features at the right place. 0 uses the cached one, "
                            "sampled up to half a patch off, as every run before "
                            "2026-10-09 did.")
        p.add_argument("--locality", type=int, default=1, choices=[0, 1])
        p.add_argument("--locality-mode", default="sched", choices=["legacy", "patch", "sched"],
                       help="patch: the prior is a Gaussian one patch spacing wide, "
                            "measured on each frame's own grid. sched: the same, "
                            "but --locality-wide spacings wide at k = 0 and "
                            "narrowing to one as the sample gets clean. legacy: a "
                            "strength of 1.3 in scene units squared, which at real "
                            "patch spacing is flat, as every run up to run14 had it.")
        p.add_argument("--locality-wide", type=float, default=8.0,
                       help="sched only: the prior's starting width at pure noise, "
                            "in patch spacings. 8 is a third of a 24x24 frame. "
                            "Both ends are learned from there.")
        # Score each patch against the query template, so "does this look like me"
        # is an input rather than something to be discovered. 0 is the run5 model.
        p.add_argument("--correlate", type=int, default=1, choices=[0, 1])
        p.add_argument("--corr-mode", default="cosine", choices=["legacy", "cosine"],
                       help="cosine: score patches by cosine similarity of the raw "
                            "features, and hand the model the best match's offset "
                            "in the units of its target. legacy: two random "
                            "projections behind zero gates, as every run up to "
                            "run14 had it.")
        # 0 lets a frame see later frames. Legitimate for tracking, where every
        # image is available, and what the published trackers do. Forecasting does
        # not depend on it: a masked frame holds only `null_ctx`.
        p.add_argument("--causal", type=int, default=1, choices=[0, 1])
        # Compare a support WINDOW around the query against the patches nearest the
        # current estimate, and MLP the resulting table. CoTracker3/genpt core.
        p.add_argument("--costvol", type=int, default=1, choices=[0, 1])
        p.add_argument("--cv-k", type=int, default=16, help="neighbour patches")
        p.add_argument("--cv-support", type=int, default=8, help="support patches")
        p.add_argument("--match-learn", type=int, default=0, choices=[0, 1],
                       help="train the match. The whole-frame search for the patch "
                            "that looks like the point runs on frozen features and "
                            "is right about half the time. 1 adds a small residual "
                            "MLP on the features it compares, starting as the raw "
                            "cosine, and a loss that names the right patch for "
                            "every visible point. 0 is every run before it.")
        p.add_argument("--match-topk", type=int, default=1,
                       help="how many candidate matches the model is handed per "
                            "point and frame. 1 is the single best patch, every "
                            "run before this. K > 1 adds the next K-1 distinct "
                            "peaks of the score map (each peak rules out its 3x3 "
                            "before the next is picked), in rank order; their "
                            "weights start at zero, so step 0 is the K=1 model")
        p.add_argument("--match-loss-weight", type=float, default=1.0,
                       help="--match-learn only: weight of the matching loss "
                            "(cross-entropy over a frame's patches) next to the "
                            "method's own loss")
        p.add_argument("--match-dim", type=int, default=64,
                       help="--match-learn only: hidden width of the residual MLP. "
                            "Its memory is a few (batch, frames, patches, this) "
                            "tensors, so it is kept well under the feature width")
        p.add_argument("--fine", type=int, default=0, choices=[0, 1],
                       help="the relative finder: a small CNN on the RGB frames, "
                            "one cell per 4 px, trained with the model, and a "
                            "7x7 window of it around the current guess compared "
                            "with one around the query. What gives the reference "
                            "trackers their last few pixels. Needs --frames-cache. "
                            "0 is every run before it.")
        p.add_argument("--frames-cache", default=None,
                       help="--fine only: directory from scripts/cache_frames.py, "
                            "one uint8 array of frames per clip of --cache")
        p.add_argument("--fine-res", type=int, default=384,
                       help="--fine only: side of the square frame the CNN is "
                            "given, which is the side --frames-cache was written "
                            "at. A cell is 4 px of it: 384 is 96x96 cells.")
        p.add_argument("--fine-levels", type=int, default=4,
                       help="--fine only: pyramid levels, each half as fine")
        p.add_argument("--fine-radius", type=int, default=3,
                       help="--fine only: the window is this many cells either "
                            "side of its centre, at every level. 3 is 7x7")
        p.add_argument("--fine-xyz", type=int, default=0, choices=[0, 1],
                       help="--fine only: the finder also reads where each "
                            "cell of its windows is in 3D, from the depth map "
                            "at 4 px per cell, as an offset from the current "
                            "guess. Needs depth in --frames-cache "
                            "(scripts/cache_frames.py writes it). Starts at "
                            "zero, so step 0 is the model without it")
        p.add_argument("--fine-init", default=None,
                       help="--fine only: weights for the CNN, as saved by "
                            "scripts/fetch_fnet.py (CoTracker3's). Without it "
                            "the CNN starts from random")
        p.add_argument("--dino", type=int, default=1, choices=[0, 1],
                       help="0 gives the model nothing from DINOv3: no patch "
                            "tokens to attend to, no match, no cost volume, no ID "
                            "card. With --fine 1 that is a tracker on the CNN "
                            "alone, as the reference trackers are, and the query "
                            "is described by its window of CNN features")
        p.add_argument("--random-cutoff", type=int, default=0, choices=[0, 1],
                       help="forecasting, as Gen-points trains it: each clip draws "
                            "a cutoff T_C uniform in [--min-context, T] and keeps "
                            "the images of frames before it only; the model "
                            "predicts every frame. T_C = T is pure tracking. 0 "
                            "trains pure tracking, every run before it.")
        p.add_argument("--min-context", type=int, default=2,
                       help="--random-cutoff only: the fewest frames a clip "
                            "keeps the images of. At least 1, frame 0 being the "
                            "query frame.")
        p.add_argument("--eval-cutoff", type=int, default=0, metavar="C",
                       help="validate as a forecast: every clip keeps the images "
                            "of frames before C only, and the scores are also "
                            "reported apart for those frames (_obs) and the rest "
                            "(_fut), with the stop and constant-velocity "
                            "baselines beside the future ones. At least 2, which "
                            "constant velocity needs. 0 validates pure tracking.")
        p.add_argument("--vel-loss", type=float, default=0.0, metavar="W",
                       help="regress only: the loss also scores how much the "
                            "guess moves from each frame to the next against "
                            "how much the truth does, weighted W against the "
                            "position loss, on the frames --hidden-loss counts. "
                            "0 scores positions only.")
        p.add_argument("--acc-loss", type=float, default=0.0, metavar="W",
                       help="regress only: as --vel-loss, for how the guess's "
                            "speed changes from each frame to the next against "
                            "how the truth's does. 0 leaves it out.")
        p.add_argument("--rigid-loss", type=float, default=0.0, metavar="W",
                       help="regress only: the loss also scores the gap between "
                            "every two points the truth shows on one moving "
                            "rigid object, in every frame, hidden ones too, "
                            "weighted W against the position loss. 0 leaves it out.")
        p.add_argument("--motion-waves", type=int, default=0, choices=[0, 1],
                       help="--fine only: also give each token, Fourier-encoded, "
                            "how its guess moved from the frame before and to the "
                            "frame after, and where it is in the image "
                            "(TAPIP3D's updater input).")
        p.add_argument("--buddies", type=int, default=0, choices=[0, 1],
                       help="give each point where the points that have kept "
                            "their distance to it say it is: carried by their "
                            "rotation and shift since frame 0 (Buddies.py).")
        p.add_argument("--buddy-bias", type=int, default=0, choices=[0, 1],
                       help="--buddies only: also a learned spatial attention "
                            "bias per pair of points, from the same numbers.")
        p.add_argument("--fine-waves", type=int, default=0, metavar="W",
                       help="--fine-xyz only: also write each window cell's 3D "
                            "offset as sin and cos at W frequencies doubling from "
                            "1/2 per unit, a finer ruler for the finder. 0 keeps "
                            "the raw offsets alone.")
        p.add_argument("--fine-width", type=int, default=64,
                       help="numbers the finder keeps per pyramid level from each "
                            "window's comparison table.")
        p.add_argument("--carry", type=int, default=0, choices=[0, 1],
                       help="regress only: between refinement iterations, the "
                            "frames past a horizon that moves to the last frame "
                            "restart from the horizon's guess, so each frame's "
                            "window starts near where the point last was "
                            "(`regress._carry`).")
        p.add_argument("--fine-neigh", type=int, default=0, metavar="K",
                       help="--fine-xyz only: replace the finder's 7x7 window "
                            "with TAPIP3D's lookup, the K cells nearest the guess "
                            "in 3D against the K nearest the query, compared by "
                            "attention (FineFinder._Neighbours). TAPIP3D uses 32; "
                            "0 keeps the window.")
        p.add_argument("--virtual", type=int, default=0, metavar="M",
                       help="--dino 0 only: points meet through M learned virtual "
                            "tracks in each frame instead of attending to each "
                            "other directly (TAPIP3D uses 64). 0 keeps direct "
                            "spatial attention.")
        p.add_argument("--hidden-loss", type=float, default=1.0, metavar="W",
                       help="the method's loss also counts the frames where a "
                            "point is hidden (occluded or out of the frame), "
                            "weighted W against a visible one. Kubric gives the "
                            "true position there too. 1 counts them the same; 0 "
                            "trains on visible frames only, every run before it. "
                            "The match loss stays on visible points.")
        p.add_argument("--vis-loss", type=float, default=0.0, metavar="W",
                       help="a visibility head: for every (point, frame) the "
                            "model also says whether it can see the point "
                            "there, trained by cross-entropy weighted W next to "
                            "the method's loss. Not visible is occluded, out of "
                            "the frame, or in a forecast frame with no image. "
                            "0 builds no head: every run before it.")
        p.add_argument("--vis-feedback", type=int, default=0, choices=[0, 1],
                       help="regress only, with --vis-loss: each refinement "
                            "iteration is also given the visibility the one "
                            "before it predicted (0 before the first), as "
                            "TAPIP3D does, so a point the model thinks is hidden "
                            "can learn to stop trusting what the image shows "
                            "there. 0: every run before it.")
        p.add_argument("--anchor-frame0", type=int, default=1, choices=[0, 1])
        # Evaluate and checkpoint the averaged weights, not the jittering ones.
        # 0 disables. The paper lists EMA among its training ingredients.
        p.add_argument("--ema", type=float, default=0.999)
        # l21 is the Euclidean distance, which is what the metric counts. l2 squares
        # it, so a few catastrophic points dominate -- visible in run3493 as rmse
        # sitting flat at 0.11 while APD tripled.
        p.add_argument("--loss-type", default="l21", choices=["l2", "l21"])
        p.add_argument("--target", default="absolute",
                       choices=["absolute", "displacement"],
                       help="what the model denoises towards; see ClipDataset")
        p.add_argument("--norm-mode", default="median",
                       choices=["median", "mean", "centroid_max"],
                       help="scene normalisation; see genpoint3d/ScaleStats.py")
        p.add_argument("--overfit", type=int, default=0,
                       help="diagnostic: train AND validate on the first N clips. "
                            "APD should reach ~100%%; if it plateaus, the pipeline "
                            "has a bug rather than a data problem")
        p.add_argument("--seed", type=int, default=0)
        settings = p.parse_args(argv, namespace=cls())
        if settings.eval_cutoff == 1 or settings.eval_cutoff < 0:
            p.error("--eval-cutoff is 0 (off) or at least 2")
        if settings.fine_xyz and not settings.fine:
            p.error("--fine-xyz adds to the relative finder: it needs --fine 1")
        if (settings.vel_loss or settings.acc_loss or settings.rigid_loss) and settings.method != "regress":
            p.error("--vel-loss, --acc-loss and --rigid-loss are only built for --method regress")
        if settings.motion_waves and not settings.fine:
            p.error("--motion-waves places the guess in the image through the finder's camera: needs --fine 1")
        if settings.motion_waves and settings.causal:
            p.error("--motion-waves reads the step to the next frame: needs --causal 0")
        if settings.virtual and settings.dino:
            p.error("--virtual is built for the CNN-only model: --dino 0")
        if settings.carry and settings.method != "regress":
            p.error("--carry restarts frames between refinement iterations: regress only")
        if settings.fine_neigh and (not settings.fine_xyz or settings.fine_waves):
            p.error("--fine-neigh needs --fine-xyz, and encodes its own offsets (no --fine-waves)")
        if settings.fine_waves and not settings.fine_xyz:
            p.error("--fine-waves encodes the offsets --fine-xyz reads")
        if settings.hidden_loss < 0:
            p.error("--hidden-loss is a weight: 0 or more")
        if settings.vis_loss < 0:
            p.error("--vis-loss is a weight: 0 or more")
        if settings.vis_feedback and (settings.vis_loss <= 0 or settings.method != "regress"):
            p.error("--vis-feedback feeds the visibility head back between "
                    "refinement iterations: it needs --vis-loss > 0 and --method regress")
        if settings.param != "position" and (settings.method != "regress"
                                             or settings.target != "displacement"):
            p.error("--param velocity/acceleration sums the corrections along time "
                    "from frame 0, which stays where it is: it needs --method "
                    "regress and --target displacement")
        if settings.min_context < 1:
            p.error("--min-context is at least 1: frame 0 always keeps its image")
        return settings
