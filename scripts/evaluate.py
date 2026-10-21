"""
Score a trained checkpoint on a cache. No training, no checkpoint selection.

This is what a held-out TEST set is for. The val split inside training picks
`best.pt`, so its APD is mildly optimistic -- it is the score of whichever
checkpoint happened to suit those exact clips. A set that is never used to
choose anything does not have that problem, and its number is the one to report.

Sequences 3500+ were downloaded separately for this and have never been seen by
training.

Run:  python scripts/evaluate.py --ckpt outputs/run3493/best.pt \
                                 --cache ~/scratch/cache/kubric_test
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genpoint3d.CachedClip import CachedClip
from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.score.Predictor import EVAL_SEED
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.score.main import score
from genpoint3d.train.tools.batch import fine_inputs, known_frame0, to_device, to_metres
from genpoint3d.train.tools.flow import sample as flow_sample
from genpoint3d.train.tools.matching import MATCH_KEYS
from genpoint3d.train.tools.metrics import HIDDEN_KEYS, clip_mean, motion_px, tapvid3d_metrics
from genpoint3d.train.tools.regress import refine


@torch.no_grad()
def per_frame_apd(model, loader, device, steps: int, amp: bool,
                  anchor_frame0: bool = False, scaling: str = "median",
                  method: str = "flow", refine_iters: int = 4,
                  param: str = "position", carry: bool = False) -> list[float]:
    """APD for each frame index separately: APD3D over every visible point
    that moves (more than 4 px), with no rescale -- the points the tracking
    is about, and where it falls off through the clip.

    The model is handed frame 0's true position as `anchor`, but still has to
    generate frame 0 from noise like every other frame -- nothing pins it. If
    frame 0 scores no better than the rest, the model is failing to reproduce a
    position it was given, and clamping it during sampling is worth doing. If
    frame 0 is clearly the best and the score decays with time, the error is
    accumulating drift instead and clamping would not address it.

    Seeded like `score()`, so on the same loader these are the frames of the
    very samples it scores. A mean over clips; a clip with no visible point in
    a frame is left out of that frame.
    """
    model.eval()
    vals = None
    gen = torch.Generator(device=device).manual_seed(EVAL_SEED)
    for batch in loader:
        b = to_device(batch, device, amp)
        traj, anchor, vis = b["traj"], b["anchor"], b["visibility"]
        ctx, idc, pxyz = b["context"], b["id_card"], b["patch_xyz"]
        vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=device) if ctx is not None else None
        kx0 = known_frame0(b) if anchor_frame0 else None
        cond = dict(context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz,
                    **fine_inputs(model, b, amp))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if method == "regress":
                pred = refine(model, anchor, num_frames=traj.shape[1],
                              iters=refine_iters, known_x0=kx0, param=param,
                              carry=carry, **cond)
            else:
                pred = flow_sample(model, anchor, num_frames=traj.shape[1], steps=steps,
                                   generator=gen, known_x0=kx0, **cond)
        pred_m, gt_m = to_metres(pred.float(), b), to_metres(traj, b)

        if vals is None:
            vals = [[] for _ in range(traj.shape[1])]
        moving = motion_px(gt_m, vis, b["intrinsics_256"], b["extrinsics"]) > 4
        seen = vis.bool() & moving[:, None]
        for t in range(traj.shape[1]):
            # One frame at a time, so each score is that frame's alone.
            m = tapvid3d_metrics(
                pred_m[:, t : t + 1], gt_m[:, t : t + 1], seen[:, t : t + 1],
                b["intrinsics_256"][:, t : t + 1], b["extrinsics"][:, t : t + 1],
                scaling="none", per_clip=True,
            )
            vals[t].append(m["average_pts_within_thresh"].cpu())
    model.train()
    return [clip_mean(v) for v in vals]


# Each ablation destroys ONE thing the model is supposed to rely on. A score
# that does not move is a pathway the model is not using.
#   swap-features   right clip, another clip's appearance
#   noise-features  appearance replaced by noise of the same mean and std
#   shuffle-pos     right patches, scrambled 3D positions
#   swap-id         another clip's query ID cards
#   blind           no features at all -- the forecasting null embedding
#   swap-frames     another clip's RGB frames, for the relative finder's CNN
#   no-fine         no RGB frames: the relative finder is skipped
#   hide-hidden     every image input blanked for each point on each frame
#                   where it is truly hidden, as if it were a forecast frame
# The first five are about DINOv3 and leave the frames alone; on a checkpoint
# without the relative finder the next two change nothing. hide-hidden reads
# the true visibility, so it is a diagnostic only: does the image of whatever
# is in front pull a hidden point onto it?
ABLATIONS = ("none", "swap-features", "noise-features", "shuffle-pos", "swap-id", "blind",
             "swap-frames", "no-fine", "hide-hidden")


class Ablated:
    """Wraps a loader and corrupts each batch on the way out.

    `score()` moves each batch itself, so this hands back CPU batches of the
    same shapes and dtypes and nothing downstream needs to know.
    """

    def __init__(self, loader, kind: str) -> None:
        self.loader, self.kind = loader, kind

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for b in self.loader:
            b = dict(b)
            ctx, pxyz, idc = b["context"], b["patch_xyz"], b["id_card"]
            if self.kind == "swap-features":
                b["context"] = ctx.roll(1, 0)          # needs batch > 1
            elif self.kind == "noise-features":
                b["context"] = torch.randn_like(ctx.float()).to(ctx.dtype) \
                    * ctx.float().std() + ctx.float().mean()
            elif self.kind == "shuffle-pos":
                b["patch_xyz"] = pxyz[:, :, torch.randperm(pxyz.shape[2])]
            elif self.kind == "swap-id":
                b["id_card"] = idc.roll(1, 0)
            elif self.kind == "blind":
                # to_device turns an empty tensor into None, and the model skips
                # cross-attention entirely -- no features, no positions.
                b["context"] = b["patch_xyz"] = b["id_card"] = torch.zeros(0)
            elif self.kind == "swap-frames":
                # with their depth: another clip's frames, whole
                b["frames"] = b["frames"].roll(1, 0)
                if b["depth"].numel():
                    b["depth"] = b["depth"].roll(1, 0)
            elif self.kind == "no-fine":
                b["frames"] = b["depth"] = torch.zeros(0)
            yield b


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="best.pt or ckpt.pt from a run")
    p.add_argument("--cache", required=True, help="cache DIRECTORY to score")
    p.add_argument("--frames-cache", default=None,
                   help="--fine checkpoints only: RGB frames of --cache's clips,"
                        " from scripts/cache_frames.py at the checkpoint's --fine-res")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--points", type=int, default=None,
                   help="default: whatever the checkpoint trained with")
    p.add_argument("--sample-steps", type=int, default=50,
                   help="Euler steps when sampling; the published protocol is not"
                        " prescriptive, so report whatever you use")
    p.add_argument("--refine-iters", type=int, default=None,
                   help="regress checkpoints only: refinement iterations. "
                        "default: whatever the checkpoint trained with. More "
                        "than that is a test of whether extra looks help")
    p.add_argument("--out", default=None, help="write the metrics as JSON here")
    p.add_argument("--ablate", nargs="*", default=None, choices=ABLATIONS,
                   help="also score with visual inputs corrupted. A score that"
                        " does not move names a pathway the model ignores."
                        " No argument means all of them")
    p.add_argument("--space", default="3d", choices=["3d", "2d"],
                   help="3d is APD3D, the reported metric. 2d projects both "
                        "tracks into the image and scores the original TAP-Vid "
                        "pixel thresholds, which is blind to depth error. The "
                        "two together say whether depth is what we get wrong.")
    p.add_argument("--scaling", default="median", choices=["median", "none"],
                   help="median is the TAPVid-3D protocol: the whole prediction "
                        "is multiplied by one factor so its median norm matches "
                        "the truth, because monocular depth has no scale. We "
                        "have metric depth, so that factor can only move a "
                        "correct point. none scores the prediction as it is.")
    p.add_argument("--min-motion-px", type=float, default=0.0,
                   help="score only points that move more than this many "
                        "(back-projected) pixels from their start. A point that "
                        "never moves is tracked perfectly by doing nothing, so "
                        "an average dominated by them hides the tracking.")
    p.add_argument("--oracle-axis", default="", choices=["", "x", "y", "z"],
                   help="overwrite one channel of the PREDICTION with the truth "
                        "before scoring. An error decomposition, not an input. "
                        "Correcting any single axis removes part of the error "
                        "budget, so this is only informative run on all three: "
                        "depth is the problem only if z buys more than x and y.")
    p.add_argument("--eval-cutoff", type=int, default=0, metavar="C",
                   help="forecast: frames C on get no image, and the observed"
                        " (_obs) and forecast (_fut) frames are scored apart,"
                        " beside the stop/cv baselines from the true past and"
                        " the own-stop one from the model's frame C-1. 0 = off")
    p.add_argument("--samples", type=int, default=1, metavar="K",
                   help="flow with --eval-cutoff: also score the oracle best of"
                        " K forecasts (_fut_bestK). Regress is deterministic"
                        " and ignores this")
    p.add_argument("--support", type=int, default=0, metavar="S",
                   help="regress, CNN-only models: also track an S x S grid of "
                        "helper points over frame 0 and score the queries only "
                        "(TAPIP3D's test-time support grid uses 16)")
    p.add_argument("--per-frame", action="store_true",
                   help="also report APD per frame index, which separates a bad"
                        " start from accumulating drift")
    args = p.parse_args()
    if args.eval_cutoff == 1 or args.eval_cutoff < 0 or args.samples < 1:
        p.error("--eval-cutoff is 0 (off) or at least 2, and --samples at least 1")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    # Every clip in the cache is the test set. split() is not reused here: it
    # always holds back at least one clip for validation, which is right for
    # training and would silently drop a test clip.
    clips = sorted(Path(args.cache).glob("*.pt"))
    if not clips:
        raise SystemExit(f"no cached clips in {args.cache}")
    probe = CachedClip.from_dict(torch.load(clips[0], weights_only=False, mmap=True))
    feat_dim, has_feats = probe.feat_dim, probe.context is not None
    points = args.points or targs["points"]
    # Absent means a checkpoint from before there was a second method.
    method = targs.get("method", "flow")
    refine_iters = args.refine_iters or targs.get("refine_iters", 4)
    # Absent means a checkpoint from before corrections could mean a speed.
    param = targs.get("param", "position")
    carry = bool(targs.get("carry", 0))
    print(f"{len(clips)} test clips | {device} | amp {'bf16' if amp else 'off'}"
          f" | points {points} | "
          + (f"{refine_iters} refinement iterations" if method == "regress"
             else f"{args.sample_steps} sampling steps"), flush=True)

    model = build_model(Settings(**targs), feat_dim, has_feats).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded step {ckpt['step']} from {args.ckpt}", flush=True)

    loader = DataLoader(
        # resample=False: a test score must not change between runs.
        # Every one of these changes what the numbers MEAN. Defaulting any of
        # them scores the model in a space it was not trained in, and the result
        # is not a bad score but a meaningless one.
        ClipDataset(clips, points, resample=False,
                    norm_mode=targs.get("norm_mode", "median"),
                    target=targs.get("target", "absolute"),
                    # Absent means trained on the ID cards as they were cached.
                    fix_idcard=bool(targs.get("fix_idcard", 0)),
                    frames=(ClipDataset.frames_dir(args.frames_cache, clips, targs.get("fine_res", 384),
                                                   depth=bool(targs.get("fine_xyz", 0)))
                            if targs.get("fine", 0) else None),
                    features=bool(targs.get("dino", 1)),
                    depth=bool(targs.get("fine_xyz", 0))),
        batch_size=args.batch, shuffle=False, num_workers=4,
    )

    t0 = time.time()
    anchor_frame0 = bool(targs.get("anchor_frame0", 0))
    loss_type = targs.get("loss_type", "l21")  # training's default
    print(f"scoring as trained: target {targs.get('target', 'absolute')}"
          f" | norm {targs.get('norm_mode', 'median')}"
          f" | anchor_frame0 {int(anchor_frame0)} | loss {loss_type}", flush=True)

    per_frame = per_frame_apd(model, loader, device, args.sample_steps, amp,
                              anchor_frame0, args.scaling, method,
                              refine_iters, param, carry) if args.per_frame else None
    # One set of scoring rules for the intact score and every ablation of it.
    judging = ScoreSettings(device=device, steps=args.sample_steps, amp=amp,
                            anchor_frame0=anchor_frame0, loss_type=loss_type,
                            space=args.space, oracle_axis=args.oracle_axis,
                            scaling=args.scaling, min_motion_px=args.min_motion_px,
                            method=method, refine_iters=refine_iters, param=param, carry=carry,
                            cutoff=args.eval_cutoff, samples=args.samples,
                            support=args.support)
    m = score(model, loader, judging, split="test")
    print(f"\nscored in {(time.time() - t0) / 60:.1f} min\n", flush=True)

    for k in ("average_pts_within_thresh", "apd_static", "val_loss", "moving_frac",
              "disp_ratio", "disp_cos", *MATCH_KEYS):
        if k in m:
            print(f"  {k:<26} {m[k]:.4f}")
    if "moving_clips" in m:
        print(f"  {'moving_clips':<26} {m['moving_clips']} of {len(clips)}")
    visible = [k for k in ("vis_acc", "vis_acc_hidden", "vis_acc_visible",
                           "average_jaccard", "occlusion_accuracy") if k in m]
    if visible:
        print("\n  visibility head, right where it can and cannot see the point,"
              " and AJ and OA on the observed frames:")
        for k in visible:
            print(f"  {k:<26} {m[k]:.4f}")
    hidden = [k for k in HIDDEN_KEYS if k in m]
    if hidden:
        print("\n  hidden on the observed frames, occluded or out of the frame,"
              " each beside the last-seen and constant-speed baselines:")
        for k in hidden:
            print(f"  {k:<30} {m[k]:.4f}")
    forecast = [k for k in m if k.endswith("_obs") or "_fut" in k]
    if forecast:
        print(f"\n  forecast, frames {args.eval_cutoff}+ without images:")
        for k in forecast:
            print(f"  {k:<26} {m[k]:.4f}")
    print()
    for t in (1, 2, 4, 8, 16):
        print(f"  pts_within_{t:<15} {m[f'pts_within_{t}']:.4f}")

    if per_frame is not None:
        print("\n  APD by frame -- does error start at frame 0 or accumulate?")
        if args.eval_cutoff:
            print("    NOTE: per-frame APD always gives every frame its image;"
                  " --eval-cutoff does NOT apply to it.")
        if args.space != "3d" or args.min_motion_px or args.oracle_axis:
            print("    NOTE: per-frame APD is always 3D, over all visible points,"
                  " with no oracle axis.\n    --space, --min-motion-px and"
                  " --oracle-axis do NOT apply to it; only --scaling does.")
        for t, v in enumerate(per_frame):
            bar = "#" * int(round(v / max(max(per_frame), 1e-9) * 40))
            print(f"    frame {t:>2}  {v:.4f}  {bar}")
        print(f"\n    frame 0 {per_frame[0]:.4f}  ->  frame {len(per_frame)-1}"
              f" {per_frame[-1]:.4f}")
        # Two separate questions, and comparing frames only answers the second.
        if per_frame[0] < 0.5:
            print(f"    frame 0 scores {per_frame[0]:.3f} despite being handed its"
                  " own position as `anchor`, so pinning it is worth doing --"
                  " every later frame inherits that error.")
        if per_frame[-1] < 0.5 * per_frame[0]:
            print("    the score also decays with time, so error accumulates"
                  " during the rollout on top of any bad start.")

    if args.ablate is not None:
        kinds = [k for k in (args.ablate or ABLATIONS) if k != "none"]
        print("\n  Ablations -- APD with one visual input destroyed:\n")
        print(f"    {'condition':<16} {'APD':>7} {'vs intact':>10}")
        print(f"    {'intact':<16} {m['average_pts_within_thresh']:>7.4f} {'--':>10}")
        for kind in kinds:
            # hide-hidden changes what the model is shown per point, which the
            # batch alone cannot say, so it goes through the scoring rules.
            a = (score(model, loader, replace(judging, hide_hidden=True), split=kind)
                 if kind == "hide-hidden"
                 else score(model, Ablated(loader, kind), judging, split=kind))
            apd = a["average_pts_within_thresh"]
            print(f"    {kind:<16} {apd:>7.4f}"
                  f" {apd / max(m['average_pts_within_thresh'], 1e-9):>9.2f}x", flush=True)
            m[f"ablate_{kind}"] = apd
            if kind == "hide-hidden":
                # The point of this one is the hidden score, so print it too.
                for k in HIDDEN_KEYS:
                    if k in a:
                        print(f"      {k:<30} {a[k]:.4f}  (intact {m.get(k, float('nan')):.4f})")
                        m[f"ablate_{kind}_{k}"] = a[k]
        print("\n    ~1.00x means the model is not using that input at all.")
        print(f"    For scale, the static baseline is {m['apd_static']:.4f}.")

    if m["average_pts_within_thresh"] <= m["apd_static"]:
        print("\n  APD is at or below the static baseline -- assuming the points"
              " never move would score as well as this model.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"ckpt": args.ckpt, "cache": args.cache, "step": ckpt["step"],
             "sample_steps": args.sample_steps, "method": method,
             "refine_iters": refine_iters, "clips": len(clips),
             "eval_cutoff": args.eval_cutoff, "samples": args.samples, **m},
            indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
