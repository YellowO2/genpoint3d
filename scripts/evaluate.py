"""
Score a trained checkpoint on a cache. No training, no checkpoint selection.

This is what a held-out TEST set is for. The val split inside `train.py` picks
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
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genpoint3d.eval.metrics import clip_mean, tapvid3d_metrics
from genpoint3d.models.model import PointDiT
from genpoint3d.data.cache import CachedClip
from genpoint3d.data.transform import TRAJ_SCALE, TRAJ_SCALE_DISP
from train import EVAL_SEED, MATCH_KEYS, ClipDataset, evaluate, known_frame0, to_device, to_metres
from genpoint3d.models.flow import sample as flow_sample
from genpoint3d.models.regress import refine


@torch.no_grad()
def per_frame_apd(model, loader, device, steps: int, amp: bool,
                  anchor_frame0: bool = False, scaling: str = "median",
                  method: str = "flow", refine_iters: int = 4) -> list[float]:
    """APD for each frame index separately: APD3D over every visible point.

    The model is handed frame 0's true position as `anchor`, but still has to
    generate frame 0 from noise like every other frame -- nothing pins it. If
    frame 0 scores no better than the rest, the model is failing to reproduce a
    position it was given, and clamping it during sampling is worth doing. If
    frame 0 is clearly the best and the score decays with time, the error is
    accumulating drift instead and clamping would not address it.

    Seeded like `evaluate()`, so on the same loader these are the frames of the
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
        cond = dict(context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if method == "regress":
                pred = refine(model, anchor, num_frames=traj.shape[1],
                              iters=refine_iters, known_x0=kx0, **cond)
            else:
                pred = flow_sample(model, anchor, num_frames=traj.shape[1], steps=steps,
                                   generator=gen, known_x0=kx0, **cond)
        pred_m, gt_m = to_metres(pred.float(), b), to_metres(traj, b)

        if vals is None:
            vals = [[] for _ in range(traj.shape[1])]
        for t in range(traj.shape[1]):
            # One frame at a time, so each score is that frame's alone.
            m = tapvid3d_metrics(
                pred_m[:, t : t + 1], gt_m[:, t : t + 1], vis[:, t : t + 1],
                b["intrinsics_256"][:, t : t + 1], b["extrinsics"][:, t : t + 1],
                scaling=scaling, per_clip=True,
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
ABLATIONS = ("none", "swap-features", "noise-features", "shuffle-pos", "swap-id", "blind")


class Ablated:
    """Wraps a loader and corrupts each batch on the way out.

    `evaluate()` calls `to_device` itself, so this hands back CPU batches of the
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
            yield b


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="best.pt or ckpt.pt from a run")
    p.add_argument("--cache", required=True, help="cache DIRECTORY to score")
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
    p.add_argument("--per-frame", action="store_true",
                   help="also report APD per frame index, which separates a bad"
                        " start from accumulating drift")
    args = p.parse_args()

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
    print(f"{len(clips)} test clips | {device} | amp {'bf16' if amp else 'off'}"
          f" | points {points} | "
          + (f"{refine_iters} refinement iterations" if method == "regress"
             else f"{args.sample_steps} sampling steps"), flush=True)

    model = PointDiT(dim=targs["dim"], depth=targs["depth"],
                     num_heads=targs["heads"], cross_attn=has_feats,
                     feat_dim=feat_dim,
                     adapter_depth=targs.get("adapter_depth", 1),
                     upsample=targs.get("upsample", 1),
                     traj_scale=(TRAJ_SCALE_DISP if targs.get("target", "absolute") == "displacement"
                                 else TRAJ_SCALE),
                     displacement=targs.get("target", "absolute") == "displacement",
                     # Absent means trained before the lookup was fixed, and a
                     # checkpoint has to be evaluated the way it was trained.
                     legacy_lookup=bool(targs.get("legacy_lookup", 1)),
                     time_norm=bool(targs.get("time_norm", 0)),
                     # Absent means a checkpoint from before the prior existed.
                     locality=bool(targs.get("locality", 0)),
                     correlate=bool(targs.get("correlate", 0)),
                     # Absent means trained with the flat prior and the random
                     # projections; their parameters are what the file holds.
                     locality_mode=targs.get("locality_mode", "legacy"),
                     locality_wide=targs.get("locality_wide", 8.0),
                     corr_mode=targs.get("corr_mode", "legacy"),
                     # Absent means a checkpoint from before the flag existed,
                     # and every one of those was causal.
                     causal=bool(targs.get("causal", 1)),
                     costvol=bool(targs.get("costvol", 0)),
                     cv_k=targs.get("cv_k", 16),
                     cv_support=targs.get("cv_support", 8),
                     # Absent means a checkpoint from before the match could be
                     # trained: no head in the file, the raw cosine in the model.
                     match_learn=bool(targs.get("match_learn", 0)),
                     match_dim=targs.get("match_dim", 64)).to(device)
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
                    fix_idcard=bool(targs.get("fix_idcard", 0))),
        batch_size=args.batch, shuffle=False, num_workers=4,
    )

    t0 = time.time()
    anchor_frame0 = bool(targs.get("anchor_frame0", 0))
    loss_type = targs.get("loss_type", "l21")  # train.py's default
    print(f"scoring as trained: target {targs.get('target', 'absolute')}"
          f" | norm {targs.get('norm_mode', 'median')}"
          f" | anchor_frame0 {int(anchor_frame0)} | loss {loss_type}", flush=True)

    per_frame = per_frame_apd(model, loader, device, args.sample_steps, amp,
                              anchor_frame0, args.scaling, method,
                              refine_iters) if args.per_frame else None
    # One set of scoring rules for the intact score and every ablation of it.
    how = dict(steps=args.sample_steps, amp=amp, anchor_frame0=anchor_frame0,
               loss_type=loss_type, space=args.space, oracle_axis=args.oracle_axis,
               scaling=args.scaling, min_motion_px=args.min_motion_px,
               method=method, refine_iters=refine_iters)
    m = evaluate(model, loader, device, **how)
    print(f"\nscored in {(time.time() - t0) / 60:.1f} min\n", flush=True)

    for k in ("average_pts_within_thresh", "apd_static", "val_loss", "moving_frac",
              "disp_ratio", "disp_cos", *MATCH_KEYS):
        if k in m:
            print(f"  {k:<26} {m[k]:.4f}")
    if "moving_clips" in m:
        print(f"  {'moving_clips':<26} {m['moving_clips']} of {len(clips)}")
    print()
    for t in (1, 2, 4, 8, 16):
        print(f"  pts_within_{t:<15} {m[f'pts_within_{t}']:.4f}")

    if per_frame is not None:
        print("\n  APD by frame -- does error start at frame 0 or accumulate?")
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
            a = evaluate(model, Ablated(loader, kind), device, **how)
            apd = a["average_pts_within_thresh"]
            print(f"    {kind:<16} {apd:>7.4f}"
                  f" {apd / max(m['average_pts_within_thresh'], 1e-9):>9.2f}x", flush=True)
            m[f"ablate_{kind}"] = apd
        print("\n    ~1.00x means the model is not using that input at all.")
        print(f"    For scale, the static baseline is {m['apd_static']:.4f}.")

    if m["average_pts_within_thresh"] <= m["apd_static"]:
        print("\n  APD is at or below the static baseline -- assuming the points"
              " never move would score as well as this model.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"ckpt": args.ckpt, "cache": args.cache, "step": ckpt["step"],
             "sample_steps": args.sample_steps, "method": method,
             "refine_iters": refine_iters, "clips": len(clips), **m},
            indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
