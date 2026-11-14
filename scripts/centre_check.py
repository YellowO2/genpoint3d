"""How well the points' arrows find their object's centre of mass, on clips
the run never trained on (its own held-out split of a cache with centres,
our rendered clips): in each frame, each true object's centre from the tips
of its points' arrows (their mean), against the middle of its points seen
there (the model's guesses, and the truth: the best that way can do) and the
middle of all its points' guesses. Object-frames split by how much of the
object's points is seen there; the middle of the seen points has nothing to
go on where none is. Errors in cm, the median over object-frames.

Tracked as at test time: with `--fill`, points are first added on new
surface (they track but are not scored: they have no truth).

Run:  python scripts/centre_check.py --ckpt outputs/RUN/best.pt --cache cache/kubric_obj48_centre \
          --frames-cache cache/kubric_obj48_centre_frames384
"""

import argparse

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.score.Run import Run
from genpoint3d.train.score.spawn_points import spawn_points
from genpoint3d.train.tools.batch import fine_inputs, to_device, to_metres, visual_mask
from genpoint3d.train.tools.kabsch import MIN_POSE_POINTS
from genpoint3d.train.tools.regress import Sweeps


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True, help="a cache with centres; its held-out split is scored")
    p.add_argument("--frames-cache", required=True)
    p.add_argument("--fill", type=int, default=1000, help="points each clip is filled up to first (0: none)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run.load(args.ckpt, device)
    model, s = run.model, run.settings
    if not model.centre:
        raise SystemExit("the run has no arrows")
    clips = split_clips([args.cache], s.val_frac, s.seed).val
    loader = DataLoader(run.dataset(clips, args.frames_cache), batch_size=1, shuffle=False, num_workers=4)
    amp = device.type == "cuda"
    print(f"{len(clips)} held-out clips | step {run.step} of {args.ckpt} | filled to {args.fill}", flush=True)

    names = ("arrows", "seen (guess)", "seen (truth)", "all (guess)")
    bins = ("none seen", "under half", "half or more")
    errs = {}
    for batch in loader:
        b = to_device(batch, device)
        N, T = b["traj"].shape[2], b["traj"].shape[1]
        fine, view = fine_inputs(model, b, amp), visual_mask(b, None)

        def track(bb):
            cond = dict(context=bb["context"], visual_mask=view, id_card=bb["id_card"], patch_xyz=bb["patch_xyz"],
                        **{**fine, "query_t": bb["query_t"]})
            sweeps = Sweeps(model, bb["anchor"], T, s.schedule, s.window, s.overlap, cond)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in sweeps:
                    pass
            return sweeps

        sweeps = track(b)
        if args.fill:
            every = spawn_points(model, b, fine, view, args.fill, run.spawn_gap,
                                 lambda bb: (lambda sw: (sw.whole, sw.vis))(track(bb)))
            sweeps = track(every)
        guess = sweeps.whole[:, :, :N].float()
        tips = to_metres(guess + sweeps.arrow[:, :, :N], b)[0]                       # (T, N, 3) metres
        guess, true = to_metres(guess, b)[0], to_metres(b["traj"].float(), b)[0]
        centre = to_metres(b["centre"].float(), b)[0]
        vis, seg, known = b["visibility"][0].bool(), b["segment"][0], b["centre_known"][0]
        for k in seg[known].unique().tolist():
            mine = known & (seg == k)
            if mine.sum() < MIN_POSE_POINTS:
                continue
            moving = (centre[:, mine][:, 0] - centre[0, mine][0]).norm(dim=-1).max() > 0.01
            for t in range(T):
                seen = mine & vis[t]
                share = seen.sum() / mine.sum()
                truth = centre[t, mine][0]
                at = {"arrows": tips[t, mine].mean(0), "all (guess)": guess[t, mine].mean(0)}
                if seen.any():
                    at.update({"seen (guess)": guess[t, seen].mean(0), "seen (truth)": true[t, seen].mean(0)})
                label = bins[0] if share == 0 else bins[1] if share < 0.5 else bins[2]
                for name, c in at.items():
                    for kind in ("all", "moving") if moving else ("all",):
                        errs.setdefault((kind, label, name), []).append(100 * (c - truth).norm().item())

    print("\nObject centre error (cm, median over object-frames) by how much of the object is seen:")
    print(f"  {'objects':<8} {'seen':<14} {'count':>6}" + "".join(f"{n:>14}" for n in names))
    for kind in ("all", "moving"):
        for label in bins:
            e = errs.get((kind, label, "arrows"), [])
            cells = [torch.tensor(errs[kind, label, n]).median().item() if (kind, label, n) in errs else float("nan")
                     for n in names]
            print(f"  {kind:<8} {label:<14} {len(e):>6}" + "".join(f"{c:>14.1f}" if c == c else f"{'-':>14}"
                                                                  for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
