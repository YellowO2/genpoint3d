"""The motion-state head checked: how far the velocity and acceleration it
says for each point and frame are from the truth's, beside the same taken from
the model's own final guess (its frame-to-frame differences, which the model
is given anyway), on seen moving point-frames; then the tracks with what it
says fed back, against "unknown" fed back instead. A head no better than the
guess's differences adds nothing; tracks better without it say it misleads.
Every frame keeps its image.

Run:  python scripts/state_check.py --ckpt outputs/RUN/best.pt --cache DIR --frames-cache FRAMES
"""

import argparse

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.score.Run import Run
from genpoint3d.train.tools.batch import fine_inputs, to_device, to_metres, visual_mask
from genpoint3d.train.tools.metrics import clip_mean, motion_px, tapvid3d_metrics
from genpoint3d.train.tools.regress import Sweeps, _state_target


def _off(err: torch.Tensor, true: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(B,) the mean error over `m` (B, T, N) as a share of the truth's mean size."""
    e = (err.norm(dim=-1) * m).flatten(1).sum(1)
    t = (true.norm(dim=-1) * m).flatten(1).sum(1)
    return torch.where(m.flatten(1).any(1), e / t.clamp_min(1e-8), torch.full_like(e, float("nan")))


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--frames-cache", required=True)
    p.add_argument("--points", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run.load(args.ckpt, device)
    model, s = run.model, run.settings
    ds = run.dataset(args.cache, args.frames_cache, points=args.points)
    amp = device.type == "cuda"
    print(f"{len(ds)} clips | step {run.step} of {args.ckpt}", flush=True)
    said_forward = model.forward
    unknown = lambda *a, **k: said_forward(*a, **{**k, "state_known": torch.zeros_like(k["state_known"])})

    rows = {}
    for batch in DataLoader(ds, batch_size=1, shuffle=False, num_workers=4):
        b = to_device(batch, device)
        traj, vis, q = b["traj"], b["visibility"].bool(), b["query_t"]
        cond = dict(context=b["context"], visual_mask=visual_mask(b, None), id_card=b["id_card"],
                    patch_xyz=b["patch_xyz"], **fine_inputs(model, b, amp))
        gt = to_metres(traj, b)
        moving = (motion_px(gt, vis, b["intrinsics_256"], b["extrinsics"], q) > 4)[:, None]
        for fed in ("said", "unknown"):
            model.forward = said_forward if fed == "said" else unknown
            sweeps = Sweeps(model, b["anchor"], traj.shape[1], s.refine_iters, s.schedule, s.window, s.overlap, cond)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in sweeps:
                    pass
            guess = sweeps.whole.float()
            apd = tapvid3d_metrics(to_metres(guess, b), gt, vis & moving, b["intrinsics_256"], b["extrinsics"],
                                   scaling="none", per_clip=True)["average_pts_within_thresh"].cpu()
            rows.setdefault(("apd", fed), []).append(apd)
            if fed == "said":
                true, ok = _state_target(traj.float(), vis)
                from_guess, _ = _state_target(guess, vis)
                m = (ok & moving).float()
                for k, sl in (("velocity", slice(0, 3)), ("acceleration", slice(3, 6))):
                    rows.setdefault((k, "head"), []).append(
                        _off(sweeps.state[..., sl] - true[..., sl], true[..., sl], m).cpu())
                    rows.setdefault((k, "guess"), []).append(
                        _off(from_guess[..., sl] - true[..., sl], true[..., sl], m).cpu())
        model.forward = said_forward

    print("\nSeen moving point-frames, error as a share of the true size (lower is better):")
    print(f"  {'':<14} {'head':>7} {'from the guess':>15}")
    for k in ("velocity", "acceleration"):
        print(f"  {k:<14} {clip_mean(rows[k, 'head']):>7.3f} {clip_mean(rows[k, 'guess']):>15.3f}")
    print("\nAPD (no rescale), seen moving, with the head's state fed back and with \"unknown\" instead:")
    print(f"  fed back {clip_mean(rows['apd', 'said']):.4f}   unknown {clip_mean(rows['apd', 'unknown']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
