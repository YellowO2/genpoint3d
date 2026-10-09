"""
Can the cached features find a moving point at all, with no model?

For every query point, take its frame-0 feature (the "ID card"), find the patch
in each later frame that looks most like it, and call that patch's 3D position
the prediction. Nothing is trained. It answers one question about the INPUTS:

  matcher beats "nothing moves" on moving points
      -> the features do say where things went; the model is failing to use it.
  matcher does not
      -> the features cannot place the point even roughly, and no model reading
         them could.

Four rows are scored on the same clips and points:

  static        the point never leaves its start. The baseline.
  match         the single most similar patch.
  match-soft    the same, blended with its 8 grid neighbours by similarity, so
                the answer is not stuck on patch centres.
  oracle patch  the patch whose position is truly nearest the point. Not a
                method -- the ceiling for anything that picks one patch, i.e.
                what the grid's coarseness alone costs.

Scored without the global rescale and at the benchmark's 256 px thresholds. A
patch is 16 px at 384 px input (8 px at 768), so read the loose thresholds.

Run:  python scripts/match_baseline.py --cache ~/scratch/cache/kubric --n 100
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from genpoint3d.data.cache import CachedClip
from genpoint3d.eval.metrics import THRESHOLDS, motion_px, tapvid3d_metrics
from genpoint3d.models.encoder import VisualEncoder
from train import _resize_intrinsics, split


@torch.no_grad()
def predictions(clip: CachedClip, dev) -> dict[str, torch.Tensor]:
    traj = clip.traj.to(dev).float()                              # (T, N, 3) metres
    pxyz = clip.patch_xyz.to(dev).float()                         # (T, P, 3) metres
    ctx = F.normalize(clip.context.to(dev).float(), dim=-1)       # (T, P, F)
    T, P, _ = pxyz.shape
    g = math.isqrt(P)
    # The card the model is given, not the cached one: that was read up to half
    # a patch off, and `ClipDataset` re-reads it from frame 0's grid like this.
    f0 = clip.context[0].to(dev).float()                          # (P, F) row-major
    idc = F.normalize(VisualEncoder.sample_at(
        f0.T.reshape(1, -1, g, g), clip.query_uv[None].to(dev).float(), clip.hw,
    )[0], dim=-1)                                                 # (N, F)
    N = idc.shape[0]

    sim = torch.einsum("nf,tpf->tnp", idc, ctx)                   # (T, N, P)
    pick = lambda idx: pxyz.gather(1, idx[..., None].expand(-1, -1, 3))

    best = sim.argmax(-1)                                         # (T, N)
    hard = pick(best)

    # The 3x3 block of grid cells around the best match, clamped at the border.
    r, c = best // g, best % g
    off = torch.tensor([-1, 0, 1], device=dev)
    rr = (r[..., None, None] + off[:, None]).clamp(0, g - 1)      # (T, N, 3, 1)
    cc = (c[..., None, None] + off[None, :]).clamp(0, g - 1)      # (T, N, 1, 3)
    nb = (rr * g + cc).reshape(T, N, 9)
    w = (sim.gather(-1, nb) / 0.05).softmax(-1)                   # (T, N, 9)
    nb_xyz = pxyz.gather(1, nb.reshape(T, N * 9)[..., None].expand(-1, -1, 3)).reshape(T, N, 9, 3)
    soft = (w[..., None] * nb_xyz).sum(-2)

    oracle = pick(torch.cdist(traj, pxyz).argmin(-1))
    static = traj[:1].expand_as(traj)

    out = {"static": static, "match": hard, "match-soft": soft, "oracle patch": oracle}
    # Frame 0 is given to the model too (anchor_frame0), so it is given here.
    return {k: torch.cat([traj[:1], v[1:]]) for k, v in out.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--n", type=int, default=100, help="val clips to score")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-motion-px", type=float, default=4.0)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_clips, _, _ = split(args.cache, args.val_frac, args.seed)
    clips = val_clips[: args.n]
    keys = [f"pts_within_{t}" for t in THRESHOLDS] + ["average_pts_within_thresh"]
    sums = {}      # (subset, method) -> [sum per key], count
    moving_frac = 0.0

    for path in clips:
        clip = CachedClip.from_dict(torch.load(path, weights_only=False))
        gt = clip.traj.to(dev).float()[None]
        vis = clip.visibility.to(dev)[None]
        K = _resize_intrinsics(clip.intrinsics, clip.hw).to(dev).float()[None]
        E = clip.extrinsics.to(dev).float()[None]
        moving = motion_px(gt, vis, K, E) > args.min_motion_px        # (1, N)
        moving_frac += moving.float().mean().item()
        preds = predictions(clip, dev)
        for subset, mask in (("all", vis), ("moving", vis & moving[:, None])):
            if not mask.any():
                continue
            for name, pred in preds.items():
                m = tapvid3d_metrics(pred[None], gt, mask, K, E, scaling="none")
                s = sums.setdefault((subset, name), [[0.0] * len(keys), 0])
                s[0] = [a + m[k] for a, k in zip(s[0], keys)]
                s[1] += 1

    grid = int(CachedClip.from_dict(torch.load(clips[0], weights_only=False)).patch_xyz.shape[1] ** 0.5)
    print(f"\n{len(clips)} val clips from {args.cache} | {grid}x{grid} patch grid"
          f" | {moving_frac / len(clips):.0%} of points move > {args.min_motion_px:g}px\n")
    for subset in ("all", "moving"):
        print(f"--- {subset} points ---")
        print(f"{'':14}" + "".join(f"{'<' + str(t) + 'px':>8}" for t in THRESHOLDS) + f"{'APD':>8}")
        for name in ("static", "match", "match-soft", "oracle patch"):
            tot, n = sums[(subset, name)]
            print(f"{name:14}" + "".join(f"{v / n:>8.3f}" for v in tot))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
