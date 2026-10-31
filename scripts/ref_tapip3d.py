"""
Run a published tracker, TAPIP3D, on OUR validation clips and score it with OUR
scorer.

Two things come out of it that no run of our own model can give:

  - a number from a tracker that is known to work, on exactly our clips and
    query points. If it scores well, our data path and our scoring are sound.
  - what "tracking the moving points" looks like under the score we trust
    (moving points only, no global rescale), next to "nothing moves".

Runs inside TAPIP3D's own environment, with their repo on the path; only our
cache format, raw loader and metric are imported from this repo.

Their model works in a world frame. Our tracks are stored in the frame-0
camera, so that camera IS the world here: the cached extrinsics (cam_0 -> cam_t)
are passed as world -> camera, and each query starts where our model's does,
at the clip's `anchor`: the true 3D start on Kubric, the query pixel at Depth
Anything 3's depth on TAPVid-3D.

The frames and depth are read at full size from the raw clips, as their own
demo does. A TAPVid-3D half-clip (`scripts/tapvid3d_cache.py`) is
`<clip>_q<q>_<f|b>`: frames q, q+1, ... of `<clip>.npz`, or q, q-1, ... for
the backward half, with the depth `<clip>.da3.npy` in the same order.

Run (see scripts/ref_tapip3d.pbs and scripts/tapvid3d_ref.pbs):
  python scripts/ref_tapip3d.py --ref-repo ~/scratch/refs/tapip3d \\
      --clips ~/scratch/ref_eval/val100 --raw ~/scratch/kubric
  python scripts/ref_tapip3d.py --ref-repo ~/scratch/refs/tapip3d \\
      --clips ~/scratch/tapvid3d/pstudio_cache/clips --raw ~/scratch/tapvid3d/pstudio \\
      --raw-kind tapvid3d
"""

import argparse
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

OURS = Path(__file__).resolve().parent.parent


def kubric_video(raw: Path, seq: str) -> tuple[np.ndarray, np.ndarray]:
    """(T, H, W, 3) uint8 frames and (T, H, W) depth in metres of a Kubric clip."""
    from genpoint3d.prepare.KubricSequenceDataset import KubricSequenceDataset
    r = KubricSequenceDataset(str(raw), seq_ids=[seq], num_query_points=None)[0]
    return r.frames, r.depths


@lru_cache(maxsize=1)
def _tapvid3d_clip(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A whole TAPVid-3D clip's frames and DA3 depth; its halves come in a row."""
    return (np.load(path.with_suffix(".npz"))["images"],
            np.load(path.with_suffix(".da3.npy")).astype(np.float32))


def tapvid3d_video(raw: Path, seq: str) -> tuple[np.ndarray, np.ndarray]:
    """The frames and depth of one half-clip, in the order tapvid3d_cache.py
    put them: from its query frame forward to the end, or back to frame 0."""
    name, q, way = seq.rsplit("_", 2)
    frames, depth = _tapvid3d_clip(raw / name)
    q = int(q[1:])
    order = list(range(q, len(frames))) if way == "f" else list(range(q, -1, -1))
    return frames[order], depth[order]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-repo", required=True, help="clone of zbw001/TAPIP3D")
    p.add_argument("--ckpt", default="checkpoints/tapip3d_final.pth")
    p.add_argument("--clips", required=True, help="directory of cached clips to score")
    p.add_argument("--raw", required=True,
                   help="the raw clips: Kubric root, or the TAPVid-3D .npz with their .da3.npy")
    p.add_argument("--raw-kind", default="kubric", choices=["kubric", "tapvid3d"])
    p.add_argument("--out", default=None, help="save predictions here, one .pt per clip")
    p.add_argument("--resolution-factor", type=float, default=1.0,
                   help="1 runs at the 384x512 the model was trained at")
    p.add_argument("--num-iters", type=int, default=6)
    p.add_argument("--support-grid", type=int, default=16)
    p.add_argument("--min-motion-px", type=float, default=4.0)
    p.add_argument("--max-points", type=int, default=0,
                   help="track only the first M query points (0 = all)")
    p.add_argument("--chunk", type=int, default=0,
                   help="track the queries K at a time, so each sees only K-1 "
                        "others (0 = all together). With --support-grid 0 and "
                        "--chunk 1 every point is tracked alone.")
    args = p.parse_args()

    ref = Path(args.ref_repo).expanduser().resolve()
    os.chdir(ref)
    sys.path.insert(0, str(ref))
    sys.path.append(str(OURS))
    try:
        import sophuspy  # noqa: F401  (their data_ops imports it; unused on this path)
    except ImportError:
        sys.modules["sophuspy"] = types.ModuleType("sophuspy")

    from datasets.data_ops import _filter_one_depth
    from utils.inference_utils import inference, load_model, resize_depth_bilinear

    from genpoint3d.train.tools.metrics import THRESHOLDS, motion_px, tapvid3d_metrics

    dev = torch.device("cuda")
    model = load_model(args.ckpt).to(dev)
    res = tuple(int(s * np.sqrt(args.resolution_factor)) for s in model.image_size)
    model.set_image_size(res)
    print(f"TAPIP3D {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
          f" | inference at {res[0]}x{res[1]} | {args.num_iters} iters", flush=True)

    read_video = kubric_video if args.raw_kind == "kubric" else tapvid3d_video
    clips = sorted(Path(args.clips).expanduser().glob("*.pt"))
    out = Path(args.out).expanduser() if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    keys = [f"pts_within_{t}" for t in THRESHOLDS] + ["average_pts_within_thresh"]
    sums, moving_frac, oa = {}, 0.0, 0.0

    for i, path in enumerate(clips, 1):
        c = torch.load(path, weights_only=False)
        seq = c["seq_id"]
        frames, raw_depths = read_video(Path(args.raw).expanduser(), seq)
        H0, W0 = frames.shape[1:3]
        if (H0, W0) != tuple(c["hw"]) or len(frames) != c["traj_metric"].shape[0]:
            raise SystemExit(f"{seq}: raw video {frames.shape[:3]} does not match the clip")

        K = c["intrinsics"].clone().float()                         # (T, 3, 3) native pixels
        K_in = K.clone().numpy()
        K_in[:, 0, :] *= (res[1] - 1) / (W0 - 1)
        K_in[:, 1, :] *= (res[0] - 1) / (H0 - 1)
        video = np.stack([cv2.resize(f, (res[1], res[0]), interpolation=cv2.INTER_LINEAR)
                          for f in frames])
        depths = np.stack([resize_depth_bilinear(d, (res[1], res[0])) for d in raw_depths])
        depths = np.stack([_filter_one_depth(d, 0.08, 15, k) for d, k in zip(depths, K_in)])

        gt = c["traj_metric"].float().to(dev)                              # (T, N, 3) frame-0 camera, metres
        vis = c["visibility"].to(dev)
        start = c["anchor_metric"].float().to(dev)                          # (N, 3) as our model gets it
        if args.max_points:
            gt, vis, start = gt[:, :args.max_points], vis[:, :args.max_points], start[:args.max_points]
        E = c["extrinsics"].float().to(dev)                         # cam_0 -> cam_t
        query = torch.cat([torch.zeros_like(start[:, :1]), start], dim=-1)   # (N, 4): t, x, y, z

        chunk = args.chunk or query.shape[0]
        parts = []
        for q in query.split(chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                coords, visibs = inference(
                    model=model,
                    video=(torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0).to(dev),
                    depths=torch.from_numpy(depths).float().to(dev),
                    intrinsics=torch.from_numpy(K_in).float().to(dev),
                    extrinsics=E, query_point=q,
                    num_iters=args.num_iters, grid_size=args.support_grid,
                )
            # Their inference squeezes its output, so one query comes back
            # without its point axis.
            T_ = coords.shape[0]
            parts.append((coords.float().reshape(T_, -1, 3)[:, : q.shape[0]],
                          visibs.reshape(T_, -1)[:, : q.shape[0]].bool()))
        pred = torch.cat([c_ for c_, _ in parts], dim=1)
        pvis = torch.cat([v_ for _, v_ in parts], dim=1)
        if out:
            torch.save({"seq_id": seq, "coords": pred.cpu(), "visibs": pvis.cpu()}, out / path.name)

        K256 = K.clone()
        K256[..., :2, :] *= 256 / min(c["hw"])
        K256 = K256.to(dev)[None]
        g, v, e = gt[None], vis[None], E[None]
        moving = motion_px(g, v, K256, e) > args.min_motion_px
        moving_frac += moving.float().mean().item()
        oa += (pvis == vis).float().mean().item()
        preds = {"static": g[:, :1].expand_as(g), "TAPIP3D": pred[None]}
        for label, mask, scaling in (("all points, protocol rescale", v, "median"),
                                     ("all points, no rescale", v, "none"),
                                     ("moving points, no rescale", v & moving[:, None], "none")):
            if not mask.any():
                continue
            for name, pr in preds.items():
                m = tapvid3d_metrics(pr, g, mask, K256, e, scaling=scaling)
                s = sums.setdefault((label, name), [[0.0] * len(keys), 0])
                s[0] = [a + m[k] for a, k in zip(s[0], keys)]
                s[1] += 1
        if i % 10 == 0 or i == len(clips):
            print(f"  {i}/{len(clips)}", flush=True)

    print(f"\n{len(clips)} clips | {moving_frac / len(clips):.0%} of points move >"
          f" {args.min_motion_px:g}px | TAPIP3D occlusion accuracy {oa / len(clips):.3f}\n")
    for label in ("all points, protocol rescale", "all points, no rescale", "moving points, no rescale"):
        print(f"--- {label} ---")
        print(f"{'':10}" + "".join(f"{'<' + str(t) + 'px':>8}" for t in THRESHOLDS) + f"{'APD':>8}")
        for name in ("static", "TAPIP3D"):
            tot, n = sums[(label, name)]
            print(f"{name:10}" + "".join(f"{x / n:>8.3f}" for x in tot))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
