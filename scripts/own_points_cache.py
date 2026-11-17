"""A copy of a feature cache with each clip's own points (scripts/movi_points.py)
in place of the ones it came with. All that does not depend on which points
there are is kept (the DINOv3 tokens, the cameras, the scale); a point's look is
read from the cached tokens of its query frame. A clip with no points file is
left out; val.txt goes along.

Run:  python scripts/own_points_cache.py --src CACHE --points DIR --dst CACHE
"""

import argparse
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
import torch

from genpoint3d.tools.feature_lookup import id_cards

FLIP = np.diag([1.0, -1.0, -1.0])     # Blender's camera axes to OpenCV's


def own(job) -> tuple[str, int, float]:
    src, points, dst = job
    clip, o = torch.load(src, weights_only=False), np.load(points)
    e0 = np.linalg.inv(o["matrix_world"][0].astype(np.float64))
    to_cam0 = lambda x: torch.from_numpy((x.astype(np.float64) @ e0[:3, :3].T + e0[:3, 3]) @ FLIP.T).transpose(0, 1).float()
    traj, q = to_cam0(o["traj_3d"]), torch.from_numpy(o["queries"][:, 0]).long()         # (T, N, 3), (N,)
    T, N = traj.shape[:2]
    n = torch.arange(N)
    P = torch.cat([traj, torch.ones(T, N, 1)], -1)
    pix = torch.einsum("tij,tnj->tni", clip["intrinsics"].float(),
                       torch.einsum("tij,tnj->tni", clip["extrinsics"].float()[:, :3, :], P))
    uv = pix[..., :2] / pix[..., 2:].clamp_min(1e-6)                                      # the clip's own cameras
    # The clip's cameras must put each point on the pixel MOVi-F's put it on: the same video, the same camera.
    theirs = torch.from_numpy(o["coords"]).transpose(0, 1) * (clip["hw"][1] / 256)
    seen = torch.from_numpy(~o["occluded"]).T                                              # (T, N)
    off = float((uv - theirs)[seen].norm(dim=-1).median())
    clip.update(traj_metric=traj, anchor_metric=traj[q, n], visibility=seen, query_uv=uv[q, n], query_t=q,
                segment=torch.from_numpy(o["segment"]).long(), centre_metric=to_cam0(o["centre_3d"]), points=N)
    clip["id_card"] = id_cards(clip["context"], q, clip["query_uv"], clip["hw"]).half()
    tmp = dst / f".{src.stem}.tmp"
    torch.save(clip, tmp)
    tmp.rename(dst / src.name)
    return src.stem, N, off


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--points", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    src, points, dst = (Path(x).expanduser() for x in (args.src, args.points, args.dst))
    dst.mkdir(parents=True, exist_ok=True)
    if (src / "val.txt").exists():
        shutil.copy(src / "val.txt", dst / "val.txt")
    clips = sorted(src.glob("*.pt"))[: args.limit or None]
    todo = [(c, points / f"{c.stem}.npz", dst) for c in clips if (points / f"{c.stem}.npz").exists() and not (dst / c.name).exists()]
    print(f"{len(clips)} clips, {sum((points / f'{c.stem}.npz').exists() for c in clips)} with their own points, {len(todo)} to do", flush=True)
    counts, offs = [], []
    with mp.Pool(args.workers) as pool:
        for name, n, off in pool.imap_unordered(own, todo):
            counts.append(n)
            offs.append(off)
            if off > 1:
                print(f"  {name}: its cameras put the points {off:.2f} px from MOVi-F's", flush=True)
    if counts:
        print(f"points a clip: median {int(np.median(counts))}, least {min(counts)}, most {max(counts)}; "
              f"the clip's cameras against MOVi-F's, px: median {np.median(offs):.3f}, worst {max(offs):.3f}", flush=True)
