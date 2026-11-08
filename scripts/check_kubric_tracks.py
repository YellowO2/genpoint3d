"""Check raw Kubric clips' tracks against what was rendered, independently of
how the tracks were made: at every frame a point is visible, its 3D distance
from the camera should be the rendered depth at its pixel (Kubric's depth is
the distance from the camera, stored as 16 bit over the clip's depth range),
and at its query frame it should be visible. Run on our clips and on
TAPIP3D's for a reference.

Run:  python scripts/check_kubric_tracks.py DIR [DIR ...] --clips 5
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("dirs", nargs="+")
p.add_argument("--clips", type=int, default=5)
args = p.parse_args()

for root in args.dirs:
    clips = sorted(d for d in Path(root).iterdir() if (d / f"{d.name}.npy").exists())[: args.clips]
    print(f"### {root}: {len(clips)} clips")
    print(f"  {'clip':<8} {'T':>3} {'points':>6} {'visible':>8} {'median rel err':>15} {'over 2%':>8} "
          f"{'over 10%':>9} {'query hidden':>13} {'moving':>7}")
    for clip in clips:
        d = np.load(clip / f"{clip.name}.npy", allow_pickle=True).item()
        xyz, occ, uv = d["traj_3d"], d["occluded"], d["coords"]          # (N, T, 3), (N, T), (N, T, 2)
        N, T = occ.shape
        lo, hi = d["depth_range"]
        depth = np.stack([np.asarray(Image.open(clip / "frames" / f"{t:03d}_depth.png"), np.float64)
                          for t in range(T)]) / 65535 * (hi - lo) + lo   # (T, H, W)
        H, W = depth.shape[1:]
        cam = d["matrix_world"][:, :3, 3]                                  # (T, 3)
        dist = np.linalg.norm(xyz - cam[None], axis=-1)                    # (N, T)
        x = np.clip(np.floor(uv[..., 0]).astype(int), 0, W - 1)
        y = np.clip(np.floor(uv[..., 1]).astype(int), 0, H - 1)
        rendered = depth[np.arange(T)[None], y, x]
        vis = ~occ
        rel = np.abs(rendered - dist)[vis] / dist[vis]
        if "queries" in d:
            q = d["queries"][:, 0]
        else:                                                              # TAPIP3D's: the first visible frame
            q = np.argmax(vis, axis=1)
        query_hidden = occ[np.arange(N), q].mean()
        moving = (np.linalg.norm(xyz - xyz[:, :1], axis=-1).max(1) > 0.01).mean()
        print(f"  {clip.name:<8} {T:>3} {N:>6} {vis.mean():>8.3f} {np.median(rel):>15.5f} "
              f"{(rel > 0.02).mean():>8.4f} {(rel > 0.10).mean():>9.4f} {query_hidden:>13.4f} {moving:>7.3f}")
