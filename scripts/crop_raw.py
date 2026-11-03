"""Cut rendered Kubric clips (scripts/kubric_movi.pbs) to their first F frames,
as a raw root our cache builder reads unchanged (genpoint3d/prepare), and leave
out clips with little moving in view.

Each kept clip gets `<dst>/<id>/<id>.npy` with every per-frame array cut to F
frames, and `<dst>/<id>/frames/` linking to the first F frames and depths. A
clip is kept when at least --min-moving of its tracks are on a moving object
(more than 1 cm from their first position at some frame) and seen in frame 0,
the points our cache keeps as queries.

  python scripts/crop_raw.py --src ~/scratch/kubric_movi_256 --dst ~/scratch/kubric_movi48 --frames 48
"""

import argparse
import os
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--dst", required=True)
p.add_argument("--frames", type=int, default=48)
p.add_argument("--min-moving", type=int, default=150)
args = p.parse_args()
F = args.frames
src, dst = Path(args.src), Path(args.dst)
kept = dropped = 0
for clip in sorted(d for d in src.iterdir() if (d / f"{d.name}.npy").exists()):
    d = np.load(clip / f"{clip.name}.npy", allow_pickle=True).item()
    T = d["intrinsics"].shape[0]
    xyz, occ = d["traj_3d"][:, :F], d["occluded"][:, :F]
    moving = np.linalg.norm(xyz - xyz[:, :1], axis=-1).max(1) > 0.01
    if T < F or int((moving & ~occ[:, 0]).sum()) < args.min_moving:
        dropped += 1
        continue
    out = dst / clip.name
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cut = {k: (v[:, :F] if k in ("coords", "occluded", "traj_3d") else
               v[:F] if k in ("intrinsics", "matrix_world") else v) for k, v in d.items()}
    np.save(out / f"{clip.name}.npy", cut)
    for t in range(F):
        for name in (f"{t:03d}.png", f"{t:03d}_depth.png"):
            link = out / "frames" / name
            if not link.exists():
                os.symlink(clip / "frames" / name, link)
    kept += 1
print(f"{kept} clips kept, {dropped} left out (shorter than {F} frames, or under "
      f"{args.min_moving} moving tracks seen in frame 0)")
