"""Do the true shapes sit where our stored points are? Each object's 3D model, placed by Kubric's pose and
brought into a cached clip's frame (the first frame's camera), against the clip's own tracks on that object:
how far each track is from its object's shape, in every frame. Kubric's physics shape is a coarse one, so a
few centimetres are its own; a wrong placing would be tens. No model.

  python scripts/check_world_shapes.py --cache CACHE --objects OBJECTS --assets DIR --clips 40
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kubric_tracks import quat_to_rot  # noqa: E402

from genpoint3d.CachedClip import CachedClip  # noqa: E402
from genpoint3d.tools.kubric_models import shape  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--objects", required=True)
p.add_argument("--assets", required=True)
p.add_argument("--clips", type=int, default=40)
args = p.parse_args()

FLIP = np.diag([1.0, -1.0, -1.0])  # Blender's axes to OpenCV's
ON = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [.5, 0, .5], [0, .5, .5], [1 / 3, 1 / 3, 1 / 3]])  # places on a wall

files = sorted(Path(args.cache).expanduser().glob("*.pt"))
files = files[:: max(len(files) // args.clips, 1)][: args.clips]
off, share, clips = [], [], 0
for f in files:
    o = Path(args.objects).expanduser() / f"{f.stem}.npz"
    clip = CachedClip.from_dict(torch.load(f, weights_only=False))
    if not o.exists() or clip.segment is None:
        continue
    o = np.load(o)
    if "asset" not in o:
        continue
    turn0, at0 = quat_to_rot(o["camera_quaternions"][:1].astype(np.float64))[0], o["camera_positions"][0].astype(np.float64)
    traj, here = clip.traj.double(), []
    for k, name in enumerate(o["asset"].tolist()):
        mine = clip.segment == k + 1
        if not mine.any():
            continue
        corners, walls = shape(name, Path(args.assets).expanduser(), float(o["scale"][k]))
        surface = np.einsum("pc,fcx->fpx", ON, corners[np.array(walls)]).reshape(-1, 3)                  # the walls' corners, edges and middles
        turn = quat_to_rot(o["quaternions"][k].astype(np.float64))                                         # (T, 3, 3) the object's to the world's
        world = np.einsum("tij,sj->tsi", turn, surface) + o["positions"][k][:, None]                      # (T, S, 3)
        cam0 = torch.from_numpy(((world - at0) @ turn0) @ FLIP.T)                                         # the first frame's camera, as the cache has it
        d = torch.cdist(traj[:, mine], cam0).min(-1).values                                               # (T, n)
        here.append(d.flatten())
        share.append((d / float(o["scale"][k])).flatten())
    if here:
        clips += 1
        off.append(torch.cat(here))
        print(f"clip {f.stem}: a track from its object's shape, cm: median {float(off[-1].median()) * 100:.1f}, "
              f"90% under {float(off[-1].quantile(0.9)) * 100:.1f}", flush=True)
off, share = torch.cat(off), torch.cat(share)
print(f"=== {clips} clips, {len(off)} track-frames: a track from its object's shape, cm: median {float(off.median()) * 100:.1f}, "
      f"90% under {float(off.quantile(0.9)) * 100:.1f}, 99% under {float(off.quantile(0.99)) * 100:.1f}; "
      f"as a share of the object's longest side: median {float(share.median()):.1%}, 90% under {float(share.quantile(0.9)):.1%} ===", flush=True)
