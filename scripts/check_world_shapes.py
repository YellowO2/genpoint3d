"""Do the true shapes sit where our stored points are? Each object's 3D model, placed by Kubric's pose and
brought into a cached clip's frame (the first frame's camera), against the clip's own tracks on that object:
how far each track is from its object's shape, in every frame. Kubric's physics shape is a coarse one, so a
few centimetres are its own; a wrong placing would be tens. No model.

  python scripts/check_world_shapes.py --cache CACHE --objects OBJECTS --assets DIR --clips 40
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from genpoint3d.CachedClip import CachedClip
from genpoint3d.tools.kubric_models import FLIP, shape, turns

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--objects", required=True)
p.add_argument("--assets", required=True)
p.add_argument("--clips", type=int, default=40)
args = p.parse_args()


def off_walls(p: torch.Tensor, wall: torch.Tensor) -> torch.Tensor:
    """(P,) how far each place `p` (P, 3) is from the nearest of the walls (F, 3, 3), each a triangle."""
    a, b, c = wall[:, 0], wall[:, 1], wall[:, 2]
    out = []
    for q in p.split(1000):
        r = q[:, None] - a[None]                                                           # (P, F, 3)
        e0, e1 = (b - a)[None], (c - a)[None]
        d00, d01, d11, d20, d21 = (e0 * e0).sum(-1), (e0 * e1).sum(-1), (e1 * e1).sum(-1), (r * e0).sum(-1), (r * e1).sum(-1)
        den = (d00 * d11 - d01 * d01).clamp_min(1e-18)
        v, w = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den               # where it falls on the wall's flat
        normal = torch.nn.functional.normalize(torch.linalg.cross(b - a, c - a), dim=-1)
        flat = (r * normal[None]).sum(-1).abs()
        edge = torch.full_like(flat, float("inf"))
        for s, e in ((a, b), (b, c), (c, a)):
            along = (((q[:, None] - s[None]) * (e - s)[None]).sum(-1) / ((e - s) ** 2).sum(-1).clamp_min(1e-18)[None]).clamp(0, 1)
            edge = torch.minimum(edge, (q[:, None] - s[None] - along[..., None] * (e - s)[None]).norm(dim=-1))
        out.append(torch.where((v >= 0) & (w >= 0) & (v + w <= 1), flat, edge).min(1).values)
    return torch.cat(out)


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
    turn0, at0 = turns(o["camera_quaternions"][0].astype(np.float64)), o["camera_positions"][0].astype(np.float64)
    traj, here = clip.traj.double(), []
    for k, name in enumerate(o["asset"].tolist()):
        mine = clip.segment == k + 1
        if not mine.any():
            continue
        corners, walls = shape(name, Path(args.assets).expanduser(), float(o["scale"][k]))
        turn = turns(o["quaternions"][k].astype(np.float64))                                         # (T, 3, 3) the object's to the world's
        world = (traj[:, mine].numpy() @ FLIP) @ turn0.T + at0                                            # (T, n, 3) the tracks, out of the first frame's camera
        local = np.einsum("tji,tnj->tni", turn, world - o["positions"][k][:, None])                       # in the object's own frame, where its shape is
        d = off_walls(torch.from_numpy(local).flatten(0, 1), torch.from_numpy(corners[np.array(walls)])).view(local.shape[:2])   # (T, n)
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
