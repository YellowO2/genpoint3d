"""Is each clip's answer (`kubric_models.answer`) where the clip is, as training is served it? For a few cached
clips, through the dataset itself: how many corners an object has, how many a frame sees, and how far the
clip's own tracks, which are on the true surface, sit from their object's nearest corner. Corners 8 cells
across an object are about an eighth of it apart, so a track is within about that of one; a wrong placing
would be many times it. No model.

  python scripts/check_answer.py --cache CACHE --frames FRAMES --objects OBJECTS --assets DIR --clips 20
"""

import argparse
from pathlib import Path

import torch

from genpoint3d.train.data.ClipDataset import ClipDataset

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--frames", required=True)
p.add_argument("--objects", required=True)
p.add_argument("--assets", required=True)
p.add_argument("--clips", type=int, default=20)
p.add_argument("--slots", type=int, default=64)
args = p.parse_args()

files = sorted(Path(args.cache).expanduser().glob("*.pt"))
files = files[:: max(len(files) // args.clips, 1)][: args.clips]
data = ClipDataset(files, 10 ** 6, resample=False, frames=Path(args.frames).expanduser(), objects=Path(args.objects).expanduser(),
                   assets=Path(args.assets).expanduser(), slots=args.slots)
off, count, seen, never, bars, level = [], [], [], [], [], []
for i, f in enumerate(files):
    b = data[i]
    mesh, of, shown = b["mesh"], b["mesh_body"], b["mesh_seen"]
    if not len(of):
        print(f"clip {f.stem}: no objects known", flush=True)
        continue
    pos = b["traj"] * b["norm_traj_scale"] + b["norm_offset"]                    # (T, N, 3) the clip's unit, as the answer
    here = []
    for o in of[of > 0].unique().tolist():
        mine, own = of == o, b["segment"] == o
        count.append(int(mine.sum()))
        if own.any():
            size = torch.cdist(mesh[0, mine], mesh[0, mine]).max()
            here.append((torch.cdist(pos[:, own], mesh[:, mine]).min(-1).values / size).flatten())
    flat = of == 0
    if flat.any():
        a0, a1, a2 = mesh[0, flat][0], mesh[0, flat][1], mesh[0, flat][3]
        up = torch.nn.functional.normalize(torch.linalg.cross(a1 - a0, a2 - a0), dim=0)
        ground = pos[:, b["segment"] == 0].reshape(-1, 3)
        level.append(((ground - a0) @ up).abs() * float(b["norm_scale"]))        # metres
    seen.append(shown.float().mean())
    never.append((~shown.any(0)).float().mean())
    bars.append(len(b["mesh_bars"]) / len(of))
    off.append(torch.cat(here))
    print(f"clip {f.stem}: {len(of.unique())} objects, {len(of)} corners; a track from its object's nearest corner, as a share of the "
          f"object's size: median {float(off[-1].median()):.3f}, 90% under {float(off[-1].quantile(0.9)):.3f}; a frame sees "
          f"{float(seen[-1]):.0%} of the corners, {float(never[-1]):.0%} are never seen", flush=True)
off, count = torch.cat(off), torch.tensor(count)
print(f"=== {len(seen)} clips: corners an object, median {int(count.median())}, most {int(count.max())} (cap {args.slots}), "
      f"{float((count == args.slots).float().mean()):.0%} of objects at the cap; bars a corner {sum(bars) / len(bars):.1f}; a track from its "
      f"object's nearest corner: median {float(off.median()):.3f} of the object's size, 90% under {float(off.quantile(0.9)):.3f}, "
      f"99% under {float(off.quantile(0.99)):.3f}; a frame sees {float(torch.stack(seen).mean()):.0%} of the corners, "
      f"{float(torch.stack(never).mean()):.0%} are never seen ===", flush=True)
level = torch.cat(level)
print(f"=== the floor's points from the floor's four corners' plane, cm: median {float(level.median()) * 100:.1f}, 90% under "
      f"{float(level.quantile(0.9)) * 100:.1f} (the floor is flat: a few cm is depth's own) ===", flush=True)
