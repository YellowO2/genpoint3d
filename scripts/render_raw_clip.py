"""A raw Kubric clip as a GIF with its true tracks drawn on, each object in its
own colour (the background grey), a point drawn only where it is visible:
to look over generated clips and their object labels before training on them.

Run:  python scripts/render_raw_clip.py CLIP_DIR [CLIP_DIR ...] --out DIR
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("clips", nargs="+")
p.add_argument("--out", required=True)
p.add_argument("--scale", type=int, default=2)
p.add_argument("--fps", type=int, default=12)
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
palette = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]

for clip in map(Path, args.clips):
    d = np.load(clip / f"{clip.name}.npy", allow_pickle=True).item()
    uv, occ, seg = d["coords"], d["occluded"], d["segment"]
    moving = np.linalg.norm(d["traj_3d"] - d["traj_3d"][:, :1], axis=-1).max(1) > 0.01
    frames = []
    for t in range(occ.shape[1]):
        im = Image.open(clip / "frames" / f"{t:03d}.png").convert("RGB")
        im = im.resize((im.width * args.scale, im.height * args.scale), Image.NEAREST)
        draw = ImageDraw.Draw(im)
        for n in np.nonzero(~occ[:, t])[0]:
            x, y = uv[n, t] * args.scale
            colour = (150, 150, 150) if seg[n] == 0 else palette[(seg[n] - 1) % len(palette)]
            r = 1 if seg[n] == 0 else 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)
        draw.text((4, 4), f"{clip.name} frame {t}", fill=(255, 255, 255))
        frames.append(im)
    path = out / f"{clip.name}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // args.fps, loop=0)
    counts = np.bincount(seg)
    print(f"{path}: {occ.shape[1]} frames, {len(seg)} points, objects {len(counts) - 1}, "
          f"points per object {counts.tolist()}, moving {moving.mean():.3f}, "
          f"moving and seen in frame 0 {(moving & ~occ[:, 0]).sum()}")
