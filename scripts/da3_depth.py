"""Metric depth for every frame of the TAPVid-3D clips, from Depth Anything 3.

Real video comes with no depth, and the model reads each cell's 3D position
(`--fine-xyz`); TAPVid-3D's trackers all run on a depth model's output.
DA3METRIC-LARGE, one frame at a time (it is a monocular model), turned into
metres as its README says -- `focal * output / 300`, with `focal` the mean of
fx and fy in pixels of the resolution it ran at -- and resized back to the
frame's own size.

Runs in Depth Anything 3's own environment (github.com/ByteDance-Seed/
Depth-Anything-3). Beside each `<clip>.npz`, `<clip>.da3.npy`: (T, H, W)
float16 metres.

  python scripts/da3_depth.py --clips DIR [--model depth-anything/DA3METRIC-LARGE]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from depth_anything_3.api import DepthAnything3

p = argparse.ArgumentParser()
p.add_argument("--clips", required=True)
p.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
args = p.parse_args()

model = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
for clip in sorted(Path(args.clips).glob("*.npz")):
    out = clip.with_suffix(".da3.npy")
    if out.exists():
        continue
    d = np.load(clip, allow_pickle=True)
    frames, (fx, fy, _, _) = d["images"], d["fx_fy_cx_cy"]
    H, W = frames.shape[1:3]
    depth = []
    for f in frames:
        pred = model.inference([f])
        net = torch.from_numpy(pred.depth[0]).float()                 # at the resolution it ran at
        h, w = net.shape
        focal = (fx * w / W + fy * h / H) / 2
        metres = focal * net / 300.0
        depth.append(F.interpolate(metres[None, None], size=(H, W), mode="bilinear",
                                   align_corners=False)[0, 0].numpy())
    np.save(out, np.stack(depth).astype(np.float16))
    print(f"  {clip.name}  depth {np.median(depth):.2f} m median", flush=True)
