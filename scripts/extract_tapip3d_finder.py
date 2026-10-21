"""Save TAPIP3D's trained CNN and 3D-neighbourhood comparison as our finder's weights.

Its checkpoint pickles its config, so this runs in TAPIP3D's own environment:

  source ~/scratch/venvs/tapip3d/bin/activate
  python scripts/extract_tapip3d_finder.py --ckpt <tapip3d_final.pth> --out <tapip3d_finder.pt>

The result loads with `--fine-init` into a finder built with `--fine-xyz 1
--fine-neigh 32` (`FineFinder` keeps TAPIP3D's names for exactly this).
"""

import argparse

import torch

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["weight"]
out = {}
for k, v in sd.items():
    if k.startswith("encoder.backbone."):
        out["fnet." + k[len("encoder.backbone."):]] = v
    elif k.startswith("corr_processor."):
        out[k[len("corr_processor."):]] = v
print(f"{sum(k.startswith('fnet.') for k in out)} CNN and "
      f"{sum(not k.startswith('fnet.') for k in out)} comparison tensors")
torch.save(out, args.out)
