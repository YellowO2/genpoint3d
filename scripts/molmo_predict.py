"""MolmoMotion's forecast for each clip that scripts/molmo_export.py wrote.

Runs in MolmoMotion's own environment (github.com/allenai/molmo-motion); it
reads only its inputs, and writes one .npz of `future_3d` (8, F, 3) per clip,
in the camera of the last seen frame, metres.

  python scripts/molmo_predict.py --model <MolmoMotion-4B-H3-F30 dir> \\
      --in molmo/in --out molmo/pred [--caption "..."]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from molmo_motion import MolmoMotion, MolmoMotionProcessor

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out", required=True)
p.add_argument("--caption", default="The objects fly through the air, fall and collide")
p.add_argument("--horizon", type=int, default=30)
args = p.parse_args()

processor = MolmoMotionProcessor.from_pretrained(args.model)
model = MolmoMotion.from_pretrained(args.model)
model._internal = model._internal.to(torch.bfloat16).cuda()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for f in sorted(Path(args.inp).glob("*.npz")):
    if (out / f.name).exists():
        continue
    d = np.load(f)
    start = time.time()
    inputs = processor(history_frames=[Image.fromarray(x) for x in d["frames"]],
                       points_2d_at_t0=torch.tensor(d["points_2d"]),
                       points_3d_history=torch.tensor(d["history_3d"]),
                       action=args.caption, future_horizon=args.horizon)
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model.predict_trajectory(**inputs)
    # No `<tracks>` block in its answer comes back as zeros: kept, and marked.
    parsed = bool(pred.future_3d.abs().sum() > 0)
    np.savez(out / f.name, future_3d=pred.future_3d.float().cpu().numpy(), parsed=parsed)
    print(f"  {f.stem}  {tuple(pred.future_3d.shape)}  parsed {parsed}  {time.time() - start:.0f}s",
          flush=True)
