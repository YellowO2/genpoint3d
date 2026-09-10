"""
Measure `MOTION_SCALE` -- the dataset-level constant in
`genpoint3d/data/transform.py`.

The model denoises a *residual* (offset from each point's frame-0 position).
Flow matching mixes that residual with `x0 ~ N(0, I)`, so the residual must
have roughly unit variance or the interpolant is nearly pure noise and the
model learns only to output `-x0`.

This measures how big residuals actually are, in scene-normalised units,
pooled over the whole training set. One global number -- not per-clip -- so no
individual sample's future leaks into its own normalisation.

Run:  .venv/bin/python scripts/calibrate_motion_scale.py [data_root]
Then paste the printed value into `MOTION_SCALE`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from genpoint3d.data.kubric import KubricSequenceDataset
from genpoint3d.data.transform import MOTION_SCALE, transform


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "data/kubric_test"
    ds = KubricSequenceDataset(root, num_query_points=256)

    residuals = []
    for i in range(len(ds)):
        sample = ds[i]
        T = sample.frames.shape[0]
        # motion_scale=1.0 -> `traj` stays in raw scene-normalised units,
        # which is exactly the quantity we are trying to measure.
        inputs = transform(sample, num_context_frames=T, motion_scale=1.0)
        residuals.append(inputs.traj[inputs.visibility])
        print(f"  {sample.seq_id}: residual std {inputs.traj.std():.4f}")

    pooled = torch.cat(residuals)
    scale = pooled.std().item()

    print(f"\npooled over {len(ds)} clip(s), {pooled.shape[0]} visible point-frames")
    print(f"  MOTION_SCALE = {scale:.4f}      (current: {MOTION_SCALE})")
    print(f"  -> target std after scaling: {pooled.std().item() / scale:.3f}")
    print("\nPaste that into MOTION_SCALE in genpoint3d/data/transform.py.")
    print("NOTE: 2 clips is far too few for a real constant -- recalibrate")
    print("      once the full training set is downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
