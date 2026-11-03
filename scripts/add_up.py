"""Add each cached clip's up direction (Kubric's world +z, in its frame-0
camera) to a cache built before the cache held it, from the raw clips'
cameras, as `genpoint3d/prepare/transform.py` now computes it on caching.

  python scripts/add_up.py --cache ~/scratch/cache/kubric_movi48 --raw ~/scratch/kubric_movi48
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from genpoint3d.prepare.KubricSequenceDataset import _AXIS_FLIP
from genpoint3d.prepare.transform import up_in_frame0

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--raw", required=True)
args = p.parse_args()
done = 0
for f in sorted(Path(args.cache).glob("*.pt")):
    d = torch.load(f, weights_only=False)
    seq = d["seq_id"]
    raw = np.load(Path(args.raw) / seq / f"{seq}.npy", allow_pickle=True).item()
    E = np.einsum("ij,tjk->tik", _AXIS_FLIP, np.linalg.inv(raw["matrix_world"]).astype(np.float32))
    d["up"] = up_in_frame0(torch.from_numpy(E))
    torch.save(d, f)
    done += 1
print(f"{done} clips in {args.cache} given their up")
