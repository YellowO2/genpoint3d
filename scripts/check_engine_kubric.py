"""Our engine's step (`tools/engine.py`) against Kubric's own physics (PyBullet): MOVi-F's true object poses.
For every object and frame, the pose a frame on from the two true poses before it, as our step carries it with
nothing touching it, against the true one: how far the centre is off, and the turn. Where an object is flying
the step should be right; where it is not, it rests or is being pushed, the model's part.

Run (a batch job; needs torch and numpy):  python scripts/check_engine_kubric.py --objects ~/scratch/movi_objects --clips 200
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from genpoint3d.tools.traj_scale import GRAVITY, KUBRIC_FPS
from genpoint3d.train.tools.engine import Body, step
from genpoint3d.train.tools.kabsch import turn_vector

p = argparse.ArgumentParser()
p.add_argument("--objects", required=True, help="scripts/movi_objects.py's output")
p.add_argument("--clips", type=int, default=200)
args = p.parse_args()


def turns(q: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) from quaternions (..., 4) as Kubric gives them, w first."""
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


fall = torch.tensor([0.0, 0.0, -GRAVITY / KUBRIC_FPS ** 2], dtype=torch.float64)      # Kubric's world: z up
off, deg, moved, turned = [], [], [], []
for f in sorted(Path(args.objects).expanduser().glob("*.npz"))[: args.clips]:
    d = np.load(f)
    c, R = torch.from_numpy(d["positions"]).double(), turns(torch.from_numpy(d["quaternions"]).double())   # (K, T, ...)
    for t in range(2, c.shape[1]):
        spin = turn_vector(R[:, t - 1] @ R[:, t - 2].transpose(-1, -2))
        went = step(Body(c[:, t - 1], R[:, t - 1], c[:, t - 1] - c[:, t - 2], spin), fall, torch.ones(len(c), dtype=torch.float64))
        off.append((went.centre - c[:, t]).norm(dim=-1))
        deg.append(torch.rad2deg(turn_vector(went.turn @ R[:, t].transpose(-1, -2)).norm(dim=-1)))
        moved.append((c[:, t] - c[:, t - 1]).norm(dim=-1))
        turned.append(torch.rad2deg(turn_vector(R[:, t] @ R[:, t - 1].transpose(-1, -2)).norm(dim=-1)))
off, deg, moved, turned = (torch.cat(x) for x in (off, deg, moved, turned))
cm = off * 100
flying = cm < 0.5
still = (moved < 0.001) & (turned < 0.1)
print(f"{len(off)} object-frames of {args.clips} clips")
print(f"  flying (our step within 0.5 cm of Kubric's centre): {flying.float().mean():.1%}; "
      f"their centre {cm[flying].median():.3f} cm off (median), their turn {deg[flying].median():.2f} deg off "
      f"(turning {turned[flying].median():.1f} deg a frame), {(deg[flying] > 1).float().mean():.1%} over 1 deg")
print(f"  at rest (under 1 mm and 0.1 deg a frame): {still.float().mean():.1%}")
print(f"  the rest, pushed or rolling: {(~flying & ~still).float().mean():.1%}; our step {cm[~flying & ~still].median():.1f} cm off there (median)")
