"""Whether our engine's touch (`engine.step` with a `Touch`) can say what Kubric's own physics does where an
object is on the floor: MOVi-F's true object poses again. For an object-frame that is not flying (our plain step
over 0.5 cm off), the best touch with the floor is searched for: where on the floor under the object, how far
the floor holds it and grips it there, and how wide the object's mass is spread (its shape is not in the data). Then the pose a
frame on with that touch, against the true one. Near nothing left: the engine can say Kubric's touches and only
the settings are the model's to learn. Much left: the engine's touch is missing something no training gives.

Run (a batch job):  python scripts/check_touch_kubric.py --objects ~/scratch/movi_objects --clips 100 --frames 400
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from genpoint3d.tools.traj_scale import GRAVITY, KUBRIC_FPS
from genpoint3d.train.tools.engine import Body, Touch, step
from genpoint3d.train.tools.kabsch import turn_vector

p = argparse.ArgumentParser()
p.add_argument("--objects", required=True, help="scripts/movi_objects.py's output")
p.add_argument("--clips", type=int, default=100)
p.add_argument("--frames", type=int, default=400, help="object-frames searched, drawn at random")
args = p.parse_args()
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def turns(q: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) from quaternions (..., 4) as Kubric gives them, w first."""
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


fall = torch.tensor([0.0, 0.0, -GRAVITY / KUBRIC_FPS ** 2])              # Kubric's world: z up, the floor at z = 0
before, now, after = [], [], []                                          # each (centre, turn) of an object in three frames
for f in sorted(Path(args.objects).expanduser().glob("*.npz"))[: args.clips]:
    d = np.load(f)
    c, R = torch.from_numpy(d["positions"]).double(), turns(torch.from_numpy(d["quaternions"]).double())
    for t in range(2, c.shape[1]):
        before.append((c[:, t - 2], R[:, t - 2])), now.append((c[:, t - 1], R[:, t - 1])), after.append((c[:, t], R[:, t]))
cat = lambda x: (torch.cat([a for a, _ in x]), torch.cat([b for _, b in x]))
(c0, R0), (c1, R1), (c2, R2) = cat(before), cat(now), cat(after)
goes = lambda: Body(c1, R1, c1 - c0, turn_vector(R1 @ R0.transpose(-1, -2)))
plain = step(goes(), fall, torch.ones(len(c1)))
off = lambda b, i=slice(None): ((b.centre[i] - c2[i]).norm(dim=-1) * 100,
                                torch.rad2deg(turn_vector(b.turn[i] @ R2[i].transpose(-1, -2)).norm(dim=-1)))
touching = torch.nonzero(off(plain)[0] > 0.5)[:, 0]
pick = touching[torch.randperm(len(touching))[: args.frames]]
M = len(pick)
still = ((c2[pick] - c1[pick]).norm(dim=-1) < 0.001)

# The picked objects, and the floor as body 0: fixed, where it is.
body = Body(torch.cat([torch.zeros(1, 3), c1[pick]]), torch.cat([torch.eye(3)[None], R1[pick]]),
            torch.cat([torch.zeros(1, 3), (c1 - c0)[pick]]),
            torch.cat([torch.zeros(1, 3), turn_vector(R1 @ R0.transpose(-1, -2))[pick]]))
free = torch.cat([torch.zeros(1), torch.ones(M)])
height = c1[pick, 2].clamp_min(0.02)
where, wide = torch.zeros(M, 2, requires_grad=True), torch.zeros(M, requires_grad=True)
hold, grip = torch.full((M,), 2.0, requires_grad=True), torch.full((M,), 2.0, requires_grad=True)
opt = torch.optim.Adam([where, hold, grip, wide], lr=0.05)


def stepped() -> Body:
    """The objects a frame on, each touching the floor at its searched place, held and gripped as searched."""
    at = torch.cat([c1[pick, :2] + where * height[:, None], torch.zeros(M, 1)], -1)
    spread = torch.cat([torch.ones(1), 0.4 * (height * torch.exp(wide)) ** 2])[:, None, None] * torch.eye(3)
    touch = Touch(torch.arange(1, M + 1), torch.zeros(M, dtype=torch.long), at, torch.tensor([0.0, 0.0, 1.0]).expand(M, 3),
                  torch.sigmoid(hold), torch.sigmoid(grip), torch.zeros(M, 3))
    return step(body, fall, free, torch.ones(M + 1), spread, touch)


for i in range(300):
    opt.zero_grad()
    b = stepped()
    points = lambda c, R: c[:, None] + height[:, None, None] * R.transpose(-1, -2)[:, :3]        # three points a height out
    loss = (points(b.centre[1:], b.turn[1:]) - points(c2[pick], R2[pick])).pow(2).sum()
    loss.backward()
    opt.step()
    if i % 50 == 0:
        print(f"  search {i}: {loss.item():.4f}", flush=True)
with torch.no_grad():
    b = stepped()
cm, deg = ((b.centre[1:] - c2[pick]).norm(dim=-1) * 100,
           torch.rad2deg(turn_vector(b.turn[1:] @ R2[pick].transpose(-1, -2)).norm(dim=-1)))
cm0, deg0 = off(plain, pick)
print(f"{M} object-frames that are not flying, of {len(touching)} in {args.clips} clips")
for name, m in (("at rest", still), ("moving", ~still)):
    print(f"  {name} ({int(m.sum())}): with no touch {cm0[m].median():.2f} cm, {deg0[m].median():.2f} deg off (median); "
          f"with the best touch on the floor {cm[m].median():.2f} cm, {deg[m].median():.2f} deg; "
          f"within 0.5 cm and 1 deg: {((cm[m] < 0.5) & (deg[m] < 1)).float().mean():.0%}")
