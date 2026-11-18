"""The fixed parts of a physics engine; what a camera cannot tell it, the model says."""

from dataclasses import dataclass
from typing import Optional

import torch

from genpoint3d.train.tools.kabsch import robust_kabsch, turn_matrix

# Refits of a body's pose, each point counting less the further it sat from the last fit.
REFITS = 3


@dataclass
class Body:
    """Rigid bodies: where each one's centre of gravity is (K, 3), how it is
    turned (K, 3, 3), and its speed and spin (K, 3), a frame."""
    centre: torch.Tensor
    turn: torch.Tensor
    speed: torch.Tensor
    spin: torch.Tensor

    def points(self, shape: torch.Tensor) -> torch.Tensor:
        """(..., N, 3) its points, `shape` (..., N, 3) being them about the centre, unturned."""
        return (self.turn[..., None, :, :] @ shape[..., None])[..., 0] + self.centre[..., None, :]


@dataclass
class Touch:
    """Where two bodies touch: body `a` and body `b` (P,) long, the place
    (P, 3), the way from one into the other there (P, 3), a unit vector, how
    far the two are stopped going that way (`hold` (P,)) and going along the
    surface (`grip` (P,)), 1 being fully, and a push on `a`, taken from `b` (P, 3)."""
    a: torch.Tensor
    b: torch.Tensor
    at: torch.Tensor
    into: torch.Tensor
    hold: torch.Tensor
    grip: torch.Tensor
    push: torch.Tensor


def mass_spread(shape: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) a body's inertia for each unit of mass, its mass spread
    over its points `shape` (..., N, 3) by `weight` (..., N)."""
    eye = torch.eye(3, dtype=shape.dtype, device=shape.device)
    each = (shape * shape).sum(-1)[..., None, None] * eye - shape[..., :, None] * shape[..., None, :]
    return (weight[..., None, None] * each).sum(-3) / weight.sum(-1).clamp_min(1e-6)[..., None, None]


def _cross_matrix(r: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) the matrix X of `r` (..., 3) with X v = r x v."""
    x, y, z = r.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).view(*r.shape[:-1], 3, 3)


def step(body: Body, gravity: torch.Tensor, free: torch.Tensor, heavy: Optional[torch.Tensor] = None,
         spread: Optional[torch.Tensor] = None, touch: Optional[Touch] = None) -> Body:
    """The bodies a frame on. Each goes as it went and gains gravity's speed,
    as far as it is `free` (K,) to move: 0 is the fixed world, which nothing
    moves. Then, one touching pair after another, the two are pushed, equal
    and opposite: as hard as stops them going past each other at the place they
    touch, into each other as far as they `hold` and along the surface as far
    as they `grip`, plus `push`. A push changes a body's speed by itself over
    the body's mass (`heavy` (K,)), and its spin by its lever about the centre
    over the body's inertia (the mass times `spread` (K, 3, 3), unturned)."""
    speed, spin = free[:, None] * (body.speed + gravity), free[:, None] * body.spin
    if touch is not None:
        eye = torch.eye(3, dtype=speed.dtype, device=speed.device)
        give = free / heavy
        inertia = torch.linalg.inv(body.turn @ spread @ body.turn.transpose(-1, -2))
        for a, b, at, into, hold, grip, push in zip(touch.a.tolist(), touch.b.tolist(), touch.at, touch.into,
                                                    touch.hold, touch.grip, touch.push):
            ra, rb = at - body.centre[a], at - body.centre[b]
            Xa, Xb = _cross_matrix(ra), _cross_matrix(rb)
            # How a push changes the two's speed past each other at the place.
            yields = give[a] * (eye - Xa @ inertia[a] @ Xa) + give[b] * (eye - Xb @ inertia[b] @ Xb)
            slide = speed[a] + torch.linalg.cross(spin[a], ra) - speed[b] - torch.linalg.cross(spin[b], rb)
            through = (slide @ into) * into
            J = push - torch.linalg.solve(yields + 1e-9 * eye, hold * through + grip * (slide - through))
            one = lambda k: torch.nn.functional.one_hot(torch.tensor(k, device=speed.device), len(speed)).to(speed.dtype)[:, None]
            speed = speed + one(a) * give[a] * J - one(b) * give[b] * J
            spin = (spin + one(a) * give[a] * (inertia[a] @ torch.linalg.cross(ra, J))
                    - one(b) * give[b] * (inertia[b] @ torch.linalg.cross(rb, J)))
    return Body(body.centre + speed, turn_matrix(spin) @ body.turn, speed, spin)


def fitted(shape: torch.Tensor, pos: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The pose that best keeps a body's shape on its points: the turn (...,
    3, 3) and centre (..., 3) that carry `shape` (..., N, 3) onto `pos`, each
    point counting by `weight` (..., N), a few wrong points not dragging it."""
    R, ms, md, _ = robust_kabsch(weight, shape, pos, REFITS)
    return R, md - (R @ ms[..., None])[..., 0]


def pulled(pos: torch.Tensor, place: torch.Tensor, stiff: torch.Tensor) -> torch.Tensor:
    """Each point pulled toward the place that keeps its body's shape, as
    strongly as the model says it is stiff (..., N): 1 a rigid body, 0 no pull."""
    return pos + stiff[..., None] * (place - pos)
