"""The fixed parts of a physics engine; the forces and the stiffness are the model's to say."""

from dataclasses import dataclass
from typing import Optional

import torch

from genpoint3d.train.tools.kabsch import kabsch, turn_matrix


@dataclass
class Body:
    """A rigid body: where its centre of gravity is (..., 3), how it is turned
    (..., 3, 3), and its speed and spin (..., 3), a frame."""
    centre: torch.Tensor
    turn: torch.Tensor
    speed: torch.Tensor
    spin: torch.Tensor

    def points(self, shape: torch.Tensor) -> torch.Tensor:
        """(..., N, 3) its points, `shape` (..., N, 3) being them about the centre, unturned."""
        return (self.turn[..., None, :, :] @ shape[..., None])[..., 0] + self.centre[..., None, :]


def mass_spread(shape: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) a body's inertia for each unit of mass, its mass spread evenly over its points `shape`."""
    eye = torch.eye(3, dtype=shape.dtype, device=shape.device)
    return ((shape * shape).sum(-1)[..., None, None] * eye - shape[..., :, None] * shape[..., None, :]).mean(-3)


def step(body: Body, gravity: torch.Tensor, spread: torch.Tensor, force: Optional[torch.Tensor] = None,
         at: Optional[torch.Tensor] = None, mass: float = 1.0) -> Body:
    """The body a frame on. Gravity and each force (..., K, 3), acting at the
    place `at` (..., K, 3), change its speed (the force over its mass) and its
    spin (the force's lever about the centre, over its inertia, mass times
    `spread`); then it moves and turns as it now goes."""
    speed, spin = body.speed + gravity, body.spin
    if force is not None:
        speed = speed + force.sum(-2) / mass
        lever = torch.linalg.cross(at - body.centre[..., None, :], force).sum(-2)
        inertia = mass * body.turn @ spread @ body.turn.transpose(-1, -2)
        spin = spin + torch.linalg.solve(inertia, lever[..., None])[..., 0]
    return Body(body.centre + speed, turn_matrix(spin) @ body.turn, speed, spin)


def rigid_place(shape: torch.Tensor, pos: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """(..., N, 3) where each point must be for its group to keep its shape:
    `shape` (..., N, 3) carried onto the points `pos` as well as it goes, each
    point counting by `weight` (..., N)."""
    R, ms, md, _ = kabsch(weight, shape, pos)
    return (R[..., None, :, :] @ (shape - ms[..., None, :])[..., None])[..., 0] + md[..., None, :]


def pulled(pos: torch.Tensor, place: torch.Tensor, stiff: torch.Tensor) -> torch.Tensor:
    """Each point pulled toward its rigid place as strongly as its group is stiff (..., N): 1 a rigid body, 0 no pull."""
    return pos + stiff[..., None] * (place - pos)
