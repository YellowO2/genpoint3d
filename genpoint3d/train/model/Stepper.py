"""Every point stepped one frame on by a learned simulator.

The simulator is GNS (Sanchez-Gonzalez et al. 2020,
`deepmind-research/learning_to_simulate`), layer for layer where it applies. A
point's last `history` velocities and gravity make its node, its `k` nearest
points its edges (their offset, distance, and whether the two are on one
piece); ten rounds of message passing, each with its own weights and a
residual, then the point's acceleration, and the next position by
semi-implicit Euler. Trained one step at a time on true positions with a
random walk added, as GNS, so it learns to recover from its own drift.

GNS's particles fill a box; ours are points on surfaces in a clip's own unit:
the k nearest instead of a radius (the points are sparse and unevenly spread),
no box walls (the floor is points too), and gravity given (its strength along
which way is down, where that is known).
"""

import torch
from torch import nn

from genpoint3d.train.tools.kabsch import MIN_POSE_POINTS, kabsch


def _mlp(i: int, o: int, width: int = 128, norm: bool = True) -> nn.Sequential:
    """GNS's MLP: two hidden layers of `width` with ReLU, LayerNorm on the output
    (all but the decoder's)."""
    layers = [nn.Linear(i, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, o)]
    return nn.Sequential(*layers, nn.LayerNorm(o)) if norm else nn.Sequential(*layers)


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """(S, N, C) rows picked by `idx` (S, N, k) -> (S, N, k, C)."""
    S, N, k = idx.shape
    return x.gather(1, idx.reshape(S, N * k, 1).expand(-1, -1, x.shape[-1])).view(S, N, k, -1)


def _random_walk(past: torch.Tensor, std: float) -> torch.Tensor:
    """(S, H+1, N, 3) noise for `past`: GNS's random walk on the velocities
    (`get_random_walk_noise_for_position_sequence`), `std` on the last step, the
    first position untouched."""
    H = past.shape[1] - 1
    v = (torch.randn_like(past[:, 1:]) * (std / H ** 0.5)).cumsum(1)
    return torch.cat([torch.zeros_like(past[:, :1]), v.cumsum(1)], 1)


def _shape_match(x: torch.Tensor, rest: torch.Tensor, piece: torch.Tensor) -> torch.Tensor:
    """(S, N, 3) `x` with each piece's points set where the turn and shift that
    best carry the piece's `rest` (S, N, 3) onto `x` put them (shape matching,
    Mueller et al. 2005; FIGNet keeps objects rigid this way in its rollouts).
    Points on no piece (-1), or on one of under MIN_POSE_POINTS points, kept."""
    w = ((piece[:, :, None] == piece[:, None]) & (piece >= 0)[:, :, None]).double()      # (S, N, N)
    r = rest.double()
    R, ms, md, s = kabsch(w, r[:, None], x.double()[:, None])
    placed = (R @ (r - ms)[..., None])[..., 0] + md
    return torch.where((s >= MIN_POSE_POINTS)[..., None], placed.to(x.dtype), x)


class Stepper(nn.Module):
    def __init__(self, history: int = 5, k: int = 16, width: int = 128, rounds: int = 10) -> None:
        super().__init__()
        self.history, self.k = history, k
        # A node: its velocities, gravity along down and whether down is known.
        self.node_in = _mlp(3 * history + 4, width, width)
        # An edge: the offset to the neighbour, its length, one piece or not.
        self.edge_in = _mlp(5, width, width)
        self.edges = nn.ModuleList(_mlp(3 * width, width, width) for _ in range(rounds))
        self.nodes = nn.ModuleList(_mlp(2 * width, width, width) for _ in range(rounds))
        self.out = _mlp(width, 3, width, norm=False)
        # The training clips' velocities and accelerations (mean and spread per
        # axis, the noise folded in as GNS's `_combine_std`) and a typical
        # distance to the k-th nearest point: set once by `set_stats`.
        for name in ("vel_mean", "acc_mean"):
            self.register_buffer(name, torch.zeros(3))
        for name in ("vel_std", "acc_std"):
            self.register_buffer(name, torch.ones(3))
        self.register_buffer("reach", torch.ones(()))

    def set_stats(self, vel: torch.Tensor, acc: torch.Tensor, reach: float, noise: float) -> None:
        """From (M, 3) velocities and accelerations of the training clips and
        the noise trained with."""
        self.vel_mean.copy_(vel.mean(0))
        self.vel_std.copy_((vel.var(0) + noise ** 2).sqrt())
        self.acc_mean.copy_(acc.mean(0))
        self.acc_std.copy_((acc.var(0) + noise ** 2).sqrt())
        self.reach.fill_(reach)

    def forward(self, past: torch.Tensor, gravity: torch.Tensor, piece: torch.Tensor | None = None) -> torch.Tensor:
        """
        past    (S, H+1, N, 3)  each point's last H+1 positions, in the clip's unit
        gravity (S, 4)          gravity per frame squared along down, and 1 where down is known
        piece   (S, N) long     each point's piece, -1 where not known; None: none known

        returns (S, N, 3) each point's acceleration, normalised
        """
        S, _, N, _ = past.shape
        vel = (past.diff(dim=1) - self.vel_mean) / self.vel_std                 # (S, H, N, 3)
        node = torch.cat([vel.transpose(1, 2).reshape(S, N, -1), gravity[:, None].expand(S, N, 4)], -1)
        last = past[:, -1]
        with torch.no_grad():
            # GNS's neighbourhood holds the point itself; so does this one.
            idx = torch.cdist(last, last).topk(min(self.k, N), dim=-1, largest=False).indices
        off = (_gather(last, idx) - last[:, :, None]) / self.reach               # (S, N, k, 3)
        if piece is None:
            same = torch.zeros_like(off[..., :1])
        else:
            mate = _gather(piece[..., None], idx)[..., 0]
            same = ((mate == piece[:, :, None]) & (piece[:, :, None] >= 0)).to(off.dtype)[..., None]
        h = self.node_in(node)
        e = self.edge_in(torch.cat([off, off.norm(dim=-1, keepdim=True), same], -1))
        for edge, node_mlp in zip(self.edges, self.nodes):
            e_new = edge(torch.cat([e, h[:, :, None].expand_as(e), _gather(h, idx)], -1))
            h_new = node_mlp(torch.cat([h, e_new.sum(2)], -1))
            e, h = e + e_new, h + h_new
        return self.out(h)

    def step(self, past: torch.Tensor, gravity: torch.Tensor, piece: torch.Tensor | None = None) -> torch.Tensor:
        """(S, N, 3) each point's next position."""
        acc = self(past, gravity, piece) * self.acc_std + self.acc_mean
        return past[:, -1] + (past[:, -1] - past[:, -2] + acc)

    def loss(self, past: torch.Tensor, nxt: torch.Tensor, gravity: torch.Tensor, piece: torch.Tensor | None,
             noise: float, moving: torch.Tensor) -> torch.Tensor:
        """GNS's one-step loss: `past` (S, H+1, N, 3) with a random walk added,
        the target the acceleration that reaches `nxt` (S, N, 3) shifted by the
        walk's last step, squared error of the normalised acceleration. The
        moving points (`moving`, (S, N)) and the still ones count half each:
        most points are still scene."""
        walk = _random_walk(past, noise)
        noisy = past + walk
        acc = (nxt + walk[:, -1] - noisy[:, -1]) - (noisy[:, -1] - noisy[:, -2])
        err = (self(noisy, gravity, piece) - (acc - self.acc_mean) / self.acc_std).pow(2).sum(-1)
        m = moving.to(err.dtype)
        return ((err * m).sum() / m.sum().clamp_min(1) + (err * (1 - m)).sum() / (1 - m).sum().clamp_min(1)) / 2
