"""Which points move together, and where each point would be if it moved with them.

Every Kubric object is rigid, and a hidden point is still on its object. Carried
by the rotation and shift that move its object's VISIBLE points from frame 0 to
now, a hidden point scores 0.76 on moving hidden points (run33 0.28, TAPIP3D
0.635; `_buddy` in `score/track_scores.py`). That needs the truth to say which
points share an object and which are visible. This is the same idea from the
model's own guess, the way Learning to Simulate builds its messages: from the
relative position of two points, not where either is.

Two points are buddies when the gap between them, in the current guess, has
stayed the length it was at frame 0 -- in the frames up to this one, so a frame
never reads a later one. The weight of point j for point i at frame t is

    w_ij(t) = exp(-change_ij(t) / sigma) * exp(-gap_ij(0) / rho) * seen_j(t)

    change   mean over frames <= t of | |p_i - p_j| - |a_i - a_j| |, each
             frame weighted by how sure the model is it sees both there, so
             a point's own blind guesses do not cut it off from its object
    gap(0)   |a_i - a_j|, how far apart they start: nearby points are likelier
             to share an object
    seen     how sure the model was, the iteration before, that it can see j
             at t (the visibility fed back), 1/2 without it: a buddy is only
             as good as its own guess

with sigma and rho learned. Two things come out of it:

    proposal  each point carried from frame 0 to t by the rotation and shift
              that best move its buddies (weighted Kabsch), as an offset from
              its current guess, plus how much buddy weight backs it. The
              network decides how far to trust it.
    bias      optionally, a learned attention bias per pair for the spatial
              attention, from the same pair numbers, so a point can listen
              to its buddies in every block.

The rotation is fitted without gradient: an SVD's gradient blows up when two
singular values meet, which a point with few buddies hits. The shift and the
weights keep theirs.
"""

import torch
from torch import nn

from genpoint3d.train.model.tools.layers import bounded_exp, log_param, zero_init


class Buddies(nn.Module):
    def __init__(self, num_heads: int, depth: int, bias: bool = False, hidden: int = 16) -> None:
        super().__init__()
        self.log_sigma = log_param(0.02)
        self.log_rho = log_param(0.5)
        self.num_heads, self.depth = num_heads, depth
        # 3 numbers per pair -> one bias per head per block. Starts at zero.
        self.bias = (nn.Sequential(nn.Linear(3, hidden), nn.GELU(),
                                   zero_init(nn.Linear(hidden, depth * num_heads), almost=False))
                     if bias else None)

    def forward(self, pos: torch.Tensor, anchor: torch.Tensor,
                seen: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, list | None]:
        """
        pos:     (B, T, N, 3) the current guess, absolute, in `anchor`'s space
        anchor:  (B, N, 3)    frame 0
        seen:    (B, T, N)    visibility logits fed back, or None

        returns  proposal (B, T, N, 3) where the buddies say each point is,
                 minus `pos`; support (B, T, N) log(1 + total buddy weight);
                 and, with `bias`, one (B*T, H, N, N) attention bias per
                 block, else None.
        """
        B, T, N, _ = pos.shape
        pos, anchor = pos.float(), anchor.float()
        gap0 = (anchor[:, :, None] - anchor[:, None]).norm(dim=-1)          # (B, N, N)
        gap = (pos[:, :, :, None] - pos[:, :, None]).norm(dim=-1)           # (B, T, N, N)
        now = gap - gap0[:, None]
        sj = (torch.sigmoid(seen.float()) if seen is not None
              else torch.full((B, T, N), 0.5, device=pos.device))
        both = sj[:, :, :, None] * sj[:, :, None, :]                       # (B, T, N, N)
        change = (now.abs() * both).cumsum(1) / both.cumsum(1).clamp(min=1e-6)

        sigma = bounded_exp(self.log_sigma, 1e-3, 1.0)
        rho = bounded_exp(self.log_rho, 1e-2, 10.0)
        w = torch.exp(-change / sigma - gap0[:, None] / rho) * sj[:, :, None, :]
        w = w * (1 - torch.eye(N, device=pos.device, dtype=w.dtype))
        n = w.sum(-1)                                                       # (B, T, N)

        mean = lambda x: (w[..., None] * x).sum(-2) / n.clamp(min=1e-6)[..., None]
        src = anchor[:, None, None]                                         # (B, 1, 1, j, 3)
        dst = pos[:, :, None]                                               # (B, T, 1, j, 3)
        cs, cd = mean(src.expand(B, T, N, N, 3)), mean(dst.expand(B, T, N, N, 3))
        with torch.no_grad():
            H = torch.einsum("btij,btijk,btijl->btikl", w,
                             src - cs[..., None, :], dst - cd[..., None, :])
            U, _, Vh = torch.linalg.svd(H)
            V, Ut = Vh.transpose(-1, -2), U.transpose(-1, -2)
            flip = torch.ones_like(cs)
            flip[..., 2] = torch.linalg.det(V @ Ut).sign()
            R = V @ torch.diag_embed(flip) @ Ut
        proposal = (R @ (anchor[:, None] - cs)[..., None])[..., 0] + cd - pos
        support = torch.log1p(n)

        biases = None
        if self.bias is not None:
            feats = torch.stack([gap0[:, None].expand_as(now), change, now], dim=-1)
            b = self.bias(feats)                                            # (B, T, N, N, D*H)
            b = b.view(B * T, N, N, self.depth, self.num_heads).permute(3, 0, 4, 1, 2)
            biases = list(b.unbind(0))
        return proposal, support, biases
