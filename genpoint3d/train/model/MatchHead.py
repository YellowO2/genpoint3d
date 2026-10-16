"""Score how alike each patch looks to each point; the best is the match."""

import torch
import torch.nn.functional as F
from torch import nn

from genpoint3d.train.model.tools.layers import bounded_exp, log_param


class MatchHead(nn.Module):
    """Cosine between `f' = normalize(f + MLP(f))` for template and patches.

    One MLP, shared by both sides, `Linear(C, h) -> GELU -> Linear(h, C)`. The
    last layer starts at exactly zero, so at step 0 `f' = normalize(f)` and the
    scores are the raw cosine, bit for bit.

    `f'` is never built for the patches. They are (B, T, P, C) with P up to
    2304 -- 1.4 GB in fp32 at batch 16 -- and the obvious code holds three more
    of those for the backward pass. With `d = W2 h` the residual and `h` the
    hidden activation, everything the cosine needs is an h-wide product:

        q' . f'    = q' . f  +  (q' W2) . h
        |f'|^2     = |f|^2  +  (2 f W2 + h W2^T W2) . h

    so the largest new tensors are (B, T, P, h), a sixth of the size at h = 64.
    The last layer has no bias for this reason: a bias is a full-width term.
    """

    def __init__(self, feat_dim: int, hidden: int = 64, tau: float = 0.07) -> None:
        super().__init__()
        self.inp = nn.Linear(feat_dim, hidden)
        self.out = nn.Linear(hidden, feat_dim, bias=False)
        nn.init.zeros_(self.out.weight)
        # Temperature of the matching loss only. The trunk reads the scores
        # through its own (`match_log_tau`, `corr_log_scale`).
        self.loss_log_tau = log_param(tau)

    def tau(self) -> torch.Tensor:
        return bounded_exp(self.loss_log_tau, 0.01, 1.0)

    def forward(self, feats: torch.Tensor, id_card: torch.Tensor) -> torch.Tensor:
        """feats (B, T, P, C), id_card (B, N, C), fp32 -> (B, T, N, P) cosine."""
        W2 = self.out.weight                                          # (C, h)
        q = F.normalize(id_card + F.gelu(self.inp(id_card)) @ W2.T, dim=-1)
        h = F.gelu(self.inp(feats))                                   # (B, T, P, h)
        corr = (torch.einsum("bnc,btpc->btnp", q, feats)
                + torch.einsum("bnh,btph->btnp", q @ W2, h))
        # |f'| as |f| times a factor that is exactly 1 while W2 is zero.
        norm = torch.linalg.vector_norm(feats, dim=-1)                # (B, T, P)
        extra = ((2 * (feats @ W2) + h @ (W2.T @ W2)) * h).sum(-1)
        norm = norm * (1 + extra / norm.pow(2).clamp_min(1e-12)).clamp_min(1e-12).sqrt()
        return corr / norm.clamp_min(1e-6)[:, :, None]
