"""The depth head's loss: how far its depth is from the truth, up to scale."""

import torch


def depth_loss(head, fmap: torch.Tensor, depth: torch.Tensor, lam: float = 0.85) -> torch.Tensor:
    """head: the depth head; fmap (B, T, C, h, w) the CNN's map; depth
    (B, T, h, w) the true z-depth in metres at each cell, 0 where unknown.

    The scale-invariant log error of Eigen et al. (2014), per frame and then
    averaged: sqrt(mean(g^2) - lam * mean(g)^2) with g = log(said) - log(true).
    At lam = 1 a depth right only up to one scale per frame costs nothing,
    which is all a single video can tell; 0.85 keeps a little of the scale,
    as Eigen's and most depth networks since.
    """
    B, T = fmap.shape[:2]
    said = head(fmap.flatten(0, 1)).reshape(B, T, *fmap.shape[-2:]).float()     # log depth
    known = depth > 0
    g = (said - depth.float().clamp_min(1e-6).log()) * known
    n = known.flatten(2).sum(-1).clamp_min(1)
    mean = g.flatten(2).sum(-1) / n
    sq = (g ** 2).flatten(2).sum(-1) / n
    return (sq - lam * mean ** 2).clamp_min(1e-12).sqrt().mean()
