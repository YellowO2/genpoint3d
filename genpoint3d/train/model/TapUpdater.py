"""TAPIP3D's updater, in place of PointDiT's blocks.

To tell whether ours is the gap.

TAPIP3D (`point_tracker_3d.py`, `efficient_updateformer.py`, config
`efficient_updateformer_64track`): every (point, frame) token is

    [visibility logit, correlation (512), Fourier-encoded step from the frame
     before and to the frame after (6 x 33), Fourier-encoded pixel (2 x 33)]

777 numbers, plus a 1D sin-cos code of the frame, through one linear layer to
384. Three rounds of: attention along each point's frames, then the points
meeting through 64 virtual tracks (the tracks read the points, read each
other, the points read them back). A linear head gives the correction. No
round clock, no position of the point itself, no conditioning.

Differences kept on purpose: it takes all the clip's frames at once (TAPIP3D
works in 16-frame windows), and its positions are ours, scene-normalised
(TAPIP3D rescales each window to unit spread).
"""

import math

import torch
from torch import nn

from genpoint3d.train.model.FineFinder import _Attention, _CrossAttnBlock, _Mlp, _posenc


class _AttnBlock(nn.Module):
    """TAPIP3D's `AttnBlock`: self-attention and a feed-forward, pre-norm."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = _Attention(dim, heads, dim // heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h)
        return x + self.mlp(self.norm2(x))


def _sincos(dim: int, T: int, device) -> torch.Tensor:
    """(T, dim) CoTracker's 1D sin-cos code of the frame index."""
    half = dim // 2
    omega = 1.0 / 10000 ** (torch.arange(half, dtype=torch.float64, device=device) / half)
    out = torch.arange(T, dtype=torch.float64, device=device)[:, None] * omega
    emb = torch.cat([out.sin(), out.cos()], dim=-1).float()
    return torch.cat([emb, emb.new_zeros(T, dim - emb.shape[-1])], dim=-1)


class TapUpdater(nn.Module):
    def __init__(self, corr_dim: int, hidden: int = 384, heads: int = 8, depth: int = 3,
                 virtual: int = 64, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.input_dim = 1 + corr_dim + 6 * 33 + 2 * 33
        self.input_transform = nn.Linear(self.input_dim, hidden)
        self.virtual_tracks = nn.Parameter(torch.randn(1, virtual, 1, hidden))
        self.time_blocks = nn.ModuleList(_AttnBlock(hidden, heads, mlp_ratio) for _ in range(depth))
        hd = hidden // heads
        self.v_reads_p = nn.ModuleList(_CrossAttnBlock(hidden, mlp_ratio, heads, hd) for _ in range(depth))
        self.v_self = nn.ModuleList(_AttnBlock(hidden, heads, mlp_ratio) for _ in range(depth))
        self.p_reads_v = nn.ModuleList(_CrossAttnBlock(hidden, mlp_ratio, heads, hd) for _ in range(depth))
        self.flow_head = nn.Linear(hidden, 3)
        self.vis_head = nn.Linear(hidden, 1)
        # TAPIP3D's initialisation: xavier for every linear, the heads small.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
        nn.init.trunc_normal_(self.vis_head.weight, std=0.001)

    def forward(self, corr: torch.Tensor, pos: torch.Tensor, uv: torch.Tensor,
                vis_prev: torch.Tensor | None, uv_mask: torch.Tensor | None = None):
        """
        corr      (B, T, N, C) the finder's comparison
        pos       (B, T, N, 3) the current guess, scene-normalised
        uv        (B, T, N, 2) its pixel, [0, 1] across
        vis_prev  (B, T, N) the visibility logits fed back, or None (zeros)
        uv_mask   (B, T, N) or (B, T, 1) frames with an image; the pixel needs the camera

        returns   (B, T, N, 3) the correction, (B, T, N) visibility logits
        """
        B, T, N, _ = pos.shape
        with torch.autocast(device_type=pos.device.type, enabled=False):
            pos = pos.float()
            step = pos.diff(dim=1)
            fwd = torch.cat([-step, torch.zeros_like(pos[:, :1])], dim=1)
            bwd = torch.cat([torch.zeros_like(pos[:, :1]), step], dim=1)
            pix = _posenc(uv.float())
            if uv_mask is not None:
                pix = pix * uv_mask[..., None]
            vis = (vis_prev.float() if vis_prev is not None else pos.new_zeros(B, T, N))[..., None]
            x = torch.cat([vis, corr.float(), _posenc(torch.cat([fwd, bwd], -1)), pix], dim=-1)
            x = x + _sincos(self.input_dim, T, pos.device)[None, :, None]
        tok = self.input_transform(x).permute(0, 2, 1, 3)                    # (B, N, T, D)
        M = self.virtual_tracks.shape[1]
        tok = torch.cat([tok, self.virtual_tracks.to(tok.dtype).expand(B, M, T, -1)], dim=1)
        Nt = tok.shape[1]
        for i in range(len(self.time_blocks)):
            tok = self.time_blocks[i](tok.reshape(B * Nt, T, -1)).view(B, Nt, T, -1)
            s = tok.permute(0, 2, 1, 3).reshape(B * T, Nt, -1)
            p, v = s[:, :N], s[:, N:]
            v = self.v_reads_p[i](v, p)
            v = self.v_self[i](v)
            p = self.p_reads_v[i](p, v)
            tok = torch.cat([p, v], dim=1).view(B, T, Nt, -1).permute(0, 2, 1, 3)
        tok = tok[:, :N].permute(0, 2, 1, 3)                                  # (B, T, N, D)
        return self.flow_head(tok), self.vis_head(tok)[..., 0]
