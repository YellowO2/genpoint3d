"""
THE ARCHITECTURE. Open this file to see the design.

This is stages 2-5 of `docs/MAP.md`, in order. The transformer machinery it
uses (RMSNorm, RoPE, Attention, ...) lives in `layers.py` -- you should not
need to read that to understand what happens here.

    [2] tokenise     3D positions          -> tokens
    [3] conditioning anchor + noise level  -> one vector c per (frame, point)
    [4] blocks       temporal + spatial attention, x depth, modulated by c
    [5] head         tokens                -> predicted velocity

Stage 1 (the DINOv3 encoder) and the point-image cross-attention inside
stage 4 arrive in step 3 of the build order. Until then the model sees no
images at all.
"""

import torch
from torch import nn

from genpoint3d.models.layers import Block, FourierEmbedding, RMSNorm, RoPE, zero_init


class PointDiT(nn.Module):
    """Predicts a flow-matching velocity for `(T, N)` 3D point tokens.

    Tokens live on a `(T frames) x (N points)` grid, so attention is factorised
    into two cheap slices instead of one `T*N` blob:

        temporal   (B*N, T, D)   one point across time, causally masked
        spatial    (B*T, N, D)   all points within one frame

    Without images, the conditioning is stage 3 minus its DINOv3 term: the
    model still knows *which* point each token is (from its frame-0 position)
    and how noisy the input is, but nothing about what the scene looks like.
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_mult: int = 3,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.dim, self.depth, self.num_heads = dim, depth, num_heads
        head_dim = dim // num_heads

        # [2] point tokeniser -- genuinely just an embedding layer
        self.token_proj = nn.Linear(3, dim, bias=False)

        # [3] conditioning vector c
        self.anchor_emb = FourierEmbedding(3, cond_dim)
        self.time_emb = FourierEmbedding(1, cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim)
        )

        # [4] DiT blocks -- temporal and spatial interleaved, `depth` times
        self.temporal_blocks = nn.ModuleList(
            Block(dim, num_heads, mlp_mult, cond_dim) for _ in range(depth)
        )
        self.spatial_blocks = nn.ModuleList(
            Block(dim, num_heads, mlp_mult, cond_dim) for _ in range(depth)
        )
        self.time_rope = RoPE(head_dim, num_heads, n_axes=1)   # position = frame index
        self.space_rope = RoPE(head_dim, num_heads, n_axes=3)  # position = query xyz

        # [5] position head
        self.out_norm = RMSNorm(dim)
        self.out = zero_init(nn.Linear(dim, 3, bias=False))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, k: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, T, N, 3)    noisy trajectory, in model units
        k:      (B,) or (B, T)  noise level in [0, 1]. Per-frame `k` is what
                                diffusion forcing needs later, so it is
                                supported from the start.
        anchor: (B, N, 3)       scene-normalised frame-0 position of each query

        returns (B, T, N, 3)    predicted velocity
        """
        B, T, N, _ = x.shape
        dev, dt = x.device, x.dtype

        # --- [3] one conditioning vector per (frame, point) ---
        if k.dim() == 1:
            k = k[:, None].expand(B, T)
        cond = self.time_emb(k[..., None])[:, :, None] + self.anchor_emb(anchor)[:, None]
        cond = self.cond_mlp(cond)                                    # (B, T, N, C)

        # --- [2] tokenise ---
        h = self.token_proj(x)                                        # (B, T, N, D)

        # --- positions for RoPE, and the causal mask ---
        t_pos = torch.arange(T, device=dev, dtype=dt)[None, :, None].expand(B * N, T, 1)
        theta_time = self.time_rope(t_pos)                             # (B*N, H, T, d)
        theta_space = self.space_rope(anchor.repeat_interleave(T, 0))  # (B*T, H, N, d)
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()[None, None]

        # --- [4] blocks ---
        for temporal, spatial in zip(self.temporal_blocks, self.spatial_blocks):
            # each point's own timeline; a frame may only see its own past
            ht = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
            ct = cond.permute(0, 2, 1, 3).reshape(B * N, T, -1)
            h = temporal(ht, ct, theta=theta_time, attn_mask=causal)
            h = h.view(B, N, T, -1).permute(0, 2, 1, 3)

            # all points within one frame
            hs = h.reshape(B * T, N, -1)
            cs = cond.reshape(B * T, N, -1)
            h = spatial(hs, cs, theta=theta_space).view(B, T, N, -1)

        # --- [5] head ---
        return self.out(self.out_norm(h))
