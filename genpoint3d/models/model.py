"""
THE ARCHITECTURE. Open this file to see the design.

This is stages 2-5 of `docs/map.html`, in order. The transformer machinery it
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

from genpoint3d.models.layers import (
    Block, CrossBlock, FourierEmbedding, RMSNorm, RoPE, zero_init,
)


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
        cond_dim: int | None = None,
        cross_attn: bool = False,
        feat_dim: int | None = None,
    ) -> None:
        super().__init__()
        # The conditioning width has no reason to differ from the model width,
        # and silently defaulting it to a constant made a size-64 model try to
        # add a 256-wide vector. Follow `dim` unless told otherwise.
        cond_dim = cond_dim or dim
        feat_dim = feat_dim or dim
        self.dim, self.depth, self.num_heads = dim, depth, num_heads
        self.cross_attn = cross_attn
        head_dim = dim // num_heads

        # [2] path tokeniser -- genuinely just an embedding layer
        self.token_proj = nn.Linear(3, dim, bias=False)

        # [3] conditioning vector c
        # The encoder's output width is its own choice, so project rather than
        # assume it matches.
        self.id_proj = nn.Linear(feat_dim, cond_dim, bias=False) if cross_attn else None
        self.ctx_proj = nn.Linear(feat_dim, dim, bias=False) if (cross_attn and feat_dim != dim) else None
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

        # [4c] point-image cross-attention, and the tracking/forecasting switch.
        # A masked frame's features are replaced wholesale by `null_ctx`, which
        # means "nothing here". One shared vector suffices because RoPE already
        # tells the model which frame it is looking at.
        if cross_attn:
            self.cross_blocks = nn.ModuleList(
                CrossBlock(dim, num_heads, mlp_mult, cond_dim) for _ in range(depth)
            )
            self.null_ctx = nn.Parameter(torch.randn(dim) * 0.02)

        # [5] position head
        self.out_norm = RMSNorm(dim)
        self.out = zero_init(nn.Linear(dim, 3, bias=False))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        anchor: torch.Tensor,
        context: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
        id_card: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:           (B, T, N, 3)    noisy trajectory, in model units
        k:           (B,) or (B, T)  noise level in [0, 1]. Per-frame `k` is
                                     what diffusion forcing needs later, so it
                                     is supported from the start.
        anchor:      (B, N, 3)       scene-normalised frame-0 position
        context:     (B, T, P, D)    DINOv3 patch features per frame  [4c]
        visual_mask: (B, T) bool     True = frame has visual conditioning.
                                     False swaps in `null_ctx` -> forecasting.
        id_card:     (B, N, D)       DINOv3 feature sampled at the query  [3]

        returns      (B, T, N, 3)    predicted velocity

        The last three are optional; without them this is the step-2 model that
        sees no images at all.
        """
        B, T, N, _ = x.shape
        dev, dt = x.device, x.dtype

        # --- [3] one conditioning vector per (frame, point) ---
        if k.dim() == 1:
            k = k[:, None].expand(B, T)
        cond = self.time_emb(k[..., None])[:, :, None] + self.anchor_emb(anchor)[:, None]
        if id_card is not None:
            cond = cond + self.id_proj(id_card)[:, None]   # the "what am I" term
        cond = self.cond_mlp(cond)                                    # (B, T, N, C)

        if context is not None and self.ctx_proj is not None:
            context = self.ctx_proj(context)

        # Masked frames lose their features entirely, before any attention.
        if context is not None and visual_mask is not None:
            context = torch.where(
                visual_mask[..., None, None], context, self.null_ctx.to(context.dtype)
            )

        # --- [2] tokenise ---
        h = self.token_proj(x)                                        # (B, T, N, D)

        # --- positions for RoPE, and the causal mask ---
        t_pos = torch.arange(T, device=dev, dtype=dt)[None, :, None].expand(B * N, T, 1)
        theta_time = self.time_rope(t_pos)                             # (B*N, H, T, d)
        theta_space = self.space_rope(anchor.repeat_interleave(T, 0))  # (B*T, H, N, d)
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()[None, None]

        # --- [4] blocks ---
        cross_blocks = self.cross_blocks if (self.cross_attn and context is not None) else [None] * self.depth
        for temporal, spatial, cross in zip(self.temporal_blocks, self.spatial_blocks, cross_blocks):
            # each point's own timeline; a frame may only see its own past
            ht = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
            ct = cond.permute(0, 2, 1, 3).reshape(B * N, T, -1)
            h = temporal(ht, ct, theta=theta_time, attn_mask=causal)
            h = h.view(B, N, T, -1).permute(0, 2, 1, 3)

            # all points within one frame
            hs = h.reshape(B * T, N, -1)
            cs = cond.reshape(B * T, N, -1)
            h = spatial(hs, cs, theta=theta_space)

            # [4c] each point queries its own frame's feature map (or the null)
            if cross is not None:
                h = cross(h, cs, context.reshape(B * T, -1, self.dim))
            h = h.view(B, T, N, -1)

        # --- [5] head ---
        return self.out(self.out_norm(h))
