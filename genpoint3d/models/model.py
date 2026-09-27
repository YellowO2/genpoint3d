"""
THE ARCHITECTURE. Open this file to see the design.

This is stages 2-5 of `docs/map.html`, in order. The transformer machinery it
uses (RMSNorm, RoPE, Attention, ...) lives in `layers.py`.

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
        locality: bool = True,
        correlate: bool = True,
        causal: bool = True,
        costvol: bool = True,
        cv_k: int = 16,
        cv_support: int = 8,
        cv_dim: int = 32,
    ) -> None:
        super().__init__()
        # The conditioning width has no reason to differ from the model width,
        # and silently defaulting it to a constant made a size-64 model try to
        # add a 256-wide vector. Follow `dim` unless told otherwise.
        cond_dim = cond_dim or dim
        feat_dim = feat_dim or dim
        self.dim, self.depth, self.num_heads = dim, depth, num_heads
        self.cross_attn = cross_attn
        # Off reproduces the model that scored the static baseline while ignoring
        # which video it was given, which is the comparison this exists against.
        self.locality = locality and cross_attn
        self.correlate = correlate and cross_attn
        # Off lets a frame attend to later frames. Tracking has every image in
        # hand, so bidirectional context is legitimate and is what every tracker
        # in the benchmark table uses. The mask is NOT what keeps forecasting
        # honest -- a masked frame already gets `null_ctx`, so it holds nothing
        # to leak (docs/references.md:19). It exists for autoregressive rollout
        # and diffusion forcing, neither of which is built yet.
        self.causal = causal
        self.costvol = costvol and cross_attn
        self.cv_k, self.cv_support, self.cv_dim = cv_k, cv_support, cv_dim
        head_dim = dim // num_heads

        # [path encoder] representing the current diffused path for the video
        self.token_proj = nn.Linear(3, dim, bias=False)
        
        # [feature adapter] trainable, unlike encoder.py which is cached frozen.
        # Always on when there are features: the patch position is added to
        # frame_proj's output, and W(f + p) = Wf + Wp, so the what/where balance
        # is only learnable if a layer sits before the sum.
        self.frame_proj = nn.Linear(feat_dim, dim, bias=False) if cross_attn else None
        self.patch_pos = FourierEmbedding(3, dim) if cross_attn else None
        self.id_feature_proj = nn.Linear(feat_dim, cond_dim, bias=False) if cross_attn else None

        # [correlation] "does this patch look like me?", computed instead of
        # discovered. Both sides read the RAW backbone features, not the adapter's
        # output, so matching is not entangled with the patch position that the
        # adapter adds in. Separate q/k projections because a template and a
        # patch are not the same kind of thing.
        if self.correlate:
            self.corr_q = nn.Linear(feat_dim, dim, bias=False)
            self.corr_k = nn.Linear(feat_dim, dim, bias=False)
            # Where in this frame the template matches best, as a 3D position.
            # Gated to zero at init so the model starts as the run5 model.
            self.match_emb = FourierEmbedding(3, cond_dim)
            self.match_gate = nn.Parameter(torch.zeros(1))

        # [cost volume] CoTracker3 and genpt compare a 7x7 support WINDOW around
        # the query against a 7x7 crop at the current estimate, then MLP the
        # resulting 49x49 table. One similarity score per patch throws away where
        # inside the window the match peaks and how sharply; the table keeps it.
        # We take k nearest patches in 3D rather than a square image crop,
        # because our estimate lives in 3D and we never project -- the same
        # substitution TAPIP3D makes in knn_feature_4d.
        if self.costvol:
            # Matching happens in a narrow learned space: gathering feat_dim-wide
            # features per (frame, point, neighbour) is 600 MB at batch 16.
            self.cv_proj = nn.Linear(feat_dim, cv_dim, bias=False)
            self.cv_mlp = nn.Sequential(
                nn.Linear(cv_support * cv_k + cv_k * 3, cond_dim), nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.cv_gate = nn.Parameter(torch.zeros(1))

        # [condition encoder] conditioning vector c
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
                CrossBlock(dim, num_heads, mlp_mult, cond_dim,
                           locality=self.locality, correlate=self.correlate)
                for _ in range(depth)
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
        patch_xyz: torch.Tensor | None = None,
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
        patch_xyz:   (B, T, P, 3)    each patch's 3D position, `anchor`'s space.
                                     Required with `context`: *what* plus *where*.

        returns      (B, T, N, 3)    predicted velocity

        The visual arguments are optional as a group.
        """
        B, T, N, _ = x.shape
        dev, dt = x.device, x.dtype

        # --- where each point currently thinks it is, relative to every patch ---
        # x and patch_xyz share one normalised space, so this is a plain distance
        # and needs no camera. It is the lookup every established tracker does by
        # projecting into the image and sampling there; in 3D the projection is
        # unnecessary. Computed once, used by the cost volume and by every block.
        dist2 = None
        if context is not None and patch_xyz is not None and (self.locality or self.costvol):
            # ||a-b||^2 = |a|^2 + |b|^2 - 2a.b, rather than materialising the
            # (B, T, N, P, 3) difference -- that tensor is 340 MB at batch 16.
            dist2 = (x.pow(2).sum(-1)[..., None]
                     + patch_xyz.pow(2).sum(-1)[:, :, None]
                     - 2 * torch.einsum("btnc,btpc->btnp", x, patch_xyz)).clamp_min(0)

        # --- [cost volume] support window vs the neighbourhood of the estimate ---
        cv = None
        if self.costvol and dist2 is not None:
            cf = self.cv_proj(context)                                # (B, T, P, cv)
            # Support: the patches around where the point STARTS. The 3D
            # equivalent of CoTracker's window around the query pixel, and it
            # replaces a single-vector template that many patches match equally.
            d0 = (anchor.pow(2).sum(-1)[..., None]
                  + patch_xyz[:, 0].pow(2).sum(-1)[:, None]
                  - 2 * torch.einsum("bnc,bpc->bnp", anchor, patch_xyz[:, 0]))
            si = d0.topk(self.cv_support, dim=-1, largest=False).indices   # (B, N, S)
            sup = cf[:, 0].gather(1, si.reshape(B, -1)[..., None]
                                  .expand(-1, -1, self.cv_dim)).reshape(B, N, self.cv_support, -1)

            # Neighbourhood: the patches nearest the CURRENT estimate, per frame.
            ni = dist2.topk(self.cv_k, dim=-1, largest=False).indices      # (B, T, N, K)
            flat = ni.reshape(B, T, -1)[..., None]
            nb = cf.gather(2, flat.expand(-1, -1, -1, self.cv_dim)) \
                   .reshape(B, T, N, self.cv_k, -1)
            # Offsets, not absolute positions: "two cells up-left of you" is the
            # part a single similarity score cannot express.
            off = patch_xyz.gather(2, flat.expand(-1, -1, -1, 3)) \
                           .reshape(B, T, N, self.cv_k, 3) - x[..., None, :]

            cost = torch.einsum("bnsc,btnkc->btnsk", sup, nb) * self.cv_dim ** -0.5
            cv = self.cv_mlp(torch.cat([cost.flatten(3), off.flatten(3)], dim=-1))
            if visual_mask is not None:
                cv = cv * visual_mask[..., None, None]

        # --- [correlation] template vs every patch, on the RAW features ---
        # Read before the adapter overwrites `context`. The model previously had
        # the template and the patches but nothing that compared them, so the
        # one route from image to position had to be discovered from position
        # error alone -- while "predict little motion" cut the loss immediately.
        corr = match_xyz = None
        if (self.correlate and context is not None and id_card is not None
                and patch_xyz is not None):
            q = self.corr_q(id_card)                                  # (B, N, D)
            corr = torch.einsum("bnc,btpc->btnp", q, self.corr_k(context))
            corr = corr * self.dim ** -0.5
            if visual_mask is not None:
                corr = corr * visual_mask[..., None, None]
            # Softly, where in this frame the template matches best. Independent
            # of the diffused path, so it is evidence rather than an echo of the
            # model's own guess -- the one term here that the image alone decides.
            match_xyz = torch.einsum("btnp,btpc->btnc",
                                     corr.float().softmax(-1).to(patch_xyz.dtype),
                                     patch_xyz)
            if visual_mask is not None:
                match_xyz = match_xyz * visual_mask[..., None, None]

        # --- [3] one conditioning vector per (frame, point) ---
        if k.dim() == 1:
            k = k[:, None].expand(B, T)
        cond = self.time_emb(k[..., None])[:, :, None] + self.anchor_emb(anchor)[:, None]
        if id_card is not None:
            cond = cond + self.id_feature_proj(id_card)[:, None]   # the "what am I" term
        if match_xyz is not None:
            cond = cond + self.match_gate * self.match_emb(match_xyz)  # "and here"
        if cv is not None:
            cond = cond + self.cv_gate * cv.to(cond.dtype)   # "and this is the fit"
        cond = self.cond_mlp(cond)                                    # (B, T, N, C)

        # --- [feature adapter] raw backbone width -> model width, plus position ---
        if context is not None:
            if patch_xyz is None:
                raise ValueError("context needs patch_xyz -- features without "
                                 "positions say what is in the frame but not where")
            context = self.frame_proj(context) + self.patch_pos(patch_xyz)

            # Masked frames lose features AND positions, before any attention.
            if visual_mask is not None:
                context = torch.where(
                    visual_mask[..., None, None], context, self.null_ctx.to(context.dtype)
                )

        # --- the same distances, folded into the attention bias ---
        if dist2 is not None and self.locality:
            # A masked frame is one the model is not allowed to see, and its
            # patch positions come from that frame's depth. Its features are
            # already replaced, so nothing can flow today -- every value is the
            # same null vector, whatever the weights. Zeroed anyway: this is the
            # forecasting path, and a hole in it should not wait for the null
            # handling to change before it becomes a leak.
            if visual_mask is not None:
                dist2 = dist2 * visual_mask[..., None, None]
            dist2 = dist2.reshape(B * T, N, -1)
        else:
            dist2 = None
        if corr is not None:
            corr = corr.reshape(B * T, N, -1)

        # --- [2] tokenise ---
        h = self.token_proj(x)                                        # (B, T, N, D)

        # --- positions for RoPE, and the causal mask ---
        t_pos = torch.arange(T, device=dev, dtype=dt)[None, :, None].expand(B * N, T, 1)
        theta_time = self.time_rope(t_pos)                             # (B*N, H, T, d)
        theta_space = self.space_rope(anchor.repeat_interleave(T, 0))  # (B*T, H, N, d)
        causal = (torch.ones(T, T, dtype=torch.bool, device=dev).tril()[None, None]
                  if self.causal else None)

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
                h = cross(h, cs, context.reshape(B * T, -1, self.dim),
                          dist2=dist2, corr=corr)
            h = h.view(B, T, N, -1)

        # --- [5] head ---
        return self.out(self.out_norm(h))
