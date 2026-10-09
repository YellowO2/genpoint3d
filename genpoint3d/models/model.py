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

import math

import torch
import torch.nn.functional as F
from torch import nn

from genpoint3d.models.layers import (
    Block, CrossBlock, FourierEmbedding, RMSNorm, RoPE, bounded_exp, log_param,
    zero_init,
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
        adapter_depth: int = 1,
        upsample: int = 1,
        traj_scale: float = 1.0,
        displacement: bool = False,
        legacy_lookup: bool = False,
        time_norm: bool = True,
        locality: bool = True,
        correlate: bool = True,
        locality_mode: str = "patch",
        corr_mode: str = "cosine",
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
        # What turns the sample `x` back into a position in `patch_xyz`'s space;
        # see `_lookup_pos`.
        self.traj_scale, self.displacement = traj_scale, displacement
        self.legacy_lookup = legacy_lookup
        # Frame times in [-1, 1] for temporal RoPE; see `forward`. Off feeds the
        # raw frame index, only so checkpoints trained that way still evaluate
        # as they were trained.
        self.time_norm = time_norm
        self.locality = locality and cross_attn
        self.correlate = correlate and cross_attn
        # "legacy" on either is the prior / the matching as every run up to
        # run14 had them (docs/issues.md B1, B2), only so those checkpoints
        # still evaluate as they were trained.
        if locality_mode not in ("legacy", "patch") or corr_mode not in ("legacy", "cosine"):
            raise ValueError(f"unknown mode: locality {locality_mode!r}, corr {corr_mode!r}")
        self.patch_locality = self.locality and locality_mode == "patch"
        self.cosine = self.correlate and corr_mode == "cosine"
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
        # adapter_depth > 1 puts a non-linearity between the frozen backbone and
        # the model. The encoder is cached frozen, so a single linear map is the
        # only thing that ever adapts DINOv3's features to tracking -- and a
        # linear map cannot do much. Depth 1 stays a bare Linear so the
        # parameter keeps its name and old checkpoints still load.
        if not cross_attn:
            self.frame_proj = None
        elif adapter_depth <= 1:
            self.frame_proj = nn.Linear(feat_dim, dim, bias=False)
        else:
            layers = [nn.Linear(feat_dim, dim, bias=False)]
            for _ in range(adapter_depth - 1):
                layers += [nn.GELU(), nn.Linear(dim, dim, bias=False)]
            self.frame_proj = nn.Sequential(*layers)
        self.patch_pos = FourierEmbedding(3, dim) if cross_attn else None

        # [feature upsampler] the paper's recipe for a finer grid: nearest
        # interpolation, then a convolution (Gen-points sec. 3, "Visual
        # Conditioning"). Nearest alone only repeats each patch into a block of
        # identical cells; the conv is what lets neighbouring cells differ.
        # Applied to the RAW features, ahead of everything, so the correlation
        # and the locality bias see the fine grid too and not only the
        # cross-attention. Residual and zero-initialised: at step 0 the model
        # is exactly the un-upsampled one looking at repeated patches.
        self.upsample = upsample if cross_attn else 1
        if self.upsample > 1:
            self.up_conv = nn.Conv2d(feat_dim, feat_dim, 3, padding=1)
            nn.init.zeros_(self.up_conv.weight)
            nn.init.zeros_(self.up_conv.bias)
        self.id_feature_proj = nn.Linear(feat_dim, cond_dim, bias=False) if cross_attn else None

        # [correlation] "does this patch look like me?", computed instead of
        # discovered. Both sides read the RAW backbone features, not the adapter's
        # output, so matching is not entangled with the patch position that the
        # adapter adds in.
        if self.cosine:
            # The score is the cosine between the two raw features, so a patch
            # identical to the template wins before any training. What the
            # model gets from it is the best match's OFFSET from the point's
            # start, in the units of its own target, added to the token -- so
            # copying it to the output is something a linear layer can do.
            # Softmax temperature of the match, and how many patch spacings
            # from the current estimate a match may be.
            self.match_log_tau = log_param(0.03)
            self.match_log_sigma = log_param(1.0)
            self.match_proj = nn.Linear(4, dim, bias=False)
        elif self.correlate:
            # LEGACY. Two independent random projections, so a patch identical
            # to the template was not preferred; and both readers of the score
            # were gated at zero, so the projections got no gradient at all.
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
        self.time_rope = RoPE(head_dim, num_heads, n_axes=1)   # position = frame time
        self.space_rope = RoPE(head_dim, num_heads, n_axes=3)  # position = query xyz

        # [4c] point-image cross-attention, and the tracking/forecasting switch.
        # A masked frame's features are replaced wholesale by `null_ctx`, which
        # means "nothing here". One shared vector suffices because RoPE already
        # tells the model which frame it is looking at.
        if cross_attn:
            self.cross_blocks = nn.ModuleList(
                CrossBlock(dim, num_heads, mlp_mult, cond_dim,
                           locality=self.locality, correlate=self.correlate,
                           locality_mode=locality_mode, corr_mode=corr_mode)
                for _ in range(depth)
            )
            self.null_ctx = nn.Parameter(torch.randn(dim) * 0.02)

        # [5] position head
        self.out_norm = RMSNorm(dim)
        self.out = zero_init(nn.Linear(dim, 3, bias=False))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _upsample(self, context: torch.Tensor, patch_xyz: torch.Tensor):
        """(B, T, P, F) features and (B, T, P, 3) positions -> the same on a
        grid `upsample` times finer per side, so P grows by upsample^2.

        Positions are interpolated bilinearly, not repeated: four cells sharing
        one position would make the finer grid no finer for the locality bias.
        The cost is that a cell straddling a depth edge gets a position between
        the two surfaces. Caching positions at the fine grid from the depth map
        would remove that; this is the version that needs no new cache.
        """
        B, T, P, Fd = context.shape
        g = math.isqrt(P)
        if g * g != P:
            raise ValueError(f"upsampling needs a square patch grid, got P={P}")
        u = self.upsample
        c = context.reshape(B * T, g, g, Fd).permute(0, 3, 1, 2)
        c = F.interpolate(c, scale_factor=u, mode="nearest")
        c = c + self.up_conv(c).to(c.dtype)
        c = c.permute(0, 2, 3, 1).reshape(B, T, P * u * u, Fd)
        xyz = patch_xyz.reshape(B * T, g, g, 3).permute(0, 3, 1, 2)
        xyz = F.interpolate(xyz.float(), scale_factor=u, mode="bilinear",
                            align_corners=False).to(patch_xyz.dtype)
        xyz = xyz.permute(0, 2, 3, 1).reshape(B, T, P * u * u, 3)
        return c, xyz

    def _lookup_pos(self, x: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Where each point currently thinks it is, in `patch_xyz`'s space.

        `x` is what gets denoised, and that is not a position: it is divided by
        `traj_scale`, and with a displacement target it is measured from each
        point's own start. `patch_xyz` is an absolute scene-normalised position.
        Every lookup "near the point" has to undo both first.

        Until 2026-10-09 the lookups used `x` as it was. With a displacement
        target that centred them on the scene origin instead of on the point --
        the locality bias and the cost volume were looking in the wrong place
        for every run since run3493_disp. `legacy_lookup` reproduces that, only
        so checkpoints trained that way still evaluate as they were trained.
        """
        if self.legacy_lookup:
            return x
        pos = x * self.traj_scale
        return pos + anchor[:, None] if self.displacement else pos

    def _target_units(self, pos: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """The inverse of `_lookup_pos`: a position in `patch_xyz`'s space,
        expressed the way the model's own target is."""
        if self.legacy_lookup:
            return pos
        if self.displacement:
            pos = pos - anchor[:, None]
        return pos / self.traj_scale

    @staticmethod
    def _patch_spacing(patch_xyz: torch.Tensor) -> torch.Tensor:
        """(B, T, P, 3) -> (B, T): how far apart neighbouring patches are.

        The median distance from a patch to its right-hand neighbour on the
        grid. Median, because a pair straddling a depth edge is metres apart
        and a mean would follow those. Per frame, because the camera moves:
        on two Kubric clips it drifted 10% over the clip, at most 5% between
        frames, so a per-frame value is steady and a per-clip one is stale.
        """
        B, T, P, _ = patch_xyz.shape
        g = math.isqrt(P)
        if g * g != P:
            raise ValueError(f"patch spacing needs a square patch grid, got P={P}")
        grid = patch_xyz.reshape(B, T, g, g, 3)
        step = (grid[:, :, :, 1:] - grid[:, :, :, :-1]).norm(dim=-1).flatten(2)
        # Floored: patches stacked on one spot would otherwise divide by zero.
        return step.median(dim=-1).values.clamp_min(1e-4)

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

        if self.upsample > 1 and context is not None and patch_xyz is not None:
            context, patch_xyz = self._upsample(context, patch_xyz)

        # --- where each point currently thinks it is, relative to every patch ---
        # x and patch_xyz share one normalised space, so this is a plain distance
        # and needs no camera. It is the lookup every established tracker does by
        # projecting into the image and sampling there; in 3D the projection is
        # unnecessary. Computed once, used by the cost volume and by every block.
        #
        # Every position computation in this method is taken out of autocast:
        # bf16 keeps 8 bits of mantissa, so a coordinate near 1 is only known to
        # about 1/256, and autocast ran the einsum below in it.
        dist2 = near2 = None
        with torch.autocast(device_type=dev.type, enabled=False):
            pos = self._lookup_pos(x.float(), anchor.float())
            if context is not None and patch_xyz is not None and (
                    self.locality or self.costvol or self.cosine):
                patch_xyz = patch_xyz.float()
                # ||a-b||^2 = |a|^2 + |b|^2 - 2a.b, rather than materialising the
                # (B, T, N, P, 3) difference -- that tensor is 340 MB at batch 16.
                dist2 = (pos.pow(2).sum(-1)[..., None]
                         + patch_xyz.pow(2).sum(-1)[:, :, None]
                         - 2 * torch.einsum("btnc,btpc->btnp", pos, patch_xyz)).clamp_min(0)
                if self.patch_locality or self.cosine:
                    # The same distances counted in patch spacings, which is
                    # the unit "near" is meant in: a scene-unit distance says
                    # nothing until you know the grid. Measured from the
                    # nearest patch -- a softmax cannot tell the difference,
                    # and it keeps the numbers small where they matter, so a
                    # point far from every patch still sees its closest ones.
                    near2 = dist2 / self._patch_spacing(patch_xyz).pow(2)[..., None, None]
                    near2 = near2 - near2.amin(dim=-1, keepdim=True)

        # --- [cost volume] support window vs the neighbourhood of the estimate ---
        cv = None
        if self.costvol and dist2 is not None:
            cf = self.cv_proj(context)                                # (B, T, P, cv)
            # Support: the patches around where the point STARTS. The 3D
            # equivalent of CoTracker's window around the query pixel, and it
            # replaces a single-vector template that many patches match equally.
            with torch.autocast(device_type=dev.type, enabled=False):
                a0 = anchor.float()
                d0 = (a0.pow(2).sum(-1)[..., None]
                      + patch_xyz[:, 0].pow(2).sum(-1)[:, None]
                      - 2 * torch.einsum("bnc,bpc->bnp", a0, patch_xyz[:, 0]))
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
            with torch.autocast(device_type=dev.type, enabled=False):
                off = patch_xyz.gather(2, flat.expand(-1, -1, -1, 3)) \
                               .reshape(B, T, N, self.cv_k, 3) - pos[..., None, :]

            cost = torch.einsum("bnsc,btnkc->btnsk", sup, nb) * self.cv_dim ** -0.5
            cv = self.cv_mlp(torch.cat([cost.flatten(3), off.flatten(3)], dim=-1))
            if visual_mask is not None:
                cv = cv * visual_mask[..., None, None]

        # --- [correlation] template vs every patch, on the RAW features ---
        # Read before the adapter overwrites `context`. The model previously had
        # the template and the patches but nothing that compared them, so the
        # one route from image to position had to be discovered from position
        # error alone -- while "predict little motion" cut the loss immediately.
        corr = match_xyz = match = None
        if (self.cosine and context is not None and id_card is not None
                and patch_xyz is not None):
            with torch.autocast(device_type=dev.type, enabled=False):
                # fp32: the match is decided by differences in the third
                # decimal of a cosine, which bf16 does not hold.
                feats = context.float()
                q = F.normalize(id_card.float(), dim=-1)
                corr = torch.einsum("bnc,btpc->btnp", q, feats)
                corr = corr / torch.linalg.vector_norm(feats, dim=-1).clamp_min(1e-6)[:, :, None]
                if visual_mask is not None:
                    corr = corr * visual_mask[..., None, None]
                # The best match, softly: a patch has to look like the point
                # AND be near where the point currently is. Appearance alone
                # picks any lookalike in the scene; distance alone is the
                # model's own guess echoed back.
                tau = bounded_exp(self.match_log_tau, 1e-3, 1.0)
                sigma = bounded_exp(self.match_log_sigma, 1e-2, 1e2)
                w = (corr / tau - near2 / (2 * sigma ** 2)).softmax(-1)
                offset = self._target_units(
                    torch.einsum("btnp,btpc->btnc", w, patch_xyz), anchor.float())
                # How well the chosen patches actually look like the point, so
                # an offset from a poor match can be told from a good one.
                match = torch.cat([offset, (w * corr).sum(-1, keepdim=True)], dim=-1)
                if visual_mask is not None:
                    match = match * visual_mask[..., None, None]
        elif (self.correlate and context is not None and id_card is not None
                and patch_xyz is not None):
            q = self.corr_q(id_card)                                  # (B, N, D)
            corr = torch.einsum("bnc,btpc->btnp", q, self.corr_k(context))
            corr = corr * self.dim ** -0.5
            if visual_mask is not None:
                corr = corr * visual_mask[..., None, None]
            # Softly, where in this frame the template matches best. Independent
            # of the diffused path, so it is evidence rather than an echo of the
            # model's own guess -- the one term here that the image alone decides.
            with torch.autocast(device_type=dev.type, enabled=False):
                match_xyz = torch.einsum("btnp,btpc->btnc",
                                         corr.float().softmax(-1), patch_xyz.float())
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
            # The position embedding is computed in fp32 and cast down only
            # here: a second full-size fp32 copy of the context is the memory
            # that `rms_norm` goes out of its way not to spend.
            context = self.frame_proj(context)
            context = context + self.patch_pos(patch_xyz).to(context.dtype)

            # Masked frames lose features AND positions, before any attention.
            if visual_mask is not None:
                context = torch.where(
                    visual_mask[..., None, None], context, self.null_ctx.to(context.dtype)
                )

        # --- the same distances, folded into the attention bias ---
        if self.patch_locality:
            dist2 = near2
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
        if match is not None:
            # Added, not folded into `cond`: conditioning only rescales
            # channels (AdaRMSNorm), and a displacement has to be copied.
            h = h + self.match_proj(match.to(h.dtype))

        # --- positions for RoPE, and the causal mask ---
        # RoPE's ladder runs pi..10*pi rad per unit and is built for positions in
        # [-1, 1], which is what `anchor` gives the spatial one. The raw frame
        # index turned adjacent frames by at least 180 degrees on every channel,
        # so neighbours in time looked no more related than distant frames.
        t_pos = (torch.linspace(-1, 1, T, device=dev, dtype=dt) if self.time_norm
                 else torch.arange(T, device=dev, dtype=dt))
        t_pos = t_pos[None, :, None].expand(B * N, T, 1)
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
