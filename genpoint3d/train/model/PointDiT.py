"""Predict how each point moves, from its current guess and the video.

    tokens        each point's guess per frame, how it moves, and the CNN
                  matcher's comparison of its surroundings with its template
    conditioning  where the point started, the round, its DINOv3 id card and
                  the matcher, one vector per (frame, point)
    blocks        attention along each point's frames, between the points of a
                  frame, and to the DINOv3 patches nearest its guess, x depth
    heads         a correction to each frame's position, and optionally
                  visibility and a group vector per point
"""

import torch
import torch.nn.functional as F
from torch import nn

from genpoint3d.tools.traj_scale import TRAJ_SCALE_DISP
from genpoint3d.train.model.FineFinder import FineFinder, _posenc
from genpoint3d.train.model.tools.layers import (
    Block, CrossBlock, FourierEmbedding, RMSNorm, RoPE, zero_init,
)
from genpoint3d.train.tools.camera import project


class PointDiT(nn.Module):
    """Tokens on a (T frames) x (N points) grid, attention factorised into
    one point across time and all points within one frame."""

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_mult: int = 3,
        feat_dim: int = 384,
        traj_scale: float = TRAJ_SCALE_DISP,
        cross_window: int = 49,
        fine_neigh: int = 32,
        visibility: bool = False,
        group_feedback: int = 0,
        feedback_scale: float = 1.0,
        rope_current: bool = False,
        rigid_proposal: bool = False,
        fit_left: bool = False,
    ) -> None:
        super().__init__()
        self.dim, self.depth = dim, depth
        self.traj_scale = traj_scale
        self.cross_window = cross_window
        head_dim = dim // num_heads

        self.token_proj = nn.Linear(3, dim, bias=False)
        # How the guess moves to the frames either side, and where it is in
        # the image, Fourier-encoded as TAPIP3D's updater reads them.
        self.motion_proj = zero_init(nn.Linear(6 * 33 + 2 * 33, dim, bias=False), almost=False)
        self.frame_proj = nn.Linear(feat_dim, dim, bias=False)
        self.patch_pos = FourierEmbedding(3, dim)
        self.id_feature_proj = nn.Linear(feat_dim, dim, bias=False)

        self.anchor_emb = FourierEmbedding(3, dim)
        self.time_emb = FourierEmbedding(1, dim)
        self.cond_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

        self.temporal_blocks = nn.ModuleList(Block(dim, num_heads, mlp_mult, dim) for _ in range(depth))
        self.spatial_blocks = nn.ModuleList(Block(dim, num_heads, mlp_mult, dim) for _ in range(depth))
        self.time_rope = RoPE(head_dim, num_heads, n_axes=1)
        self.space_rope = RoPE(head_dim, num_heads, n_axes=3)
        # The attention between points encodes each point's current guess
        # rather than where it started.
        self.rope_current = rope_current
        self.cross_blocks = nn.ModuleList(CrossBlock(dim, num_heads, mlp_mult, dim) for _ in range(depth))
        # What a frame with no image shows: "nothing here".
        self.null_ctx = nn.Parameter(torch.randn(dim) * 0.02)

        self.out_norm = RMSNorm(dim)
        # A correction to the point's position in each frame.
        self.place = zero_init(nn.Linear(dim, 3, bias=False))

        self.fine = FineFinder(neigh=fine_neigh)
        self.fine_cond = zero_init(nn.Linear(self.fine.out_dim, dim), almost=False)
        # Gravity's strength in the clip's own unit, always told, and which
        # way is down, its own vector when the clip does not say.
        self.gravity_in = zero_init(nn.Linear(1, dim, bias=False), almost=False)
        self.down_in = zero_init(nn.Linear(3, dim, bias=False), almost=False)
        self.down_unknown = nn.Parameter(torch.zeros(dim))
        # Which way is down, said for the clip from all its points and frames:
        # in place of `down` next call where the clip does not say.
        self.down_out = nn.Linear(dim, 3)
        self.fine_tok = zero_init(nn.Linear(self.fine.out_dim, dim, bias=False), almost=False)

        # One logit per (point, frame): can the point be seen there? What it
        # last said is fed back, so a point that looks hidden can stop
        # trusting the image there, and look again.
        self.visibility = visibility
        if visibility:
            self.vis_weight = nn.Parameter(torch.zeros(dim))
            self.vis_bias = nn.Parameter(torch.zeros(()))
            self.vis_in = nn.Parameter(torch.zeros(dim))
        # Whether a frame is before the point's query frame, or is it: a fact,
        # beside the head's belief.
        self.query_in = nn.Parameter(torch.zeros(2, dim))
        # Each point says a vector per window, close for points on one rigid
        # piece; the next call reads it in the token and as an attention bias
        # between points, scaled by feedback_scale so Adam moves it in time.
        self.group_feedback = group_feedback
        self.feedback_scale = feedback_scale
        if group_feedback:
            self.group_out = nn.Linear(dim, group_feedback)
            self.group_in = zero_init(nn.Linear(group_feedback, dim, bias=False), almost=False)
            self.group_bias = nn.Parameter(torch.zeros(depth, num_heads))
        # Where moving with its piece would put each point (`regress.rigid_proposal`):
        # the offset from its guess, Fourier-encoded, and how much backs it.
        if rigid_proposal and not group_feedback:
            raise ValueError("the rigid proposal comes from the group vectors: needs group_feedback")
        self.rigid_proposal = rigid_proposal
        if rigid_proposal:
            # With `fit_left`, also how far the mates sit from the fit.
            self.fit_left = fit_left
            self.rigid_in = zero_init(nn.Linear(99 + 1 + fit_left, dim, bias=False), almost=False)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def lookup_pos(self, x: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """The guess `x` (a displacement over traj_scale) as a position in
        `anchor`'s space."""
        return x * self.traj_scale + anchor[:, None]

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        anchor: torch.Tensor,
        context: torch.Tensor,
        id_card: torch.Tensor,
        patch_xyz: torch.Tensor,
        fine_map: torch.Tensor,
        fine_proj: torch.Tensor,
        fine_xyz: torch.Tensor,
        query_t: torch.Tensor,
        gravity: torch.Tensor,
        down: torch.Tensor,
        down_known: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
        alive: torch.Tensor | None = None,
        vis_prev: torch.Tensor | None = None,
        group_prev: torch.Tensor | None = None,
        group_known: torch.Tensor | None = None,
        frames: tuple[int, int] | None = None,
        x_before: torch.Tensor | None = None,
        rigid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        x            (B, T, N, 3)    the guess, displacement / traj_scale
        k            (B,)            the round clock in [0, 1]
        anchor       (B, N, 3)       each point at its query frame, scene-normalised
        context      (B, Tc, P, F)   DINOv3 patches of the whole clip
        id_card      (B, N, F)       DINOv3 feature at each query
        patch_xyz    (B, Tc, P, 3)   each patch's 3D position, anchor's space
        fine_map     (B, Tc, C, h, w) the CNN's maps (`FineFinder.encode`)
        fine_proj    (B, Tc, 3, 4)   anchor's space -> the image, per frame
        fine_xyz     (B, Tc, 3, h, w) each CNN cell's 3D position
        query_t      (B, N) long     each point's query frame in the clip
        gravity      (B,)            its strength in the clip's unit per frame squared, a typical clip's 1
        down         (B, 3)          which way is down, a unit vector in anchor's space
        down_known   (B,) bool       the clips whose `down` is told; the rest are unknown
        visual_mask  (B, Tc, 2, 2) bool, which quarters of each frame are seen:
                     a frame with any quarter hidden has no DINOv3, and the
                     matcher reads only its seen cells (`frames` blacked out)
        alive        (B, N) bool     points the others may attend to
        vis_prev     (B, T, N)       the visibility logits last said (with the head)
        group_prev   (B, N, E)       the group vectors the last call said
        group_known  (B, N) bool     the points that have said one (None: all)
        frames       (start, end)    the clip frames `x` covers; None = all
        x_before     (B, 1, N, 3)    the guess in the clip frame before `frames`
        rigid        (B, T, N, 5)    where moving with its piece would put each point,
                                     as an offset from its guess, its support and its
                                     mates' leftover error (`regress.rigid_proposal`)

        returns a correction to each frame's position (B, T, N, 3), the visibility logits (B, T, N),
        the group vectors (B, N, E) (those two None without their heads) and which way is
        down (B, 3), a unit vector.
        """
        B, T, N, _ = x.shape
        dev, dt = x.device, x.dtype
        start = 0 if frames is None else frames[0]
        if visual_mask is None:
            visual_mask = torch.ones(B, fine_map.shape[1], 2, 2, dtype=torch.bool, device=dev)
        # The templates and the frame before are read in the whole clip, before the window cut.
        clip_map, clip_proj, clip_xyz, clip_mask = fine_map, fine_proj, fine_xyz, visual_mask
        if frames is not None:
            cut = slice(*frames)
            context, visual_mask, patch_xyz, fine_map, fine_proj, fine_xyz = (
                None if t is None else t[:, cut]
                for t in (context, visual_mask, patch_xyz, fine_map, fine_proj, fine_xyz))
        # A frame is whole (all of it seen: DINOv3 too), partial (the matcher's
        # seen cells only) or blank.
        whole, some = visual_mask.flatten(2).all(-1), visual_mask.flatten(2).any(-1)      # (B, T)

        # Positions in fp32: bf16 holds a coordinate near 1 to about 1/256.
        with torch.autocast(device_type=dev.type, enabled=False):
            pos = self.lookup_pos(x.float(), anchor.float())
            patch_xyz = patch_xyz.float()
            # |a-b|^2 = |a|^2 + |b|^2 - 2ab, without the (B, T, N, P, 3) difference.
            dist2 = (pos.pow(2).sum(-1)[..., None]
                     + patch_xyz.pow(2).sum(-1)[:, :, None]
                     - 2 * torch.einsum("btnc,btpc->btnp", pos, patch_xyz)).clamp_min(0)
            uv, _ = project(pos, fine_proj.float())

        # Each point reads only the patches nearest its guess; in a frame
        # without DINOv3 it reads them all (they are all null there).
        d = dist2.reshape(B * T, N, -1)
        near = d.topk(self.cross_window, dim=-1, largest=False).indices
        window = torch.full_like(d, -CrossBlock.FAR).scatter(-1, near, 0.0)
        window = window * whole.reshape(B * T, 1, 1)

        # The matcher's second comparison reads each frame and the frame
        # before (the clip's, before the window's first): off unless both
        # have something seen.
        if start > 0:
            b_map, b_proj, b_xyz, b_view = (t[:, start - 1:start] for t in (clip_map, clip_proj, clip_xyz, clip_mask))
            with torch.autocast(device_type=dev.type, enabled=False):
                pos_b = self.lookup_pos(x_before.float(), anchor.float())
        else:
            b_map, b_proj, b_xyz, pos_b = fine_map[:, :1], fine_proj[:, :1], fine_xyz[:, :1], pos[:, :1]
            b_view = torch.zeros_like(visual_mask[:, :1])
        with torch.autocast(device_type=dev.type, enabled=False):
            uv_b, _ = project(pos_b, b_proj.float())
        fine = self._fine_at_queries(torch.cat([b_map, fine_map], 1), torch.cat([b_xyz, fine_xyz], 1),
                                     clip_map, clip_proj, clip_xyz, torch.cat([uv_b, uv], 1),
                                     torch.cat([pos_b, pos], 1), anchor, query_t,
                                     torch.cat([b_view, visual_mask], 1), clip_mask)
        half = self.fine.out_dim // 2
        prev_ok = torch.cat([b_view.flatten(2).any(-1), some[:, :-1]], 1) & some
        fine = torch.cat([fine[..., :half] * some[..., None, None], fine[..., half:] * prev_ok[..., None, None]], -1)

        cond = self.time_emb(k[:, None, None].expand(B, T, 1))[:, :, None] + self.anchor_emb(anchor)[:, None]
        cond = cond + self.id_feature_proj(id_card)[:, None]
        cond = cond + self.fine_cond(fine).to(cond.dtype)
        told = self.gravity_in(gravity[:, None].to(cond.dtype)) + torch.where(
            down_known[:, None], self.down_in(down.to(cond.dtype)), self.down_unknown.to(cond.dtype))
        cond = cond + told[:, None, None]
        cond = self.cond_mlp(cond)                                    # (B, T, N, C)

        context = self.frame_proj(context)
        context = context + self.patch_pos(patch_xyz).to(context.dtype)
        context = torch.where(whole[..., None, None], context, self.null_ctx.to(context.dtype))

        h = self.token_proj(x)                                        # (B, T, N, D)
        with torch.autocast(device_type=dev.type, enabled=False):
            step = pos.diff(dim=1)
            fwd = torch.cat([-step, torch.zeros_like(pos[:, :1])], dim=1)
            bwd = torch.cat([torch.zeros_like(pos[:, :1]), step], dim=1)
            # Where the guess is in the image, from the frame's camera: none without an image.
            seen = _posenc(uv) * some[..., None, None]
            motion = torch.cat([_posenc(torch.cat([fwd, bwd], -1)), seen], -1)
        h = h + self.motion_proj(motion.to(h.dtype))
        h = h + self.fine_tok(fine).to(h.dtype)
        if self.rigid_proposal != (rigid is not None) and not (self.rigid_proposal and group_prev is None):
            raise ValueError("rigid is given exactly to a model built with rigid_proposal, once it has groups")
        if rigid is not None:
            with torch.autocast(device_type=dev.type, enabled=False):
                r = torch.cat([_posenc(rigid[..., :3].float().clamp(-2, 2)),
                               rigid[..., 3:5 if self.fit_left else 4].float()], -1)
            h = h + self.rigid_in(r.to(h.dtype))
        if self.visibility != (vis_prev is not None):
            raise ValueError("vis_prev is given exactly to a model with the visibility head")
        if vis_prev is not None:
            # As a probability less a half: 0 before the head has said anything.
            belief = torch.sigmoid(vis_prev.float()) - 0.5
            h = h + (self.feedback_scale * belief[..., None].to(h.dtype)) * self.vis_in.to(h.dtype)
        # Before the point was first seen, and its query frame.
        t = torch.arange(start, start + T, device=dev)[None, :, None]
        q = query_t[:, None]
        h = h + ((t < q)[..., None] * self.query_in[0] + (t == q)[..., None] * self.query_in[1]).to(h.dtype)

        biases = [None] * self.depth
        if group_prev is not None and not self.group_feedback:
            raise ValueError("group_prev is fed back to a model built with group_feedback")
        if self.group_feedback:
            v = (group_prev.float() if group_prev is not None
                 else torch.zeros(B, N, self.group_feedback, device=dev))
            h = h + self.feedback_scale * self.group_in(v.to(h.dtype))[:, None]
            a = 2 * torch.sigmoid(-_sq_dist(v))                                  # (B, N, N)
            if group_known is not None:
                # A point with no vector yet is biased towards nobody.
                a = a * (group_known[:, :, None] & group_known[:, None, :])
            a = a[:, None, None].expand(B, T, 1, N, N).reshape(B * T, 1, N, N)
            biases = [(a * self.feedback_scale * self.group_bias[d][:, None, None]).to(h.dtype)
                      for d in range(self.depth)]
        if alive is not None:
            # Points not in this window are read by no point.
            gone = ((~alive).to(h.dtype) * -1e4)[:, None, None, None, :]           # (B, 1, 1, 1, N)
            gone = gone.expand(B, T, 1, 1, N).reshape(B * T, 1, 1, N)
            biases = [gone if b is None else b + gone for b in biases]

        # Frame times in [-1, 1], the range RoPE's ladder is built for.
        t_pos = torch.linspace(-1, 1, T, device=dev, dtype=dt)[None, :, None].expand(B * N, T, 1)
        theta_time = self.time_rope(t_pos)                             # (B*N, H, T, d)
        if self.rope_current:
            theta_space = self.space_rope(pos.reshape(B * T, N, 3).to(dt))
        else:
            theta_space = self.space_rope(anchor.repeat_interleave(T, 0))  # (B*T, H, N, d)

        for temporal, spatial, cross, bias in zip(self.temporal_blocks, self.spatial_blocks,
                                                  self.cross_blocks, biases):
            ht = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
            ct = cond.permute(0, 2, 1, 3).reshape(B * N, T, -1)
            h = temporal(ht, ct, theta=theta_time)
            h = h.view(B, N, T, -1).permute(0, 2, 1, 3)

            hs = h.reshape(B * T, N, -1)
            cs = cond.reshape(B * T, N, -1)
            h = spatial(hs, cs, theta=theta_space, attn_mask=bias)

            h = cross(h, cs, context.reshape(B * T, -1, self.dim), window=window).view(B, T, N, -1)

        h = self.out_norm(h)
        logits = (F.linear(h, self.vis_weight[None], self.vis_bias[None])[..., 0]
                  if self.visibility else None)
        # One group vector per point for the window.
        groups = self.group_out(h).float().mean(1) if self.group_feedback else None
        # Down from the points in the window, every frame.
        here = (alive if alive is not None else torch.ones(B, N, dtype=torch.bool, device=dev)).float()
        pooled = (h.float().mean(1) * here[..., None]).sum(1) / here.sum(1, keepdim=True).clamp_min(1)
        down_said = F.normalize(self.down_out(pooled).float(), dim=-1)
        return self.place(h), logits, groups, down_said

    def _fine_at_queries(self, fine_map, fine_xyz, clip_map, clip_proj, clip_xyz, uv, pos, anchor,
                         query_t, view, clip_view):
        """The matcher's comparison (B, T, N, out_dim): each point's
        neighbourhood at its guess against its template in its query frame,
        on seen cells only."""
        h, w = fine_map.shape[-2:]
        with torch.autocast(device_type=uv.device.type, enabled=False):
            proj_q = clip_proj.float().gather(1, query_t[..., None, None].expand(-1, -1, 3, 4))
            uv0 = project(anchor.float()[:, :, None], proj_q)[0][:, :, 0]              # (B, N, 2)
            places = dict(xyz=fine_xyz.float(), pos=pos, pos0=anchor.float(), xyz0=clip_xyz.float())
        return self.fine(fine_map, clip_map, uv, uv0, query_t=query_t, ok=cells(view, h, w),
                         ok0=cells(clip_view, h, w), **places)


def cells(view: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, T, 2, 2) seen quarters -> (B, T, h, w) seen cells (h, w even)."""
    return view.repeat_interleave(h // 2, 2).repeat_interleave(w // 2, 3)


def _sq_dist(v: torch.Tensor) -> torch.Tensor:
    """(B, N, E) -> (B, N, N) squared distances, written out: cdist's square
    root has no gradient where two vectors meet."""
    return (v[:, :, None] - v[:, None]).pow(2).sum(-1)
