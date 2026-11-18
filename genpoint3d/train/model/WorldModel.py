"""Put the stored world right from one frame, and say what the engine cannot know.

    tokens        each point the world has: where the engine put it, how it
                  moved since the frame before, and the CNN matcher's
                  comparison of its surroundings with its template
    conditioning  where the point was given, the round, its DINOv3 id card and
                  the matcher, one vector per point
    blocks        attention between the points, and to the DINOv3 patches
                  nearest each, x depth
    heads         per point: a correction to its place, whether it is seen, a
                  group vector (which body it is on), an arrow to its body's
                  centre of gravity, how stiffly its body holds it, how free
                  its body is to move and how heavy; per pair of bodies that
                  may touch: where, how far they stick there, and a push
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from genpoint3d.tools.traj_scale import TRAJ_SCALE_DISP
from genpoint3d.train.model.FineFinder import FineFinder, _posenc
from genpoint3d.train.model.PointDiT import _cells, _in_patches, _point_bias
from genpoint3d.train.model.tools.layers import Block, CrossBlock, FourierEmbedding, RMSNorm, RoPE, zero_init
from genpoint3d.train.tools.batch import whole_frames
from genpoint3d.train.tools.camera import project
from genpoint3d.train.tools.engine import Touch


@dataclass
class Said:
    """What the model says of one frame's points (B, N, ...)."""
    step: torch.Tensor      # (B, N, 3) a correction to each point's place, in the guess's units
    logits: torch.Tensor    # (B, N) is the point seen
    groups: torch.Tensor    # (B, N, E) its group vector
    down: torch.Tensor      # (B, 3) which way is down, a unit vector
    arrow: torch.Tensor     # (B, N, 3) from the guess to its body's centre of gravity, in the guess's units
    stiff: torch.Tensor     # (B, N) how stiffly its body holds it: 1 a rigid body, 0 not at all
    free: torch.Tensor      # (B, N) how free its body is to move: 0 the fixed world
    heavy: torch.Tensor     # (B, N) its body's mass, beside the others'
    h: torch.Tensor         # (B, N, D) what it knows of the point, for `touches`


class WorldModel(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 6, num_heads: int = 4, mlp_mult: int = 3, feat_dim: int = 384,
                 traj_scale: float = TRAJ_SCALE_DISP, cross_window: int = 49, fine_neigh: int = 32,
                 group_dim: int = 16, feedback_scale: float = 10.0, patch: int = 0) -> None:
        super().__init__()
        self.dim, self.depth = dim, depth
        self.traj_scale = traj_scale
        self.cross_window = cross_window
        head_dim = dim // num_heads

        self.token_proj = nn.Linear(3, dim, bias=False)
        # How the guess moved since the frame before, and where it is in the image, Fourier-encoded.
        self.motion_proj = zero_init(nn.Linear(5 * 33, dim, bias=False), almost=False)
        self.frame_proj = nn.Linear(feat_dim, dim, bias=False)
        self.patch_pos = FourierEmbedding(3, dim)
        self.id_feature_proj = nn.Linear(feat_dim, dim, bias=False)
        self.anchor_emb = FourierEmbedding(3, dim)
        self.time_emb = FourierEmbedding(1, dim)
        self.cond_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

        self.spatial_blocks = nn.ModuleList(Block(dim, num_heads, mlp_mult, dim) for _ in range(depth))
        self.space_rope = RoPE(head_dim, num_heads, n_axes=3)
        self.cross_blocks = nn.ModuleList(CrossBlock(dim, num_heads, mlp_mult, dim) for _ in range(depth))
        # Attention between points only within patches of `patch` points near each other (Point Transformer
        # V3's serialized attention); 0: every point reads every other.
        self.patch = patch
        # What a frame with no image shows: "nothing here".
        self.null_ctx = nn.Parameter(torch.randn(dim) * 0.02)
        self.out_norm = RMSNorm(dim)

        self.fine = FineFinder(neigh=fine_neigh)
        self.fine_cond = zero_init(nn.Linear(self.fine.out_dim, dim), almost=False)
        self.fine_tok = zero_init(nn.Linear(self.fine.out_dim, dim, bias=False), almost=False)
        # Gravity's strength in the clip's own unit, always told, and which way is down, its own vector when
        # the clip does not say; said back for the clips that do not.
        self.gravity_in = zero_init(nn.Linear(1, dim, bias=False), almost=False)
        self.down_in = zero_init(nn.Linear(3, dim, bias=False), almost=False)
        self.down_unknown = nn.Parameter(torch.zeros(dim))
        self.down_out = nn.Linear(dim, 3)
        # What the point was last said to be, fed back: seen or not, and its group vector (also a bias of the
        # attention between points, toward its own body's), scaled by feedback_scale so Adam moves them in time.
        self.feedback_scale = feedback_scale
        self.vis_in = nn.Parameter(torch.zeros(dim))
        self.group_in = zero_init(nn.Linear(group_dim, dim, bias=False), almost=False)
        self.group_bias = nn.Parameter(torch.zeros(depth, num_heads))
        # The frame the point is given in: there only its depth is not known.
        self.query_in = nn.Parameter(torch.zeros(dim))
        # From the guess to the place that keeps its body's shape (`World.rigid`), and how many seen points back it.
        self.rigid_in = zero_init(nn.Linear(99 + 1, dim, bias=False), almost=False)

        self.place = zero_init(nn.Linear(dim, 3, bias=False))
        self.vis_out = zero_init(nn.Linear(dim, 1), almost=False)
        self.group_out = nn.Linear(dim, group_dim)
        self.centre_out = zero_init(nn.Linear(dim, 3, bias=False))
        self.stiff = zero_init(nn.Linear(dim, 1, bias=False), almost=False)
        self.body_out = zero_init(nn.Linear(dim, 2), almost=False)
        # A pair of points that may be where two bodies touch: how much it is, and what it says of the touch;
        # then for the two bodies, how far they stick there and a push.
        self.pair = nn.Sequential(nn.Linear(2 * dim + 3 * 99, dim), nn.GELU(), nn.Linear(dim, dim + 1))
        self.touch_out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), zero_init(nn.Linear(dim, 4), almost=False))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def lookup_pos(self, x: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """The guess `x` (..., N, 3), a displacement over traj_scale, as a position in `anchor`'s (B, N, 3) space."""
        return x * self.traj_scale + (anchor if x.dim() == anchor.dim() else anchor[:, None])

    def forward(self, x: torch.Tensor, before: torch.Tensor, k: torch.Tensor, anchor: torch.Tensor, t: int,
                context: torch.Tensor, id_card: torch.Tensor, patch_xyz: torch.Tensor, fine_map: torch.Tensor,
                fine_proj: torch.Tensor, fine_xyz: torch.Tensor, query_t: torch.Tensor, gravity: torch.Tensor,
                down: torch.Tensor, down_known: torch.Tensor, visual_mask: torch.Tensor, vis_prev: torch.Tensor,
                group_prev: torch.Tensor, group_known: torch.Tensor, rigid: torch.Tensor) -> Said:
        """
        x            (B, N, 3)       the guess in frame `t`, displacement / traj_scale
        before       (B, N, 3)       the guess in the frame before
        k            (B,)            the round clock in [0, 1]
        anchor       (B, N, 3)       each point where it is given, scene-normalised
        context      (B, Tc, P, F)   DINOv3 patches of the whole clip
        id_card      (B, N, F)       DINOv3 feature where each point is given
        patch_xyz    (B, Tc, P, 3)   each patch's 3D position, anchor's space
        fine_map     (B, Tc, C, h, w) the CNN's maps (`FineFinder.encode`)
        fine_proj    (B, Tc, 3, 4)   anchor's space -> the image, per frame
        fine_xyz     (B, Tc, 3, h, w) each CNN cell's 3D position
        query_t      (B, N) long     the frame each point is given in
        gravity      (B,)            its strength in the clip's unit per frame squared, a typical clip's 1
        down         (B, 3)          which way is down, a unit vector in anchor's space
        down_known   (B,) bool       the clips whose `down` is told
        visual_mask  (B, Tc, 2, 2) bool, which quarters of each frame are seen: a frame with any quarter hidden
                     has no DINOv3, and the matcher reads only its seen cells
        vis_prev     (B, N)          the visibility logit last said of each point
        group_prev   (B, N, E)       the group vector last said, for the points `group_known` (B, N)
        rigid        (B, N, 4)       from the guess to the place that keeps its body's shape, and its support
        """
        B, N, _ = x.shape
        dev = x.device
        tb = max(t - 1, 0)
        view = visual_mask[:, t:t + 1]                                                    # (B, 1, 2, 2)
        b_view = visual_mask[:, tb:tb + 1] if t else torch.zeros_like(view)
        whole, some = whole_frames(view)[:, 0], view.flatten(1).any(-1)                   # (B,)
        both = some & b_view.flatten(1).any(-1)

        # Positions in fp32: bf16 holds a coordinate near 1 to about 1/256.
        with torch.autocast(device_type=dev.type, enabled=False):
            pos, pos_b = self.lookup_pos(x.float(), anchor.float()), self.lookup_pos(before.float(), anchor.float())
            pxyz = patch_xyz[:, t].float()
            dist2 = torch.cdist(pos, pxyz).pow(2)                                         # (B, N, P)
            uv, _ = project(pos[:, None], fine_proj[:, t:t + 1].float())                  # (B, 1, N, 2)
            uv_b, _ = project(pos_b[:, None], fine_proj[:, tb:tb + 1].float())
            proj_q = fine_proj.float().gather(1, query_t[..., None, None].expand(-1, -1, 3, 4))
            uv0 = project(anchor.float()[:, :, None], proj_q)[0][:, :, 0]                 # (B, N, 2)

        # Each point reads only the patches nearest its guess; in a frame without DINOv3 it reads them all
        # (they are all null there).
        near = dist2.topk(self.cross_window, dim=-1, largest=False).indices
        window = torch.full_like(dist2, -CrossBlock.FAR).scatter(-1, near, 0.0) * whole[:, None, None]

        # The matcher: each point's surroundings at its guess against its template where it was given, and
        # against the frame before; on seen cells only, off where the frame (or the one before) shows nothing.
        hh, ww = fine_map.shape[-2:]
        two = lambda a, b: torch.cat([a, b], 1)
        fine = self.fine(two(fine_map[:, tb:tb + 1], fine_map[:, t:t + 1]), fine_map, two(uv_b, uv), uv0,
                         xyz=two(fine_xyz[:, tb:tb + 1], fine_xyz[:, t:t + 1]).float(), xyz0=fine_xyz.float(),
                         pos=two(pos_b[:, None], pos[:, None]), pos0=anchor.float(), query_t=query_t,
                         ok=_cells(two(b_view, view), hh, ww), ok0=_cells(visual_mask, hh, ww))[:, 0]
        half = self.fine.out_dim // 2
        fine = torch.cat([fine[..., :half] * some[:, None, None], fine[..., half:] * both[:, None, None]], -1)

        cond = self.time_emb(k[:, None, None].expand(B, 1, 1)) + self.anchor_emb(anchor)
        cond = cond + self.id_feature_proj(id_card) + self.fine_cond(fine).to(cond.dtype)
        told = self.gravity_in(gravity[:, None].to(cond.dtype)) + torch.where(
            down_known[:, None], self.down_in(down.to(cond.dtype)), self.down_unknown.to(cond.dtype))
        cond = self.cond_mlp(cond + told[:, None])                                        # (B, N, C)

        ctx = self.frame_proj(context[:, t])
        ctx = ctx + self.patch_pos(pxyz).to(ctx.dtype)
        ctx = torch.where(whole[:, None, None], ctx, self.null_ctx.to(ctx.dtype))

        h = self.token_proj(x)
        with torch.autocast(device_type=dev.type, enabled=False):
            motion = torch.cat([_posenc(pos - pos_b), _posenc(uv[:, 0]) * some[:, None, None]], -1)
            r = torch.cat([_posenc(rigid[..., :3].float().clamp(-2, 2)), rigid[..., 3:4].float()], -1)
            # As a probability less a half: 0 before anything is said.
            belief = torch.sigmoid(vis_prev.float()) - 0.5
        h = h + self.motion_proj(motion.to(h.dtype)) + self.fine_tok(fine).to(h.dtype) + self.rigid_in(r.to(h.dtype))
        h = h + (self.feedback_scale * belief[..., None].to(h.dtype)) * self.vis_in.to(h.dtype)
        h = h + (query_t == t)[..., None] * self.query_in.to(h.dtype)
        v = group_prev.float() * group_known[..., None]
        h = h + self.feedback_scale * self.group_in(v.to(h.dtype))

        every = torch.ones(B, N, dtype=torch.bool, device=dev)
        theta = self.space_rope(pos.to(x.dtype))                                          # (B, H, N, d)
        for d, (spatial, cross) in enumerate(zip(self.spatial_blocks, self.cross_blocks)):
            scale = self.feedback_scale * self.group_bias[d]
            if self.patch:
                h = _in_patches(spatial, h, cond, theta, pos, v, group_known, every, scale, self.patch, d)
            else:
                h = spatial(h, cond, theta=theta, attn_mask=_point_bias(v, group_known, every, scale, h.dtype))
            h = cross(h, cond, ctx, window=window)

        h = self.out_norm(h)
        body = self.body_out(h).float()
        return Said(step=self.place(h), logits=self.vis_out(h)[..., 0], groups=self.group_out(h).float(),
                    down=F.normalize(self.down_out(h.float().mean(1)), dim=-1), arrow=self.centre_out(h),
                    stiff=self.stiff(h)[..., 0].float(), free=torch.sigmoid(body[..., 0]), heavy=torch.exp(body[..., 1]),
                    h=h)

    def touches(self, h: torch.Tensor, n: torch.Tensor, m: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                apart: torch.Tensor, slide: torch.Tensor, at: torch.Tensor, fall: torch.Tensor) -> Optional[Touch]:
        """Where bodies touch, from the pairs of points that may be it: points `n` and `m` (C,) of `h` (N, D),
        on bodies `a` and `b` (C,), `apart` (C, 3) from one to the other, the two bodies going `slide` (C, 3)
        past each other there, at the place `at` (C, 3); `fall` (3,) gravity a frame squared. For each two
        bodies, one touch: the place its pairs say, each by how much it counts, how far the two stick there,
        and a push, in gravity's size. None with no pair."""
        if not len(n):
            return None
        with torch.autocast(device_type=h.device.type, enabled=False):
            where = torch.cat([_posenc(apart.float()), _posenc(slide.float()), _posenc(fall.float().expand_as(apart))], -1)
        out = self.pair(torch.cat([h[n], h[m], where.to(h.dtype)], -1)).float()                  # (C, D + 1)
        two, which = torch.unique(torch.stack([a, b], 1), dim=0, return_inverse=True)            # (P, 2), (C,)
        P = len(two)
        top = torch.full((P,), -float("inf"), device=h.device).scatter_reduce(0, which, out[:, 0], "amax")
        w = torch.exp(out[:, 0] - top[which])
        w = w / torch.zeros(P, device=h.device).index_add(0, which, w)[which]
        pooled = torch.zeros(P, out.shape[1] - 1, device=h.device).index_add(0, which, w[:, None] * out[:, 1:])
        place = torch.zeros(P, 3, device=h.device).index_add(0, which, w[:, None] * at.float())
        said = self.touch_out(pooled.to(h.dtype)).float()
        return Touch(two[:, 0], two[:, 1], place, torch.sigmoid(said[:, 0]), said[:, 1:] * fall.float().norm())
