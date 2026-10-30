"""Find, to the pixel, where a point went near its current guess.

The relative finder -- "where, within a few pixels of my guess, is the thing
that looked like THIS at the query?"

DINOv3 gives one feature per 16 px patch, and nothing finer ever reached the
model: it found the right patch for 90% of moving points and the right pixel
for almost none. Every reference tracker gets its precision the same way, and
this is that way, beside DINOv3 rather than instead of it:

    BasicEncoder   a small CNN on the real RGB frames, 128-d, one cell per 4 px,
                   trained with the tracker (RAFT's encoder as CoTracker ships
                   it; ported from TAPIP3D `models/utils/cotracker_blocks.py`)
    pyramid        that map average-pooled 2x per level, so the same 7x7 window
                   covers 28, 56, 112, 224 px
    correlation    a 7x7 window around the current guess in every frame, against
                   a 7x7 support window around the query point in frame 0:
                   49 x 49 numbers per level, through an MLP
    places         optionally (`xyz`), where each cell of that window is in 3D,
                   from the depth map, as an offset from the current guess:
                   49 x 3 more numbers per level into the same MLP (TAPIP3D
                   gives every cell its 3D position the same way)

It needs the current guess as a PIXEL, which the rest of the model never does:
`image_projection` and `project` (`tools/camera.py`) are the camera, folded into one matrix per
frame so the model itself stays ignorant of intrinsics and normalisation.

Coordinates, once, because half a cell here is the precision this exists for.
A frame is `u` in [0, 1] across, left edge to right edge, however it was
resized; pixel i of a W-wide frame covers `[i, i+1) / W`. The intrinsics use
the same convention (principal point W / 2), and Kubric's own 2D tracks agree
with it to 4e-4 px. The CNN halves the frame with stride-2 convolutions padded
to stay centred, so cell j of its stride-4 map is centred on pixel INDEX 4j,
which is `(4j + 0.5) / W` -- not on the middle of a 4 px block. Hence

    cell = u * w - 1/8            w = cells across, level 0

and average-pooling twice as coarse puts cell j of level l on level-0 cell
`2^l * j + (2^l - 1) / 2`. CoTracker divides by `2^l` and leaves the second
term out; it costs them nothing because both windows are off by the same
amount, and it is kept here because the guess and the query are not in the
same frame of a moving camera.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from genpoint3d.train.model.BasicEncoder import STRIDE, BasicEncoder


class FineFinder(nn.Module):
    """The CNN, the pyramid and the windowed correlation.

    Memory is what shapes this. At batch 16, 24 frames, 384 px and 256 points:

      the CNN       its activations for all 384 frames are ~40 GB with a graph,
                    so it never has one for more than a clip. `encode` runs it
                    without, and `replay` runs it again a clip at a time to
                    carry the gradient that the correlation left on the map.
      the windows   49 x 128 numbers per (point, frame, level) is 2.5 GB a
                    level in fp32, 10 GB for four, held until backward. The
                    correlation is checkpointed instead: as many clips as hold
                    `chunk` points (at least one) and one
                    level at a time (0.3 GB of windows), only its
                    (B, T, N, out_dim) result is kept, and the windows are
                    rebuilt on the way back.

    The map is also what the method's several forward passes share. It reaches
    the model as a leaf with no graph behind it, so each pass can be
    back-propagated on its own and the CNN's gradient simply adds up there.
    """

    def __init__(self, levels: int = 4, radius: int = 3, feat_dim: int = 128,
                 hidden: int = 256, width: int = 64, chunk: int = 256,
                 xyz: bool = False, waves: int = 0, neigh: int = 0, prev: bool = False) -> None:
        super().__init__()
        self.levels, self.radius, self.chunk = levels, radius, chunk
        # TAPIP3D's comparison keeps the features' own width, 128 a level.
        width = feat_dim if neigh else width
        self.out_dim = levels * width
        self.fnet = BasicEncoder(feat_dim)
        k = (2 * radius + 1) ** 2
        # One MLP for every level, as in CoTracker3: the table means the same
        # thing at each, only the size of a cell differs.
        self.corr_mlp = nn.Sequential(nn.Linear(k * k, hidden), nn.GELU(),
                                      nn.Linear(hidden, width))
        r = torch.arange(-radius, radius + 1, dtype=torch.float32)
        dy, dx = torch.meshgrid(r, r, indexing="ij")
        self.register_buffer("delta", torch.stack([dx, dy], dim=-1).reshape(k, 2),
                             persistent=False)
        # The window's 3D offsets join the table at the MLP's first layer:
        # these are that layer's extra columns, kept apart so the first
        # layer keeps its name and shape. Zeros, made without drawing from
        # the random generator, so with them the finder starts as it is
        # without, and every weight after it is drawn as it is without.
        self.xyz = xyz
        if xyz:
            self.xyz_in = nn.Parameter(torch.zeros(hidden, k * 3))
        # With `waves`, each offset is also written as sin and cos at `waves`
        # frequencies, doubling from 1/2 per unit: a ruler with many hands,
        # TAPIP3D's encoding of each neighbour's offset (2^-2 to 2^14 there).
        # A raw offset tells 0.013 from 0.015 by its third decimal; a fast
        # hand turns that into a large change. Zeros, like `xyz_in`.
        self.waves = waves
        if waves:
            if not xyz:
                raise ValueError("waves encode the window's 3D offsets, which need xyz")
            self.register_buffer("freqs", 2.0 ** torch.arange(-1, waves - 1, dtype=torch.float32),
                                 persistent=False)
            self.wave_in = nn.Parameter(torch.zeros(hidden, k * 3 * 2 * waves))
        # With `neigh` = K, TAPIP3D's lookup replaces the 7x7 window (see
        # `_neighbourhoods`). Built last, so the weights above draw what they draw
        # without it; `corr_mlp` is then unused but kept, so the module has one
        # shape whatever the lookup.
        self.neigh = neigh
        if neigh:
            if not xyz:
                raise ValueError("the 3D neighbourhood needs each cell's 3D position (xyz)")
            if waves:
                raise ValueError("the 3D neighbourhood encodes its own offsets; waves are the window's")
            # TAPIP3D's names and shapes (`corr_processor.*` in its checkpoint),
            # so its trained weights load as they are.
            self.in_norm = nn.LayerNorm(feat_dim)
            self.posenc_mlps = nn.ModuleList(_posenc_mlp(width) for _ in range(levels))
            self.transformers = nn.ModuleList(_NeighborTransformer(width) for _ in range(levels))
        # With `prev`, the guess's neighbourhood is also compared against the
        # one at the guess in the frame before, by a second comparison of the
        # same shape: a surface that has turned since frame 0 has turned only
        # a little since the last frame. Built after everything else, so the
        # weights above draw what they draw without it; `copy_prev` starts it
        # as the frame-0 comparison once that has its weights.
        self.prev = prev
        if prev:
            if not neigh:
                raise ValueError("the frame-before comparison is the 3D neighbourhood's (neigh)")
            self.prev_transformers = nn.ModuleList(_NeighborTransformer(width) for _ in range(levels))
            self.out_dim *= 2

    def copy_prev(self) -> None:
        """Start the frame-before comparison as a copy of the frame-0 one."""
        self.prev_transformers.load_state_dict(self.transformers.state_dict())

    # ---------------------------------------------------------------- the CNN
    def _features(self, frames: torch.Tensor) -> torch.Tensor:
        """(T, H, W, 3) uint8 -> (T, C, H / 4, W / 4). CoTracker's input range."""
        x = frames.permute(0, 3, 1, 2).to(self.fnet.conv1.weight.dtype)
        return self.fnet(x / 127.5 - 1.0)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, 3) uint8 -> (B, T, C, H / 4, W / 4), with no graph."""
        return torch.stack([self._features(f) for f in frames])

    def replay(self, frames: torch.Tensor, fmap: torch.Tensor) -> None:
        """Back-propagate `fmap.grad` into the CNN, a clip at a time.

        `fmap` is what `encode` returned for these frames, made a leaf by the
        caller and since used by every backward pass of the step.
        """
        if fmap.grad is None:
            return
        for f, g in zip(frames, fmap.grad):
            out = self._features(f)
            out.backward(g.to(out.dtype))

    # ------------------------------------------------------- the correlation
    def _windows(self, fmap: torch.Tensor, cell: torch.Tensor):
        """fmap (M, C, h, w), cell (M, N, 2) `(x, y)` in level-0 cells ->
        per level (M, N, K, C): the K = (2r+1)^2 window around each point.

        Yielded a level at a time, so no more than one level's windows exist
        at once."""
        for lvl in range(self.levels):
            if lvl:
                fmap = F.avg_pool2d(fmap, 2, stride=2)
            h, w = fmap.shape[-2:]
            if min(h, w) < 2:
                raise ValueError(f"{self.levels} pyramid levels leave a {h}x{w} map; "
                                 "use fewer levels or a larger frame")
            s = 2 ** lvl
            at = (cell - (s - 1) / 2) / s
            grid = at[:, :, None] + self.delta                         # (M, N, K, 2)
            grid = 2 * grid / grid.new_tensor([w - 1, h - 1]) - 1
            # Zeros outside the frame, as CoTracker has it: a window hanging
            # off the edge should not read as more of the edge.
            # fp32 whatever the map is held in: the position within a cell is
            # the signal, and autocast would widen both arguments anyway.
            yield F.grid_sample(fmap.float(), grid, align_corners=True).permute(0, 2, 3, 1)

    def _places(self, xyz: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor):
        """xyz (b, T, 3, h, w) each cell's 3D position, pos (b, T, N, 3) the
        current guess, cell as `_windows` takes it -> per level (b, T, N, K * 3):
        where each cell of the window is, relative to the guess.

        The positions go through the same pyramid and the same sampling as the
        features, so cell i of a window's offsets is cell i of its table. A
        fourth channel of ones goes with them and comes out as how much of the
        sample fell inside the frame: outside, the offset is zero, and not
        minus the guess, which is what a zero position would give."""
        b, T, N = pos.shape[:3]
        inside = torch.ones_like(xyz[:, :, :1])
        for p in self._windows(torch.cat([xyz, inside], dim=2).flatten(0, 1), cell):
            p = p.reshape(b, T, N, -1, 4)
            yield (p[..., :3] - p[..., 3:] * pos[:, :, :, None].float()).flatten(3)

    def _read(self, table: torch.Tensor, places: torch.Tensor | None) -> torch.Tensor:
        """The MLP on one level's table, and on its window's offsets if any."""
        if places is None:
            return self.corr_mlp(table)
        first = self.corr_mlp[0](table) + F.linear(places, self.xyz_in)
        if self.waves:
            phase = 2 * torch.pi * places[..., None] * self.freqs           # (..., K*3, W)
            first = first + F.linear(torch.cat([phase.sin(), phase.cos()], -1).flatten(-2),
                                     self.wave_in)
        return self.corr_mlp[1:](first)

    def _correlate(self, fmap: torch.Tensor, fmap0: torch.Tensor, uv: torch.Tensor,
                   uv0: torch.Tensor, xyz: torch.Tensor | None,
                   pos: torch.Tensor | None) -> torch.Tensor:
        b, T, C, h, w = fmap.shape
        N = uv.shape[2]
        size = uv.new_tensor([w, h])
        # See the module docstring for the eighth of a cell.
        sup = self._windows(fmap0, uv0 * size - 0.5 / STRIDE)
        cell = (uv * size - 0.5 / STRIDE).flatten(0, 1)
        win = self._windows(fmap.flatten(0, 1), cell)
        places = (self._places(xyz, pos, cell) if xyz is not None
                  else [None] * self.levels)
        out = []
        for s, t, p in zip(sup, win, places):
            cost = torch.einsum("bnsc,btnkc->btnsk", s, t.reshape(b, T, N, -1, C)) * C ** -0.5
            out.append(self._read(cost.flatten(3), p))
        return torch.cat(out, dim=-1)

    def forward(self, fmap: torch.Tensor, fmap0: torch.Tensor, uv: torch.Tensor,
                uv0: torch.Tensor, xyz: torch.Tensor | None = None,
                xyz0: torch.Tensor | None = None, pos: torch.Tensor | None = None,
                pos0: torch.Tensor | None = None, rot: torch.Tensor | None = None) -> torch.Tensor:
        """
        fmap  (B, T, C, h, w)  from `encode`, the frames being tracked
        fmap0 (B, C, h, w)     the clip's frame 0, where the query is: the
                               template, whichever frames `fmap` holds
        uv    (B, T, N, 2)     the current guess in each frame, [0, 1] across, fp32
        uv0   (B, N, 2)        the query point in frame 0, likewise
        xyz   (B, T, 3, h, w)  each cell's 3D position, in the model's own
                               units (the finder with `xyz` only)
        xyz0  (B, 3, h, w)     the same for frame 0
        pos   (B, T, N, 3)     the current guess in 3D, in the same units
        pos0  (B, N, 3)        the query point in 3D, likewise (`neigh` only)
        rot   (B, T, N, 3, 3)  how each point's surface has turned since frame 0
                               (`PointDiT.settle`'s fit), or None: frame 0's
                               neighbourhood is turned by it before the two
                               are compared (`neigh` only)

        returns (B, T, N, out_dim)
        """
        if self.xyz != (xyz is not None):
            raise ValueError("a finder built with xyz reads each cell's 3D position, "
                             "and only that one does")
        if bool(self.neigh) != (pos0 is not None):
            raise ValueError("pos0 is given exactly to a finder built with neigh")
        if rot is not None and not self.neigh:
            raise ValueError("turning frame 0's neighbourhood is the 3D neighbourhood's (neigh)")
        out = []
        # As many clips at a time as hold `chunk` points between them, so a
        # clip with more points is not more memory per piece.
        clips = max(1, self.chunk // uv.shape[2])
        for i in range(0, fmap.shape[0], clips):
            cut = slice(i, i + clips)
            if self.neigh:
                fn = self._neighbourhoods
                part = (fmap[cut], fmap0[cut], uv[cut], uv0[cut], xyz[cut], xyz0[cut], pos[cut], pos0[cut],
                        None if rot is None else rot[cut])
            else:
                fn = self._correlate
                part = (fmap[cut], fmap0[cut], uv[cut], uv0[cut],
                        None if xyz is None else xyz[cut], None if pos is None else pos[cut])
            if torch.is_grad_enabled():
                out.append(checkpoint(fn, *part, use_reentrant=False))
            else:
                out.append(fn(*part))
        return torch.cat(out)

    # ---------------------------------------------- the 3D neighbourhood
    def _neighbourhoods(self, fmap: torch.Tensor, fmap0: torch.Tensor, uv: torch.Tensor,
                        uv0: torch.Tensor, xyz: torch.Tensor, xyz0: torch.Tensor, pos: torch.Tensor,
                        pos0: torch.Tensor, rot: torch.Tensor | None = None) -> torch.Tensor:
        """TAPIP3D's lookup (`knn_feature_4d_optimized.py`): at each level, the
        K cells nearest the current guess IN 3D, against the K cells nearest
        the query point in frame 0 plus the query's own feature, compared by
        `_NeighborTransformer`. A 7x7 window on the image straddles depth edges -- a
        point on a ball's rim takes in the wall behind it; the cells nearest
        in 3D stay on the ball.

        With `prev`, each frame's neighbourhood is also compared against the
        one at the guess in the frame before (the first frame given, which
        has none, against its own), and both results are returned, frame 0's
        first."""
        b, T, C, h, w = fmap.shape
        N = pos.shape[2]
        size = uv0.new_tensor([w, h])
        q = (uv0 * size - 0.5 / STRIDE)                                    # (b, N, 2) level-0 cells
        # Each clip's point cloud brought to unit spread, as TAPIP3D does
        # (`norm_mode: isotropic`): the scale its offset encoding is built for.
        # From frame 0's alone, which every clip sees; TAPIP3D uses the whole
        # window, which a forecast would leak through.
        xyz, xyz0, pos, pos0 = xyz.float(), xyz0.float(), pos.float(), pos0.float()
        mean = xyz0.mean(dim=(2, 3))                                        # (b, 3)
        std = (xyz0 - mean[:, :, None, None]).std(dim=(1, 2, 3)).clamp_min(1e-6)
        xyz = (xyz - mean[:, None, :, None, None]) / std[:, None, None, None, None]
        xyz0 = (xyz0 - mean[:, :, None, None]) / std[:, None, None, None]
        pos = (pos - mean[:, None, None]) / std[:, None, None, None]
        pos0 = (pos0 - mean[:, None]) / std[:, None, None]
        # The frame before each frame, the first standing in for its own.
        before = lambda t: torch.cat([t[:, :1], t[:, :-1]], dim=1)
        out, out_prev = [], []
        for lvl in range(self.levels):
            s = 2 ** lvl
            pool = lambda t: F.avg_pool2d(t, s, stride=s) if lvl else t
            f, f0 = pool(fmap.flatten(0, 1)), pool(fmap0)
            hl, wl = f.shape[-2:]
            # The cells' positions are picked, not averaged: an average of a
            # rim cell and the wall behind it is a point in mid-air.
            pick_xyz = lambda t: F.interpolate(t, size=(hl, wl), mode="nearest-exact")
            x, x0 = pick_xyz(xyz.flatten(0, 1)), pick_xyz(xyz0)
            f = f.flatten(2).transpose(1, 2).reshape(b, T, hl * wl, C)      # (b, T, P, C)
            x = x_cells = x.flatten(2).transpose(1, 2).reshape(b, T, hl * wl, 3)  # (b, T, P, 3)
            f0_cells = f0.flatten(2).transpose(1, 2)                        # (b, P, C)
            x0 = x0.flatten(2).transpose(1, 2)                              # (b, P, 3)
            k = min(self.neigh, hl * wl)
            with torch.no_grad():
                near = torch.cdist(pos, x).topk(k, dim=-1, largest=False).indices      # (b, T, N, k)
                near0 = torch.cdist(pos0, x0).topk(k, dim=-1, largest=False).indices
            pick = lambda t, i: t.gather(-2, i.flatten(-2)[..., None].expand(*i.shape[:-2], -1, t.shape[-1])
                                         ).unflatten(-2, i.shape[-2:])
            cur_f, cur_x = pick(f, near), pick(x, near)                     # (b, T, N, k, .)
            sup_f, sup_x = pick(f0_cells, near0), pick(x0, near0)          # (b, N, k, .)
            # The query's own feature, sampled where it is, leads the support.
            at = (q - (s - 1) / 2) / s
            grid = 2 * at / at.new_tensor([wl - 1, hl - 1]) - 1
            own = F.grid_sample(f0.float(),
                                grid[:, :, None], align_corners=True)[..., 0].transpose(1, 2)
            sup_f = torch.cat([own[:, :, None].to(sup_f.dtype), sup_f], dim=2)
            # Normalised after the sampling and the pooling, as TAPIP3D has it.
            sup_f, cur_f = self.in_norm(sup_f), self.in_norm(cur_f)
            sup_off = torch.cat([torch.zeros_like(sup_x[:, :, :1]), sup_x - pos0[:, :, None]], dim=2)
            cur_off = cur_x - pos[:, :, :, None]
            enc = self.posenc_mlps[lvl]
            y = cur_f + enc(_posenc(cur_off))                                # (b, T, N, k, D)
            if rot is None:
                x = sup_f + enc(_posenc(sup_off))                            # (b, N, k+1, D)
                x = x[:, None].expand(b, T, *x.shape[1:]).flatten(0, 2)
            else:
                # Frame 0's neighbourhood turned the way the surface has
                # turned since, so a turned surface is compared with its own
                # shape as it faces now. The rotation is the same in these
                # units: the clip is brought to unit spread by one scale.
                turned = torch.einsum("btnxy,bnky->btnkx", rot.float(), sup_off)
                x = (sup_f[:, None] + enc(_posenc(turned))).flatten(0, 2)    # (b T N, k+1, D)
            out.append(self.transformers[lvl](x, y.flatten(0, 2)).reshape(b, T, N, -1))
            if self.prev:
                out_prev.append(self._before(f, x_cells, pick, before, uv, pos, hl, wl, s, enc,
                                             self.prev_transformers[lvl], y))
        return torch.cat(out + out_prev, dim=-1)

    def _before(self, f, x, pick, before, uv, pos, hl, wl, s, enc, transformer, y):
        """One level's comparison of the guess's neighbourhood `y` against the
        K cells nearest the guess in the frame before, led by that guess's own
        feature, as the frame-0 support is built. (b, T, N, D)."""
        b, T, P, C = f.shape
        N = pos.shape[2]
        f_b, x_b, pos_b = before(f), before(x), before(pos)
        k = min(self.neigh, P)
        with torch.no_grad():
            near = torch.cdist(pos_b, x_b).topk(k, dim=-1, largest=False).indices
        sup_f, sup_x = pick(f_b, near), pick(x_b, near)                       # (b, T, N, k, .)
        at = (before(uv) * uv.new_tensor([wl * s, hl * s]) - 0.5 / STRIDE - (s - 1) / 2) / s
        grid = 2 * at / at.new_tensor([wl - 1, hl - 1]) - 1                   # (b, T, N, 2)
        maps = f_b.transpose(2, 3).reshape(b * T, C, hl, wl).float()
        own = F.grid_sample(maps, grid.flatten(0, 1)[:, :, None], align_corners=True)[..., 0]
        own = own.transpose(1, 2).reshape(b, T, N, 1, C)
        sup_f = self.in_norm(torch.cat([own.to(sup_f.dtype), sup_f], dim=3))
        sup_off = torch.cat([torch.zeros_like(sup_x[..., :1, :]), sup_x - pos_b[..., None, :]], dim=3)
        sup = sup_f + enc(_posenc(sup_off))
        return transformer(sup.flatten(0, 2), y.flatten(0, 2)).reshape(b, T, N, -1)


def _posenc(x: torch.Tensor, lo: int = -2, hi: int = 14) -> torch.Tensor:
    """x and its sin and cos at 2^lo .. 2^(hi-1) per unit: TAPIP3D's `posenc`."""
    scales = 2.0 ** torch.arange(lo, hi, device=x.device, dtype=x.dtype)
    xb = (x[..., None, :] * scales[:, None]).flatten(-2)
    return torch.cat([x, torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))], dim=-1)


def _posenc_mlp(dim: int) -> nn.Sequential:
    """TAPIP3D's `build_mlp(99, dim, 128, 3, last_act=GELU, output_norm=LayerNorm)`,
    layer for layer, so its indices match the checkpoint's."""
    return nn.Sequential(nn.LayerNorm(99), nn.Linear(99, 128), nn.GELU(), nn.Linear(128, 128),
                         nn.GELU(), nn.Linear(128, dim), nn.GELU(), nn.LayerNorm(dim))


class _Attention(nn.Module):
    """TAPIP3D's `Attention`: q from x, k and v from the context, one SDPA."""

    def __init__(self, dim: int, heads: int = 1, head_dim: int = 128, qkv_bias: bool = False) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.to_q = nn.Linear(dim, heads * head_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, 2 * heads * head_dim, bias=qkv_bias)
        self.to_out = nn.Linear(heads * head_dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N1, _ = x.shape
        split = lambda t: t.reshape(B, t.shape[1], self.heads, self.head_dim).transpose(1, 2)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        out = F.scaled_dot_product_attention(split(self.to_q(x)), split(k), split(v))
        return self.to_out(out.transpose(1, 2).reshape(B, N1, -1))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _CrossAttnBlock(nn.Module):
    """x reads `context`, then a feed-forward: TAPIP3D's `CrossAttnBlock`."""

    def __init__(self, dim: int, mlp_ratio: float = 1.0, heads: int = 1, head_dim: int = 128) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(dim)
        self.cross_attn = _Attention(dim, heads, head_dim, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.norm1(x), self.norm_context(context))
        return x + self.mlp(self.norm2(x))


class _NeighborTransformer(nn.Module):
    """One level's comparison, TAPIP3D's `NeighborTransformer` (128 wide, one
    head): the query's neighbourhood and the guess's read each other, and two
    learned output tokens sum each side into one vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.output_token_1 = nn.Parameter(torch.randn(1, dim))
        self.output_token_2 = nn.Parameter(torch.randn(1, dim))
        self.xblock1_2 = _CrossAttnBlock(dim)
        self.xblock2_1 = _CrossAttnBlock(dim)
        self.aggr1 = _Attention(dim)
        self.aggr2 = _Attention(dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x (M, K+1, D) around the query, y (M, K, D) around the guess -> (M, D)."""
        new_x, new_y = self.xblock1_2(x, y), self.xblock2_1(y, x)
        t1 = self.output_token_1[None].expand(x.shape[0], -1, -1)
        t2 = self.output_token_2[None].expand(x.shape[0], -1, -1)
        return (self.aggr1(t1, new_x) + self.aggr2(t2, new_y))[:, 0]
