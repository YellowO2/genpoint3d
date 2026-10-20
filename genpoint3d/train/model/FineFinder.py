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
                    correlation is checkpointed instead: `chunk` clips and one
                    level at a time (0.3 GB of windows), only its
                    (B, T, N, out_dim) result is kept, and the windows are
                    rebuilt on the way back.

    The map is also what the method's several forward passes share. It reaches
    the model as a leaf with no graph behind it, so each pass can be
    back-propagated on its own and the CNN's gradient simply adds up there.
    """

    def __init__(self, levels: int = 4, radius: int = 3, feat_dim: int = 128,
                 hidden: int = 256, width: int = 64, chunk: int = 2,
                 xyz: bool = False, waves: int = 0) -> None:
        super().__init__()
        self.levels, self.radius, self.chunk = levels, radius, chunk
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

    def _correlate(self, fmap: torch.Tensor, uv: torch.Tensor, uv0: torch.Tensor,
                   xyz: torch.Tensor | None, pos: torch.Tensor | None) -> torch.Tensor:
        b, T, C, h, w = fmap.shape
        N = uv.shape[2]
        size = uv.new_tensor([w, h])
        # See the module docstring for the eighth of a cell.
        sup = self._windows(fmap[:, 0], uv0 * size - 0.5 / STRIDE)
        cell = (uv * size - 0.5 / STRIDE).flatten(0, 1)
        win = self._windows(fmap.flatten(0, 1), cell)
        places = (self._places(xyz, pos, cell) if xyz is not None
                  else [None] * self.levels)
        out = []
        for s, t, p in zip(sup, win, places):
            cost = torch.einsum("bnsc,btnkc->btnsk", s, t.reshape(b, T, N, -1, C)) * C ** -0.5
            out.append(self._read(cost.flatten(3), p))
        return torch.cat(out, dim=-1)

    def forward(self, fmap: torch.Tensor, uv: torch.Tensor, uv0: torch.Tensor,
                xyz: torch.Tensor | None = None, pos: torch.Tensor | None = None) -> torch.Tensor:
        """
        fmap  (B, T, C, h, w)  from `encode`
        uv    (B, T, N, 2)     the current guess in each frame, [0, 1] across, fp32
        uv0   (B, N, 2)        the query point in frame 0, likewise
        xyz   (B, T, 3, h, w)  each cell's 3D position, in the model's own
                               units (the finder with `xyz` only)
        pos   (B, T, N, 3)     the current guess in 3D, in the same units

        returns (B, T, N, out_dim)
        """
        if self.xyz != (xyz is not None):
            raise ValueError("a finder built with xyz reads each cell's 3D position, "
                             "and only that one does")
        out = []
        for i in range(0, fmap.shape[0], self.chunk):
            cut = slice(i, i + self.chunk)
            part = (fmap[cut], uv[cut], uv0[cut],
                    None if xyz is None else xyz[cut], None if pos is None else pos[cut])
            if torch.is_grad_enabled():
                out.append(checkpoint(self._correlate, *part, use_reentrant=False))
            else:
                out.append(self._correlate(*part))
        return torch.cat(out)
