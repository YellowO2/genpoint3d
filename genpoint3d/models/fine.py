"""
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

It needs the current guess as a PIXEL, which the rest of the model never does:
`image_projection` and `project` are the camera, folded into one matrix per
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

# Pixels per cell of the CNN's output, fixed by its two stride-2 stages.
STRIDE = 4


class ResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.InstanceNorm2d(planes)
        self.norm2 = nn.InstanceNorm2d(planes)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride), nn.InstanceNorm2d(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    """(M, 3, H, W) in [-1, 1] -> (M, 128, H / 4, W / 4).

    Four stages at strides 2, 4, 8, 16, all resampled to stride 4 and mixed by
    two convolutions. Module names and shapes are CoTracker3's `fnet`, so its
    weights load as they are. Instance norm has no weights and no running
    statistics: a frame's features do not depend on what else is in the batch,
    which is what lets `FineFinder` run it a clip at a time.
    """

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.in_planes = output_dim // 2
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)
        self.conv1 = nn.Conv2d(3, self.in_planes, 7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)
        self.conv2 = nn.Conv2d(output_dim * 3 + output_dim // 4, output_dim * 2, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, dim: int, stride: int) -> nn.Sequential:
        layers = (ResidualBlock(self.in_planes, dim, stride), ResidualBlock(dim, dim))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = (x.shape[-2] // STRIDE, x.shape[-1] // STRIDE)
        x = self.relu1(self.norm1(self.conv1(x)))
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)
        up = lambda t: F.interpolate(t, size, mode="bilinear", align_corners=True)
        x = self.conv2(torch.cat([up(a), up(b), up(c), up(d)], dim=1))
        return self.conv3(self.relu2(self.norm2(x)))


def image_projection(intrinsics: torch.Tensor, extrinsics: torch.Tensor, hw: torch.Tensor,
                     scale: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """(B, T, 3, 4): scene-normalised position -> where it is drawn, per frame.

    intrinsics  (B, T, 3, 3) pixels of the NATIVE frame
    extrinsics  (B, T, 4, 4) cam_0 -> cam_t
    hw          (B, 2) native frame size (H, W)
    scale, mean (B,), (B, 3) the clip's normalisation: metres = p * scale + mean

    One matrix does normalisation, camera pose and intrinsics, in that order,
    and its output is `(u * z, v * z, z)` with u, v in [0, 1] across the frame
    -- see `project`. Dividing by the frame size here is what makes the result
    independent of the resolution anything was resized to.
    """
    K = intrinsics.float().clone()
    K[..., 0, :] /= hw[:, 1, None, None].float()
    K[..., 1, :] /= hw[:, 0, None, None].float()
    A = K @ extrinsics.float()[..., :3, :]                            # metres -> image
    lin = A[..., :3] * scale.float()[:, None, None, None]
    off = A[..., :3] @ mean.float()[:, None, :, None] + A[..., 3:]
    return torch.cat([lin, off], dim=-1)


def project(pos: torch.Tensor, proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pos (B, T, N, 3) scene-normalised, proj (B, T, 3, 4), both fp32 ->
    (B, T, N, 2) `(u, v)` in [0, 1] across the frame, and (B, T, N) depth.

    A point behind the camera has no pixel. Its depth is floored and the
    result held to within one frame-width of the frame, so it reads a window
    of zeros rather than an infinity.
    """
    q = torch.einsum("btij,btnj->btni", proj[..., :3], pos) + proj[:, :, None, :, 3]
    z = q[..., 2]
    uv = q[..., :2] / z.clamp_min(1e-4)[..., None]
    return uv.clamp(-1.0, 2.0), z


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
                 hidden: int = 256, width: int = 64, chunk: int = 2) -> None:
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

    def _correlate(self, fmap: torch.Tensor, uv: torch.Tensor, uv0: torch.Tensor) -> torch.Tensor:
        b, T, C, h, w = fmap.shape
        N = uv.shape[2]
        size = uv.new_tensor([w, h])
        # See the module docstring for the eighth of a cell.
        sup = self._windows(fmap[:, 0], uv0 * size - 0.5 / STRIDE)
        win = self._windows(fmap.flatten(0, 1), (uv * size - 0.5 / STRIDE).flatten(0, 1))
        out = []
        for s, t in zip(sup, win):
            cost = torch.einsum("bnsc,btnkc->btnsk", s, t.reshape(b, T, N, -1, C)) * C ** -0.5
            out.append(self.corr_mlp(cost.flatten(3)))
        return torch.cat(out, dim=-1)

    def forward(self, fmap: torch.Tensor, uv: torch.Tensor, uv0: torch.Tensor) -> torch.Tensor:
        """
        fmap  (B, T, C, h, w)  from `encode`
        uv    (B, T, N, 2)     the current guess in each frame, [0, 1] across, fp32
        uv0   (B, N, 2)        the query point in frame 0, likewise

        returns (B, T, N, out_dim)
        """
        out = []
        for i in range(0, fmap.shape[0], self.chunk):
            part = (fmap[i:i + self.chunk], uv[i:i + self.chunk], uv0[i:i + self.chunk])
            if torch.is_grad_enabled():
                out.append(checkpoint(self._correlate, *part, use_reentrant=False))
            else:
                out.append(self._correlate(*part))
        return torch.cat(out)
