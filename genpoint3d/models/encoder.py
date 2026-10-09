"""
[1] Visual encoder -- frozen DINOv3, and nothing else.

Produces the one thing stage 1 owes the rest of the model: a grid of raw
backbone feature vectors per frame, describing what is at each patch. That grid
is then read two different ways (see docs/map.html):

    sample_at()  one spot     -> the query's "ID card", feeds [3]
    tokens()     whole grid   -> searched by cross-attention in [4c]

**Nothing learnable lives here, deliberately.** This module's output is what
`preprocess.py` writes to the cache, and anything cached is frozen for the life
of the cache. A projection and a 3D-position embedding used to sit at the end of
`forward`, so the model's entire interface to the visual world was a pair of
randomly initialised layers that training could never reach: run once, output
saved, layers discarded. Worse, each preprocessing run built its own -- so a
cache assembled over several jobs held clips in mutually unreadable feature
spaces, which is exactly the sort of thing that lets a model memorise two clips
at 94% APD and learn almost nothing from three thousand. Those layers now live
in `PointDiT`, on the training side of the cache boundary.

The rule this encodes: **cache what is frozen, train what is learnable.** The
boundary belongs immediately after the backbone.

`patch_xyz` stays here even though it is not learnable, because it is a fact
about the clip and it needs `grid` to compute.

DINOv3 never sees the query points. It runs on the whole frame, once, and the
queries only index into its output afterwards.

`stub=True` swaps the backbone for a fixed random projection of the image.
Shapes and dtypes are identical, so the entire step-3 pipeline -- feature
cloud, cross-attention, null embedding, masking -- can be tested before the
real (gated) weights are available. It learns nothing; it is a wiring harness.
"""


import torch
import torch.nn.functional as F
from torch import nn


DINOV3_S = "facebook/dinov3-vits16-pretrain-lvd1689m"

# DINOv3 expects ImageNet-normalised input.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def patch_centre_xyz(pointmap: torch.Tensor, grid: int) -> torch.Tensor:
    """(T, 3, H, W) scene points -> (T, grid*grid, 3): the point at each patch's centre.

    One pixel is READ per patch; nothing is averaged. Until 2026-10-10 this was
    the mean over the patch footprint, which at an object edge mixes foreground
    and background and lands in mid-air between them (typically 21 cm from any
    surface), so "where is the patch that looks like me" pointed at nothing.

    Which pixel: the one that contains the patch centre. Patch j spans
    `[j, j+1) * W / grid` in continuous pixel coordinates, so its centre is at
    `(j + 0.5) * W / grid` and lies inside pixel `floor` of that. This is the
    same convention `sample_at` uses, so the position stored for a patch is the
    surface under the spot its feature is addressed by. The footprint is rarely
    a whole odd number of pixels (512 / 24 = 21.33, 512 / 48 = 10.67), so there
    is not always one middle pixel; when the centre falls exactly on a pixel
    boundary, `floor` takes the pixel below/right of it -- one of the four
    tied central pixels, always the same one, and still a single real surface
    point. Integer arithmetic, so the choice cannot flip with float rounding
    between the machine that built a cache and the one that rebuilds it.

    A median over the footprint was the alternative. It also never blends, but
    it picks a different pixel per coordinate unless done on depth alone, and
    then the point can sit anywhere in the patch rather than at its centre.

    Flattened in the order `VisualEncoder.tokens()` uses, so patch i of the
    feature sequence and row i here are the same patch by construction.

    A free function so `scripts/recache_xyz.py` can rewrite an existing cache
    without building a backbone.
    """
    H, W = pointmap.shape[-2:]
    j = torch.arange(grid, device=pointmap.device)
    rows = ((2 * j + 1) * H) // (2 * grid)
    cols = ((2 * j + 1) * W) // (2 * grid)
    return pointmap[:, :, rows][:, :, :, cols].flatten(2).transpose(1, 2)


class VisualEncoder(nn.Module):
    """Frozen backbone -> per-patch features, at the backbone's own width.

    `self.dim` is an OUTPUT, not a setting: the backbone's hidden size, 384 for
    ViT-S/16. Choosing a width here would mean a projection here, which is what
    put untrainable weights in front of the cache.

    Args:
        image_size: frames are resized to this square before encoding. Kubric
                    is 512 native; the paper uses 768 but ablates at 384, so
                    512 sits inside the range they validated and avoids a
                    resample.
        patch:      backbone patch size (16 for ViT-S/16)
        stub:       use a random projection instead of DINOv3 (see module doc)
    """

    def __init__(
        self,
        model_id: str = DINOV3_S,
        image_size: int = 512,
        patch: int = 16,
        stub: bool = False,
        layers: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.image_size, self.patch, self.stub = image_size, patch, stub
        self.grid = image_size // patch
        # Which backbone layers to read. None means the last one only, which is
        # what every cache before this used. A late layer knows "this is a cube
        # face" but has had its position smeared across 12 rounds of global
        # attention; an early one is spatially sharp and semantically blank. A
        # tracker needs both, which is why DELTA, TAPIP3D and SpatialTrackerV2
        # all read several scales. Concatenated, so the adapter learns the mix
        # on the training side of the cache boundary.
        self.layers = tuple(layers) if layers else None

        if stub:
            self.dim = 384 * len(self.layers or (0,))
            # Fixed random projection of raw patch pixels. Deterministic and
            # input-dependent, so a shape bug shows up but nothing is learnt.
            self.register_buffer("stub_proj", torch.randn(3 * patch * patch, self.dim) * 0.05)
            self.backbone = None
        else:
            from transformers import AutoModel  # imported lazily; heavy

            self.backbone = AutoModel.from_pretrained(model_id)
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            self.dim = self.backbone.config.hidden_size * len(self.layers or (0,))
            n = self.backbone.config.num_hidden_layers
            for i in self.layers or ():
                if not 1 <= i <= n:
                    raise ValueError(f"layer {i} out of range: backbone has {n}")

        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    # ------------------------------------------------------------- internals
    def _prepare(self, frames: torch.Tensor) -> torch.Tensor:
        """(T, H, W, 3) uint8 -> (T, 3, S, S) normalised float."""
        x = frames.permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size,) * 2, mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def _backbone_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(T, 3, S, S) -> (T, P, self.dim) patch tokens, no CLS/registers."""
        if self.stub:
            patches = F.unfold(x, kernel_size=self.patch, stride=self.patch)  # (T, 3*p*p, P)
            return patches.transpose(1, 2) @ self.stub_proj

        P = self.grid * self.grid
        with torch.no_grad():
            if self.layers is None:
                outs = [self.backbone(pixel_values=x).last_hidden_state]
            else:
                # hidden_states[0] is the embedding output, so layer i is at i.
                hs = self.backbone(pixel_values=x, output_hidden_states=True).hidden_states
                outs = [hs[i] for i in self.layers]
        # DINOv3 prepends a CLS token and register tokens. Taking the trailing
        # P entries drops both without hardcoding how many registers there are.
        return torch.cat([o[:, -P:] for o in outs], dim=-1)

    # ---------------------------------------------------------------- public
    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames:   (T, H, W, 3) uint8

        returns   (T, dim, grid, grid) -- a feature GRID, not a sequence, so
                  `sample_at` can bilinearly interpolate between patches.
        """
        T = frames.shape[0]
        tokens = self._backbone_tokens(self._prepare(frames))            # (T, P, dim)
        return tokens.transpose(1, 2).reshape(T, self.dim, self.grid, self.grid)

    @staticmethod
    def tokens(feat: torch.Tensor) -> torch.Tensor:
        """(T, C, g, g) -> (T, P, C). What cross-attention searches over."""
        return feat.flatten(2).transpose(1, 2)

    def patch_xyz(self, pointmap: torch.Tensor) -> torch.Tensor:
        """(T, 3, H, W) scene points -> (T, P, 3), one 3D position per patch.

        TAPIP3D's feature cloud: the patch feature says *what*, this says
        *where*, and the two are combined inside the model where the combining
        weights can be learnt. See `patch_centre_xyz` for which point is taken.

        Units are whatever `pointmap` is in. `preprocess.py` passes METRES, so
        the cache stores a fact and the load path normalises it -- the same rule
        the trajectory follows.
        """
        return patch_centre_xyz(pointmap, self.grid)

    @staticmethod
    def sample_at(feat: torch.Tensor, uv: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """Bilinear lookup at pixel coords -- the query's ID card.

        feat: (T, dim, g, g)      uv: (T, N, 2) pixels      hw: source (H, W)
        returns (T, N, dim)

        `uv` is in original-image pixels, so it is mapped to the [-1, 1] range
        `grid_sample` expects using the ORIGINAL size, not the resized one.

        The frame is resized with `align_corners=False`, so patch j is centred
        on native pixel `(j + 0.5) * W / g - 0.5`, and the lookup has to use the
        same convention: pixel u sits at `(u + 0.5) / W` of the way across.
        Until 2026-10-09 this used `u / (W - 1)` with `align_corners=True`,
        which puts patch 0's centre on pixel 0 and read up to half a patch away
        from the query near the image edges. `border` because a query in the
        outer half-patch has no patch centre beyond it to interpolate towards,
        and the default would blend it with zeros.

        Static so `train.py` can redo the lookup on caches written before then.
        """
        H, W = hw
        norm = (uv + 0.5) / torch.tensor([W, H], dtype=uv.dtype, device=uv.device)
        norm = norm * 2.0 - 1.0
        out = F.grid_sample(feat, norm[:, :, None], padding_mode="border",
                            align_corners=False)                         # (T, dim, N, 1)
        return out[..., 0].transpose(1, 2)
