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
    ) -> None:
        super().__init__()
        self.image_size, self.patch, self.stub = image_size, patch, stub
        self.grid = image_size // patch

        if stub:
            self.dim = 384
            # Fixed random projection of raw patch pixels. Deterministic and
            # input-dependent, so a shape bug shows up but nothing is learnt.
            self.register_buffer("stub_proj", torch.randn(3 * patch * patch, self.dim) * 0.05)
            self.backbone = None
        else:
            from transformers import AutoModel  # imported lazily; heavy

            self.backbone = AutoModel.from_pretrained(model_id)
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            self.dim = self.backbone.config.hidden_size

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
        """(T, 3, S, S) -> (T, P, backbone_dim) patch tokens, no CLS/registers."""
        if self.stub:
            patches = F.unfold(x, kernel_size=self.patch, stride=self.patch)  # (T, 3*p*p, P)
            return patches.transpose(1, 2) @ self.stub_proj

        with torch.no_grad():
            out = self.backbone(pixel_values=x).last_hidden_state  # (T, 1+R+P, D)
        # DINOv3 prepends a CLS token and register tokens. Taking the trailing
        # P entries drops both without hardcoding how many registers there are.
        return out[:, -self.grid * self.grid :]

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

        Averaged over each patch's footprint. TAPIP3D's feature cloud: the patch
        feature says *what*, this says *where*, and the two are combined inside
        the model where the combining weights can be learnt.

        Flattened with `tokens()` so patch i of the feature sequence and row i
        here are the same patch by construction, not by a matching convention
        two files apart.

        Units are whatever `pointmap` is in. `preprocess.py` passes METRES, so
        the cache stores a fact and the load path normalises it -- the same rule
        the trajectory follows.
        """
        return self.tokens(F.adaptive_avg_pool2d(pointmap, self.grid))

    def sample_at(self, feat: torch.Tensor, uv: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """Bilinear lookup at pixel coords -- the query's ID card.

        feat: (T, dim, g, g)      uv: (T, N, 2) pixels      hw: source (H, W)
        returns (T, N, dim)

        `uv` is in original-image pixels, so it is mapped to the [-1, 1] range
        `grid_sample` expects using the ORIGINAL size, not the resized one.
        """
        H, W = hw
        norm = torch.stack([uv[..., 0] / (W - 1), uv[..., 1] / (H - 1)], dim=-1)
        norm = norm * 2.0 - 1.0
        out = F.grid_sample(feat, norm[:, :, None], align_corners=True)  # (T, dim, N, 1)
        return out[..., 0].transpose(1, 2)
