"""
[1] Visual encoder -- frozen DINOv3 plus the 3D feature cloud.

Produces the one thing stage 1 owes the rest of the model: a grid of feature
vectors per frame, describing what is at each patch AND where that patch sits
in 3D. That grid is then read two different ways (see docs/map.html):

    sample_at()  one spot     -> the query's "ID card", feeds [3]
    tokens()     whole grid   -> searched by cross-attention in [4c]

DINOv3 never sees the query points. It runs on the whole frame, once, and the
queries only index into its output afterwards.

`stub=True` swaps the backbone for a fixed random projection of the image.
Shapes and dtypes are identical, so the entire step-3 pipeline -- feature
cloud, cross-attention, null embedding, masking -- can be tested before the
real (gated) weights are available. It learns nothing; it is a wiring harness.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from genpoint3d.models.layers import FourierEmbedding

DINOV3_S = "facebook/dinov3-vits16-pretrain-lvd1689m"

# DINOv3 expects ImageNet-normalised input.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class VisualEncoder(nn.Module):
    """Frozen backbone -> per-patch features, with 3D position folded in.

    Args:
        dim:        model width to project features into
        image_size: frames are resized to this square before encoding. Kubric
                    is 512 native; the paper uses 768 but ablates at 384, so
                    512 sits inside the range they validated and avoids a
                    resample.
        patch:      backbone patch size (16 for ViT-S/16)
        stub:       use a random projection instead of DINOv3 (see module doc)
    """

    def __init__(
        self,
        dim: int,
        model_id: str = DINOV3_S,
        image_size: int = 512,
        patch: int = 16,
        stub: bool = False,
        feature_cloud: bool = True,
    ) -> None:
        super().__init__()
        self.dim, self.image_size, self.patch, self.stub = dim, image_size, patch, stub
        self.grid = image_size // patch
        self.feature_cloud = feature_cloud

        if stub:
            backbone_dim = 384
            # Fixed random projection of raw patch pixels. Deterministic and
            # input-dependent, so a shape bug shows up but nothing is learnt.
            self.register_buffer("stub_proj", torch.randn(3 * patch * patch, backbone_dim) * 0.05)
            self.backbone = None
        else:
            from transformers import AutoModel  # imported lazily; heavy

            self.backbone = AutoModel.from_pretrained(model_id)
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            backbone_dim = self.backbone.config.hidden_size

        self.proj = nn.Linear(backbone_dim, dim, bias=False)
        # 3D position of each patch, Fourier-encoded and added to its feature.
        # TAPIP3D's feature cloud: the patch says *what*, this says *where*.
        self.pos_enc = FourierEmbedding(3, dim) if feature_cloud else None

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
    def forward(self, frames: torch.Tensor, pointmap: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        frames:   (T, H, W, 3) uint8
        pointmap: (T, 3, H, W) scene geometry, frame-0 frame, scene-normalised.
                  Required when `feature_cloud` is on.

        returns   (T, dim, grid, grid) -- a feature GRID, not a sequence, so
                  `sample_at` can bilinearly interpolate between patches.
        """
        T = frames.shape[0]
        tokens = self.proj(self._backbone_tokens(self._prepare(frames)))  # (T, P, dim)
        feat = tokens.transpose(1, 2).reshape(T, self.dim, self.grid, self.grid)

        if self.feature_cloud:
            if pointmap is None:
                raise ValueError("feature_cloud=True needs a pointmap")
            # Average depth-derived 3D position within each patch footprint.
            xyz = F.adaptive_avg_pool2d(pointmap, self.grid)            # (T, 3, g, g)
            xyz = xyz.permute(0, 2, 3, 1)                                # (T, g, g, 3)
            feat = feat + self.pos_enc(xyz).permute(0, 3, 1, 2)
        return feat

    @staticmethod
    def tokens(feat: torch.Tensor) -> torch.Tensor:
        """(T, dim, g, g) -> (T, P, dim). What cross-attention searches over."""
        return feat.flatten(2).transpose(1, 2)

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
