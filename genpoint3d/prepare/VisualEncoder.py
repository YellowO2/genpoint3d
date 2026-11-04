"""Encode video frames into frozen DINOv3 patch features, and each patch's 3D
position.

Nothing learnable lives here: this is what the cache stores, and anything
cached is frozen. The query's id card is sampled from the grid
(`tools/feature_lookup.py`); cross-attention reads the whole grid (`tokens`).
`stub=True` swaps DINOv3 for a fixed random projection with the same shapes,
for testing the pipeline without the weights.
"""


import torch
import torch.nn.functional as F
from torch import nn


DINOV3_S = "facebook/dinov3-vits16-pretrain-lvd1689m"

# DINOv3 expects ImageNet-normalised input.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class VisualEncoder(nn.Module):
    """Frozen backbone -> per-patch features at its own width (`self.dim`,
    384 for ViT-S/16), frames resized to `image_size` square first."""

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
        # Backbone layers to read and concatenate; None is the last only.
        self.layers = tuple(layers) if layers else None

        if stub:
            self.dim = 384 * len(self.layers or (0,))
            # A fixed random projection of each patch's pixels.
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
        # The last P tokens: DINOv3 puts CLS and registers first.
        return torch.cat([o[:, -P:] for o in outs], dim=-1)

    # ---------------------------------------------------------------- public
    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames:   (T, H, W, 3) uint8

        returns   (T, dim, grid, grid), a grid `sample_at` can interpolate
        """
        T = frames.shape[0]
        tokens = self._backbone_tokens(self._prepare(frames))            # (T, P, dim)
        return tokens.transpose(1, 2).reshape(T, self.dim, self.grid, self.grid)

    @staticmethod
    def tokens(feat: torch.Tensor) -> torch.Tensor:
        """(T, C, g, g) -> (T, P, C). What cross-attention searches over."""
        return feat.flatten(2).transpose(1, 2)

    @staticmethod
    def patch_centre_xyz(pointmap: torch.Tensor, grid: int) -> torch.Tensor:
        """(T, 3, H, W) scene points -> (T, grid*grid, 3): the one pixel that
        holds each patch's centre, `(j + 0.5) W / grid`, read and not averaged
        (an average at an edge lands in mid-air). Integer arithmetic, the same
        convention as `sample_at`, in `tokens()` order."""
        H, W = pointmap.shape[-2:]
        j = torch.arange(grid, device=pointmap.device)
        rows = ((2 * j + 1) * H) // (2 * grid)
        cols = ((2 * j + 1) * W) // (2 * grid)
        return pointmap[:, :, rows][:, :, :, cols].flatten(2).transpose(1, 2)

    def patch_xyz(self, pointmap: torch.Tensor) -> torch.Tensor:
        """(T, 3, H, W) scene points -> (T, P, 3), each patch's position, in
        the units of `pointmap` (metres in the cache)."""
        return self.patch_centre_xyz(pointmap, self.grid)
