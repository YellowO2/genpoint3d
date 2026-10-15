"""How one clip's metres map to the units the model sees."""

from dataclasses import dataclass

import torch


@dataclass
class NormStats:
    """Affine normalisation for one clip.

    `mean`/`scale` are per-clip and come from the observed scene, so a tabletop
    and a street both land in the same box. `traj_scale` is a single global
    constant applied on top, only to the trajectory, to bring the denoising
    target to unit variance.
    """

    mean: torch.Tensor        # (3,) metric centre of the observed scene
    scale: torch.Tensor       # () scalar, metres per normalised unit
    traj_scale: torch.Tensor  # () scalar, normalised units per model unit

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """(..., 3) metric -> (..., 3) scene-normalised. For geometry."""
        return (pts - self.mean) / self.scale

    def invert(self, pts: torch.Tensor) -> torch.Tensor:
        """(..., 3) scene-normalised -> (..., 3) metric."""
        return pts * self.scale + self.mean

    def apply_traj(self, pts: torch.Tensor) -> torch.Tensor:
        """(..., 3) metric -> (..., 3) model units. For the denoising target."""
        return self.apply(pts) / self.traj_scale

    def invert_traj(self, pts: torch.Tensor) -> torch.Tensor:
        """(..., 3) model units -> (..., 3) metric."""
        return self.invert(pts * self.traj_scale)
