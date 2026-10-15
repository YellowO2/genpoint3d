"""Lift every pixel of a clip's depth maps to a 3D point, in metres."""

import torch

from genpoint3d.prepare.ModelInputs import ModelInputs
from genpoint3d.prepare.tools.geometry import batch_unproject


def unproject_depths(inputs: ModelInputs) -> torch.Tensor:
    """(T, 3, H, W) scene points in the frame-0 camera, METRES, not model
    units: normalisation is applied when a clip is loaded, so changing the
    scheme never invalidates the cache again."""
    return batch_unproject(inputs.depths, inputs.intrinsics, inputs.extrinsics)
