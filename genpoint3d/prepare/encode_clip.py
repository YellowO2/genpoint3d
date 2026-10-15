"""Add a clip's DINOv3 features, its query templates and patch positions."""

import torch

from genpoint3d.CachedClip import CachedClip
from genpoint3d.prepare.ModelInputs import ModelInputs
from genpoint3d.prepare.VisualEncoder import VisualEncoder
from genpoint3d.tools.feature_lookup import sample_at


def encode_clip(clip: CachedClip, inputs: ModelInputs, pointmap: torch.Tensor,
                encoder: VisualEncoder | None) -> CachedClip:
    """The clip unchanged without an encoder."""
    if encoder is None:
        return clip
    dev = encoder.mean.device
    clip.feat_dim = encoder.dim
    clip.image_size = encoder.image_size
    with torch.no_grad():
        feat = encoder(inputs.frames.to(dev))                             # (T,D,g,g)
        # The ID card is sampled ONCE, at the query's own frame (frame 0),
        # per the paper's "unique starting context" -- not per frame.
        uv0 = inputs.query_uv[None].to(dev)
        id_card = sample_at(feat[:1], uv0, clip.hw)[0]                    # (N, D)
        # METRES, like `traj`. The load path normalises it with the same
        # statistics as the anchor, so patch positions and point
        # positions land in one shared space whatever scheme is chosen.
        xyz = encoder.patch_xyz(pointmap.to(dev))                         # (T, P, 3)
    clip.context = encoder.tokens(feat).half().cpu()                      # (T, P, D)
    clip.id_card = id_card.half().cpu()                                   # (N, D)
    clip.patch_xyz = xyz.float().cpu()                                    # (T, P, 3)
    return clip
