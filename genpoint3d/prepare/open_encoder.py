"""Load the frozen backbone the cache's features come from, if it has any."""

import logging

import torch

from genpoint3d.prepare.Pending import Pending
from genpoint3d.prepare.VisualEncoder import VisualEncoder

log = logging.getLogger(__name__)


def open_encoder(settings, pending: Pending) -> VisualEncoder | None:
    """None without `--features`. Refuses a width other than the one the
    clips already cached were written at."""
    if not settings.features:
        return None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # No width argument: the cache takes the backbone's own width, because a
    # projection to the model's width would be an untrainable layer sitting
    # in front of the data. Training reads `feat_dim` off the cache.
    encoder = VisualEncoder(image_size=settings.image_size, stub=settings.stub,
                            layers=settings.layers).to(dev).eval()
    log.info(f"encoding on {dev} at {settings.image_size}px, {encoder.dim}-wide features"
             f" from layer(s) {settings.layers or 'last'}"
             f"{' (STUB backbone)' if settings.stub else ''}")
    if pending.cached_dim is not None and pending.cached_dim != encoder.dim:
        raise SystemExit(
            f"{settings.out} holds {pending.cached_dim}-wide features but this run produces"
            f" {encoder.dim}-wide (layers {settings.layers or 'last'}). Resuming"
            " would mix two feature spaces. Use a different --out.")
    return encoder
