"""Put a new model on its device, with its starting weights."""

import logging

import torch

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.model.PointDiT import PointDiT

log = logging.getLogger(__name__)


def set_up_model(model: PointDiT, settings, hardware: Hardware) -> PointDiT:
    model = model.to(hardware.device)
    if settings.fine_init:
        # TAPIP3D's CNN and comparison (`scripts/extract_tapip3d_finder.py`);
        # the frame-before comparison starts as a copy of it.
        sd = torch.load(settings.fine_init, map_location="cpu")
        missing, unexpected = model.fine.load_state_dict(sd, strict=False)
        if unexpected or [k for k in missing if not k.startswith("prev_transformers.")]:
            raise ValueError(f"{settings.fine_init} does not fit this finder: missing "
                             f"{missing}, unexpected {unexpected}")
        model.fine.copy_prev()
        log.info(f"fine CNN and its comparison initialised from {settings.fine_init}")
    if settings.init:
        # Every weight shared with that run by name and shape; the rest new.
        old = torch.load(settings.init, map_location="cpu", weights_only=False)["model"]
        own = model.state_dict()
        same = {k: v for k, v in old.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(same, strict=False)
        new = [k for k in own if k not in same]
        log.info(f"started from {settings.init}: {len(same)} of {len(own)} weights;"
                 f" new or reshaped: {', '.join(new) if new else 'none'}")
    log.info(f"model {model.num_parameters() / 1e6:.2f}M params")
    return model
