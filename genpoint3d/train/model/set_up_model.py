"""Put a new model on its device, with any starting weights for its CNN."""

import logging

import torch

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.model.PointDiT import PointDiT

log = logging.getLogger(__name__)


def set_up_model(model: PointDiT, settings, hardware: Hardware) -> PointDiT:
    model = model.to(hardware.device)
    if settings.fine and settings.fine_init:
        # Strict: a file that is not this CNN should not half-load.
        model.fine.fnet.load_state_dict(torch.load(settings.fine_init, map_location="cpu"))
        log.info(f"fine CNN initialised from {settings.fine_init}")
    log.info(f"model {model.num_parameters() / 1e6:.2f}M params")
    return model
