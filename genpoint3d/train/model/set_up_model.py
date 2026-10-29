"""Put a new model on its device, with any starting weights for its CNN."""

import logging

import torch

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.model.PointDiT import PointDiT

log = logging.getLogger(__name__)


def set_up_model(model: PointDiT, settings, hardware: Hardware) -> PointDiT:
    model = model.to(hardware.device)
    if settings.fine and settings.fine_init:
        sd = torch.load(settings.fine_init, map_location="cpu")
        if any(k.startswith("fnet.") for k in sd):
            # The whole finder, the CNN and the 3D neighbourhood's comparison
            # (`scripts/extract_tapip3d_finder.py`). Every weight in the file
            # must land, and only the window's readers, which the
            # neighbourhood does not use, may be left as they are.
            missing, unexpected = model.fine.load_state_dict(sd, strict=False)
            unused = ("corr_mlp.", "xyz_in", "wave_in", "prev_transformers.")
            if unexpected or [k for k in missing if not k.startswith(unused)]:
                raise ValueError(f"{settings.fine_init} does not fit this finder: missing "
                                 f"{missing}, unexpected {unexpected}")
            if model.fine.prev:
                model.fine.copy_prev()
            log.info(f"fine CNN and its comparison initialised from {settings.fine_init}")
        else:
            # Strict: a file that is not this CNN should not half-load.
            model.fine.fnet.load_state_dict(sd)
            log.info(f"fine CNN initialised from {settings.fine_init}")
    if settings.init:
        # Every weight this model shares with the run, by name and shape; a
        # part that is new or has changed shape starts as it would without.
        old = torch.load(settings.init, map_location="cpu", weights_only=False)["model"]
        own = model.state_dict()
        same = {k: v for k, v in old.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(same, strict=False)
        new = [k for k in own if k not in same]
        log.info(f"started from {settings.init}: {len(same)} of {len(own)} weights;"
                 f" new or reshaped: {', '.join(new) if new else 'none'}")
    log.info(f"model {model.num_parameters() / 1e6:.2f}M params")
    return model
