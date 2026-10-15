"""The device a run computes on, and whether it computes in bf16."""

from dataclasses import dataclass

import torch


@dataclass
class Hardware:
    device: torch.device
    # bf16 autocast. bf16 rather than fp16: it has the same exponent range as
    # fp32, so no loss scaler is needed and nothing silently underflows. A100
    # and newer only.
    amp: bool

    @classmethod
    def pick(cls, settings) -> "Hardware":
        """The best device here; bf16 only where it is asked for and supported."""
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        amp = settings.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
        return cls(device, amp)
