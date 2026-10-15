"""Write one cached clip to the cache directory, all or nothing."""

from pathlib import Path

import torch

from genpoint3d.CachedClip import CachedClip


def save_clip(clip: CachedClip, settings) -> None:
    """To a temp name first, so a kill mid-write cannot leave a half-file that
    the resume check would mistake for finished."""
    out = Path(settings.out)
    tmp = out / f".{clip.seq_id}.tmp"
    torch.save(clip.to_dict(), tmp)
    tmp.rename(out / f"{clip.seq_id}.pt")
