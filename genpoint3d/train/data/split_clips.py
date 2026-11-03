"""Split a cache's clips into train and val, by whole clip."""

from pathlib import Path

import torch

from genpoint3d.CachedClip import CachedClip
from genpoint3d.train.data.Clips import Clips


def split_clips(cache: str, val_frac: float, seed: int) -> Clips:
    """A fixed train/val split over whole clips, as paths (`ClipDataset`
    loads them), and the cache's feature width from one probe clip."""
    path = Path(cache)
    if path.is_dir():
        # one .pt per clip -- written incrementally so preprocessing resumes
        clips: list[Path | dict] = sorted(path.glob("*.pt"))
        if not clips:
            raise SystemExit(f"no cached clips in {path} -- run python -m genpoint3d.prepare.main first")
        probe = CachedClip.from_dict(torch.load(clips[0], weights_only=False, mmap=True))
    else:
        clips = torch.load(path, weights_only=False)["clips"]  # legacy single file
        probe = CachedClip.from_dict(clips[0])
    if probe.stats is None:
        raise SystemExit(f"cache in {path} has no scale statistics -- "
                         "re-run python -m genpoint3d.prepare.main")
    if probe.context is None or probe.patch_xyz is None:
        raise SystemExit(f"cache in {path} has no DINOv3 features and patch positions -- "
                         "run python -m genpoint3d.prepare.main --features")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * val_frac))
    return Clips(train=[clips[i] for i in perm[n_val:]],
                 val=[clips[i] for i in perm[:n_val]],
                 feat_dim=probe.feat_dim)
