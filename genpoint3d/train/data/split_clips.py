"""Split a cache's clips into train and val, by whole clip."""

from pathlib import Path

import torch

from genpoint3d.CachedClip import CachedClip
from genpoint3d.train.data.Clips import Clips


def split_clips(cache: str, val_frac: float, seed: int) -> Clips:
    """Deterministic train/val split over whole clips -- never over points.

    Splitting by point would put the same scene in both halves and the val
    number would be meaningless.

    Returns paths, not loaded clips -- see `ClipDataset`. Also returns the
    cache's feature width and whether it has features, both read from a single
    probe clip, since the encoder chooses its own width and it need not equal
    the model's.
    """
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
    # A cache written before the projection moved into the model holds features
    # that were already projected and already had position summed in, by layers
    # that no longer exist. Nothing in the shapes gives that away: training would
    # run, read no patch positions at all, and quietly produce another number to
    # explain. There is no migration -- the backbone's raw output was never
    # saved -- so this has to be rebuilt.
    if probe.context is not None and probe.patch_xyz is None:
        raise SystemExit(
            f"cache in {path} has features but no patch_xyz -- it predates the "
            "trainable visual head and cannot be migrated. Re-run "
            "python -m genpoint3d.prepare.main --features"
        )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * val_frac))
    return Clips(train=[clips[i] for i in perm[n_val:]],
                 val=[clips[i] for i in perm[:n_val]],
                 feat_dim=probe.feat_dim,
                 has_feats=probe.context is not None)
