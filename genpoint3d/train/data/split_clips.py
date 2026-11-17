"""Split a run's caches into train and val clips, by whole clip."""

from pathlib import Path

import torch

from genpoint3d.CachedClip import CachedClip
from genpoint3d.train.data.Clips import Clips

VAL_FRAC, SEED = 0.1, 0


def _split_one(path: Path, clips: list) -> tuple[list, list]:
    """`clips` (one cache's, sorted) apart into train and val: the val clips
    are those named in the cache's val.txt, drawn once, so clips added or
    taken away later never move one from val into training."""
    names = path / "val.txt"
    if not names.exists():
        perm = torch.randperm(len(clips), generator=torch.Generator().manual_seed(SEED)).tolist()
        names.write_text("".join(clips[i].stem + "\n" for i in sorted(perm[: max(1, int(len(clips) * VAL_FRAC))])))
    val = set(names.read_text().split())
    return [c for c in clips if c.stem not in val], [c for c in clips if c.stem in val]


def split_clips(caches: list[str], mix: list[float] | None = None) -> Clips:
    """Each cache's clips split apart, as paths (`ClipDataset` loads them): the
    train clips of every cache, each cache drawn its share of `mix` (alike
    by default), and the val clips of the first, which picks best.pt."""
    train, weights, val, feat_dim = [], [], None, None
    for cache, share in zip(caches, mix or [1.0] * len(caches)):
        path = Path(cache)
        clips = sorted(path.glob("*.pt"))
        if not clips:
            raise SystemExit(f"no cached clips in {path} -- run python -m genpoint3d.prepare.main first")
        probe = CachedClip.from_dict(torch.load(clips[0], weights_only=False, mmap=True))
        if probe.context is None or probe.patch_xyz is None:
            raise SystemExit(f"cache in {path} lacks DINOv3 features -- "
                             "run python -m genpoint3d.prepare.main --features")
        if feat_dim not in (None, probe.feat_dim):
            raise SystemExit(f"cache in {path} has {probe.feat_dim}-wide features, the first {feat_dim}")
        feat_dim = probe.feat_dim
        mine, held = _split_one(path, clips)
        train += mine
        weights += [share / len(mine)] * len(mine)
        val = held if val is None else val
    return Clips(train=train, val=val, feat_dim=feat_dim, weights=weights)
