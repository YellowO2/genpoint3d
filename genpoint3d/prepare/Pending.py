"""The raw clips not yet in the cache, read one by one."""

import logging
from pathlib import Path

import torch

from genpoint3d.CachedClip import CachedClip
from genpoint3d.prepare.KubricSequenceDataset import KubricSequenceDataset

log = logging.getLogger(__name__)


class Pending:
    """One file per clip is written as it goes. A run that dies partway keeps
    everything finished so far, and the next run picks up what is left: the
    clips already in `--out` are not read again.

    `cached_dim` is the feature width of the clips already there, None if
    there are none or the run caches no features.
    """

    def __init__(self, settings) -> None:
        out = Path(settings.out)
        out.mkdir(parents=True, exist_ok=True)

        ds = KubricSequenceDataset(settings.root, num_query_points=settings.points)
        found = len(ds.seq_ids)

        # No up-front completeness scan. A half-downloaded clip raises when we try
        # to read it, and `readable` skips it -- so the cost is proportional to
        # the number of BROKEN clips, not to the size of the dataset. Scanning 3500
        # clips to find the zero-or-two bad ones was the wrong trade.
        cached = {p.stem for p in out.glob("*.pt")}
        # Resuming into a cache of a different format would leave clips the model
        # reads two different ways -- the failure this cache layout exists to stop.
        self.cached_dim = None
        if cached and settings.features:
            probe = CachedClip.from_dict(
                torch.load(out / f"{sorted(cached)[0]}.pt", weights_only=False, mmap=True))
            if probe.patch_xyz is None or probe.feat_dim != probe.context.shape[-1]:
                raise SystemExit(
                    f"{out} holds clips in an older format. Resuming would mix the two."
                    " Delete the directory or pass a different --out.")
            # Feature width, which --layers and the backbone both change. A mixed
            # cache is the bug that let a model memorise two clips and learn nothing
            # from three thousand, and a width mismatch is the readable symptom.
            self.cached_dim = probe.context.shape[-1]
        ds.seq_ids = [s for s in ds.seq_ids if s not in cached]
        log.info(f"{found} clips on disk, {len(cached)} already cached,"
                 f" {len(ds.seq_ids)} to encode")

        if settings.limit:
            ds.seq_ids = ds.seq_ids[: settings.limit]
        if not ds.seq_ids:
            log.info("nothing to do")
            raise SystemExit(1)
        self.clips = ds
        self.skipped = []

    def __len__(self) -> int:
        return len(self.clips)

    def readable(self):
        """(n, clip) for each clip that reads, n counting from 1 over every
        clip tried, readable or not."""
        for n, i in enumerate(range(len(self.clips)), 1):
            try:
                raw = self.clips[i]
            except Exception as e:
                # Almost always an incomplete download. Nothing is written for this
                # clip, so a later run retries it for free once the files arrive.
                self.skipped.append(self.clips.seq_ids[i])
                log.info(f"  SKIP {self.clips.seq_ids[i]}: {type(e).__name__}: {e}")
                continue
            yield n, raw
