"""Batches of a run's clips: an endless stream to train on, and val passes."""

from torch.utils.data import DataLoader

from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.data.Clips import Clips


class Loaders:
    """`train` shuffles and is drawn from without end; `val` and `fit` are
    scored in order. `fit` is `--train-eval` clips of the TRAIN split, None
    without it."""

    def __init__(self, clips: Clips, settings) -> None:
        # What every loader shares: how a clip is turned into a sample.
        how_data = dict(norm_mode=settings.norm_mode, target=settings.target,
                        fix_idcard=bool(settings.fix_idcard), frames=clips.frames,
                        features=clips.has_feats)
        self.train = DataLoader(
            ClipDataset(clips.train, settings.points, **how_data),
            batch_size=settings.batch, shuffle=True, num_workers=settings.workers,
            drop_last=True, persistent_workers=settings.workers > 0,
        )
        self.val = DataLoader(
            ClipDataset(clips.val, settings.points, resample=False, **how_data),
            batch_size=settings.batch, shuffle=False, num_workers=settings.workers,
        )
        # Scored with resample=False and the val loader's settings, so the only
        # difference from val is WHICH clips it holds. Anything else would
        # confound the comparison the number exists to make.
        self.fit = DataLoader(
            ClipDataset(clips.train[: settings.train_eval], settings.points, resample=False,
                        **how_data),
            batch_size=settings.batch, shuffle=False, num_workers=settings.workers,
        ) if settings.train_eval else None
        self._stream = None

    def micro_batches(self, count: int):
        """The next `count` training batches, one at a time. Epochs run into
        each other: a new one starts, reshuffled, the moment the last ends."""
        if self._stream is None:
            self._stream = self._forever()
        for _ in range(count):
            yield next(self._stream)

    def _forever(self):
        while True:
            yield from self.train
