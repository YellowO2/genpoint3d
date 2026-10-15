"""Say how far a cache-building run has got, and what it left behind."""

import logging
import time
from pathlib import Path

from genpoint3d.prepare.Pending import Pending

log = logging.getLogger(__name__)


class Progress:
    def __init__(self, pending: Pending) -> None:
        self.total = len(pending)
        self.t0 = time.time()

    def tick(self, n: int) -> None:
        """Clip n of the run is written."""
        if n % 25 == 0 or n == self.total:
            rate = (time.time() - self.t0) / n
            log.info(f"  {n}/{self.total}  {rate:.2f}s/clip"
                     f"  eta {rate * (self.total - n) / 60:.1f} min")

    def finish(self, pending: Pending, settings) -> None:
        skipped = pending.skipped
        if skipped:
            log.info(f"\nskipped {len(skipped)} unreadable clips: {' '.join(skipped[:20])}"
                     f"{' ...' if len(skipped) > 20 else ''}")
        out = Path(settings.out)
        files = sorted(out.glob("*.pt"))
        mb = sum(f.stat().st_size for f in files) / 1e6
        log.info(f"\n{len(files)} clips cached in {out}"
                 f"  ({mb:.1f} MB, {mb / max(len(files),1):.2f} MB/clip)")
