"""Turn raw Kubric clips into the cache that training reads.

Loading a clip means decoding 48 PNGs (24 RGB + 24 depth), ~1.3 s, and
encoding its frames with DINOv3 costs more. Doing either fresh every epoch
would leave the GPU idle far longer than it computes. So: do it once, write
tensors, train from those.

One file per clip, written as it goes. A run that dies partway keeps
everything finished so far and resumes on the next submission, and training
can start on a partial set.

Caveat: the cached normalisation uses statistics from the WHOLE clip, not just
the observed frames. That is a small future-leak. It is acceptable right now
because the first experiment is pure TRACKING -- every frame keeps its image,
so T_C = T and there is no "future" being withheld. Revisit before training
with random cutoffs.

Visual features are cached in fp16 at 384px (the resolution the paper's own
ablations use, §3.4), RAW at the backbone's 384-wide output. Re-encoding them
every epoch would dominate training time; projecting them before caching would
freeze a layer that has to learn. About 11 MB per clip, 37 GB for 3500.

Run:  python -m genpoint3d.prepare.main --root DATA --out local/cache/kubric --features
"""

from genpoint3d.prepare.Pending import Pending
from genpoint3d.prepare.Progress import Progress
from genpoint3d.prepare.Settings import Settings
from genpoint3d.prepare.describe_clip import describe_clip
from genpoint3d.prepare.encode_clip import encode_clip
from genpoint3d.prepare.open_encoder import open_encoder
from genpoint3d.prepare.save_clip import save_clip
from genpoint3d.prepare.transform import transform
from genpoint3d.prepare.unproject_depths import unproject_depths


def prepare(settings: Settings):
    pending = Pending(settings)
    encoder = open_encoder(settings, pending)
    progress = Progress(pending)
    for n, raw in pending.readable():
        inputs = transform(raw, num_context_frames=len(raw.frames))
        pointmap = unproject_depths(inputs)
        clip = describe_clip(inputs, pointmap, settings)
        clip = encode_clip(clip, inputs, pointmap, encoder)
        save_clip(clip, settings)
        progress.tick(n)
    progress.finish(pending, settings)


if __name__ == "__main__":
    import logging
    import sys

    # The package's progress lines, message only, on stdout.
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("genpoint3d").setLevel(logging.INFO)
    prepare(Settings.from_command_line())
