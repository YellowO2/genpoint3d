"""Everything a cache-building run is told on its command line."""

import argparse


class Settings(argparse.Namespace):
    """The run's flags, one attribute each, as `argparse` parses them."""

    @classmethod
    def from_command_line(cls, argv: list[str] | None = None) -> "Settings":
        p = argparse.ArgumentParser(
            prog="python -m genpoint3d.prepare.main",
            description="Turn raw Kubric clips into the cache training reads.")
        p.add_argument("--root", required=True)
        p.add_argument("--out", default="local/cache/kubric", help="DIRECTORY, one .pt per clip")
        p.add_argument("--points", type=int, default=256)
        p.add_argument("--features", action="store_true", help="also cache DINOv3 features")
        p.add_argument("--image-size", type=int, default=384)
        # Which backbone layers to concatenate. Omitted means the last one only,
        # which is what every cache before this held. A late layer is semantic but
        # spatially smeared; an early one is sharp but semantically blank, and a
        # tracker needs both -- DELTA, TAPIP3D and SpatialTrackerV2 all read several.
        # Each extra layer MULTIPLIES the cache: 384-wide per layer, ~39 GB per
        # layer for 3500 clips. Check quota before asking for four.
        p.add_argument("--layers", type=int, nargs="*", default=None,
                       help="e.g. --layers 6 12 (1-indexed, 12 layers in ViT-S/16)")
        p.add_argument("--stub", action="store_true", help="random backbone, for testing without DINOv3 access")
        p.add_argument("--limit", type=int, default=None, help="only the first N complete clips")
        return p.parse_args(argv, namespace=cls())
