"""One training clip per official MOVi-F video: of the old Kubric's cached
clips, those with no MOVi-F twin (no object labels) and second copies of a
video are named, and with --move put aside in OUT/<cache>/. Val clips stay.

  python scripts/one_clip_per_video.py --objects ~/scratch/movi_objects --caches DIR ... --out ~/scratch/cache_left_out [--move]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--objects", required=True)
p.add_argument("--caches", nargs="+", required=True)
p.add_argument("--out", required=True)
p.add_argument("--move", action="store_true")
args = p.parse_args()

objects = Path(args.objects).expanduser()
kept, out = {}, []                                   # video -> the clip kept; (cache, clip, why)
for rank, cache in enumerate(Path(c).expanduser() for c in args.caches):
    val = set((cache / "val.txt").read_text().split())
    counts = dict(val=0, kept=0, unlabelled=0, copy=0)
    # Val clips first, so a video's val copy is the one kept.
    for clip in sorted(cache.glob("*.pt"), key=lambda c: (c.stem not in val, c.stem)):
        labelled = torch.load(clip, weights_only=False, mmap=True).get("segment") is not None
        video = int(np.load(objects / f"{clip.stem}.npz")["video"]) if (objects / f"{clip.stem}.npz").exists() else None
        if clip.stem in val:
            counts["val"] += 1
            if labelled:
                kept.setdefault(video, clip)
            else:
                print(f"  a val clip with no labels: {cache.name}/{clip.stem}")
        elif not labelled:
            counts["unlabelled"] += 1
            out.append((cache, clip))
        elif video in kept:
            counts["copy"] += 1
            out.append((cache, clip))
        else:
            counts["kept"] += 1
            kept[video] = clip
    print(f"{cache.name}: {counts['val']} val clips stay, {counts['kept']} training clips kept, "
          f"{counts['unlabelled']} with no twin and {counts['copy']} second copies left out")
if args.move:
    for cache, clip in out:
        to = Path(args.out).expanduser() / cache.name
        to.mkdir(parents=True, exist_ok=True)
        clip.rename(to / clip.name)
    print(f"{len(out)} clips moved to {args.out}")
