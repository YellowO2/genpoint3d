"""Clips of one raw set that are the same video as a clip of another, by their
first frame: each frame 0 shrunk to 32 x 32 grey, normalised, and the nearest
across the sets. Run before training on a downloaded set, so no clip we test
on is also trained on (TAPIP3D's Kubric clips may be MOVi-F's own videos).

Run:  python scripts/find_duplicates.py --a DIR --a-ids FILE --b DIR [--threshold 0.05]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--a", required=True, help="raw clips to protect, e.g. the old Kubric (<id>/frames/000.png)")
p.add_argument("--a-ids", default=None, help="a file of ids within --a, one a line (default: all)")
p.add_argument("--b", required=True, help="raw clips that may repeat them")
p.add_argument("--threshold", type=float, default=0.05, help="mean squared difference below which two are one")
p.add_argument("--out", default=None, help="write the --b ids found to repeat an --a clip here")
args = p.parse_args()


def thumbs(root: Path, ids: list[str]) -> np.ndarray:
    out = []
    for i in ids:
        g = np.asarray(Image.open(root / i / "frames" / "000.png").convert("L").resize((32, 32), Image.BILINEAR),
                       np.float32).ravel()
        out.append((g - g.mean()) / (g.std() + 1e-6))
    return np.stack(out)


a_root, b_root = Path(args.a).expanduser(), Path(args.b).expanduser()
a_ids = (Path(args.a_ids).read_text().split() if args.a_ids
         else sorted(d.name for d in a_root.iterdir() if (d / "frames").exists()))
b_ids = sorted(d.name for d in b_root.iterdir() if (d / "frames").exists())
A, B = thumbs(a_root, a_ids), thumbs(b_root, b_ids)
# Each thumbnail has zero mean and unit spread, so the mean squared
# difference is 2 - 2 x their correlation: one matrix product.
d = 2 - 2 * (B @ A.T) / A.shape[1]                                          # (b, a)
nearest = d.min(1)
same = nearest < args.threshold
print(f"{len(a_ids)} clips in {a_root}, {len(b_ids)} in {b_root}: {same.sum()} of the second repeat one of the first")
print(f"  nearest distance: min {nearest.min():.3f}, 1st percentile {np.percentile(nearest, 1):.3f}, "
      f"median {np.median(nearest):.3f}")
if args.out:
    Path(args.out).write_text("".join(f"{b_ids[i]}\n" for i in np.nonzero(same)[0]))
