"""A few TAPVid-3D clips from Panoptic Studio, with their frames, to check on.

TAPVid-3D's own generator (`tapnet/tapvid3d/annotation_generation/
generate_pstudio.py`) downloads every annotation file of a split from its
public bucket and the Panoptic Studio frames (Dynamic3DGaussians' data.zip,
590 MB), then joins them. This does the same for only the first --clips files
of the minival split, and keeps the frames as a uint8 array rather than JPEG
bytes, so nothing here needs TensorFlow. Panoptic Studio's cameras do not move,
so there are no extrinsics.

Each `<seq>_<cam>.npz`: images (T, H, W, 3) uint8, intrinsics (fx, fy, cx,
cy), tracks_XYZ (T, N, 3) metres in the camera, visibility (T, N), queries_xyt
(N, 3), the query's pixel and frame.

  python scripts/tapvid3d_pstudio.py --tapnet references/tapnet/repo --out DIR [--clips 10]
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

BUCKET = "https://storage.googleapis.com/dm-tapnet/tapvid3d/release_files/v1.0/pstudio"
FRAMES = "https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip"

p = argparse.ArgumentParser()
p.add_argument("--tapnet", required=True, help="a clone of google-deepmind/tapnet")
p.add_argument("--out", required=True)
p.add_argument("--clips", type=int, default=10)
args = p.parse_args()

sys.path.insert(0, args.tapnet)
from tapnet.tapvid3d.splits import tapvid3d_splits  # noqa: E402

out = Path(args.out)
tmp = out / "tmp"
tmp.mkdir(parents=True, exist_ok=True)
names = tapvid3d_splits.get_minival_files("pstudio")[: args.clips]
zip_path = tmp / "data.zip"
if not zip_path.exists():
    print(f"downloading the Panoptic Studio frames to {zip_path}", flush=True)
    urllib.request.urlretrieve(FRAMES, zip_path)
archive = zipfile.ZipFile(zip_path)
members = archive.namelist()
for name in names:
    dst = out / name
    if dst.exists():
        continue
    ann = tmp / name
    if not ann.exists():
        urllib.request.urlretrieve(f"{BUCKET}/{name}", ann)
    example = dict(np.load(ann, allow_pickle=True))
    seq, cam = name[:-4].split("_")
    frames = sorted(m for m in members if m.startswith(f"data/{seq}/ims/{cam}/") and m.endswith(".jpg"))
    example["images"] = np.stack([np.array(Image.open(archive.open(m)).convert("RGB")) for m in frames])
    if len(example["images"]) != example["tracks_XYZ"].shape[0]:
        raise ValueError(f"{name}: {len(example['images'])} frames for "
                         f"{example['tracks_XYZ'].shape[0]} track frames")
    np.savez(dst, **example)
    print(f"  {name}  frames {example['images'].shape}  tracks {example['tracks_XYZ'].shape}"
          f"  query frames {np.unique(example['queries_xyt'][:, 2]).size} distinct", flush=True)
