"""For each old Kubric clip, its MOVi-F twin's objects: which object each pixel
shows and each object's pose, from Google's released shards. A clip's twin is
the MOVi-F video with the same first frame (as scripts/find_duplicates.py).

Runs in the TensorFlow Datasets environment (~/scratch/venvs/movi):
  python scripts/movi_objects.py --root ~/scratch/movi_f/256x256/1.0.0 --split train --shards 0 8 --old ~/scratch/kubric --out DIR
"""

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402
from PIL import Image  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--split", default="train")
p.add_argument("--shards", type=int, nargs=2, required=True, metavar=("FIRST", "END"))
p.add_argument("--old", required=True, help="the old Kubric's raw clips (<id>/frames/000.png)")
p.add_argument("--out", required=True)
p.add_argument("--threshold", type=float, default=0.05)
args = p.parse_args()

root, old, out = Path(args.root).expanduser(), Path(args.old).expanduser(), Path(args.out).expanduser()
out.mkdir(parents=True, exist_ok=True)


def thumb(image: Image.Image) -> np.ndarray:
    g = np.asarray(image.convert("L").resize((32, 32), Image.BILINEAR), np.float32).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


ids = sorted(d.name for d in old.iterdir() if (d / "frames" / "000.png").exists())
saved = old.parent / "kubric_thumbs.npy"
if saved.exists():
    olds = np.load(saved)
else:
    olds = np.stack([thumb(Image.open(old / i / "frames" / "000.png")) for i in ids])
    np.save(saved, olds)
assert len(olds) == len(ids)

features = tfds.builder_from_directory(str(root)).info.features
files = sorted(root.glob(f"movi_f-{args.split}.tfrecord-*"))
files = [str(f) for f in files if args.shards[0] <= int(f.name.split("-")[-3]) < args.shards[1]]
videos = twins = 0
for ex in tfds.as_numpy(tf.data.TFRecordDataset(files).map(features.deserialize_example)):
    videos += 1
    d = ((olds - thumb(Image.fromarray(ex["video"][0]))) ** 2).mean(1)
    for j in np.nonzero(d < args.threshold)[0]:
        twins += 1
        inst, cam = ex["instances"], ex["camera"]
        # What each object is: its model (a Google Scanned Object), size and the engine's numbers for it; and the camera.
        np.savez_compressed(out / f"{ids[j]}.npz", seg=ex["segmentations"][..., 0].astype(np.uint8),
                            positions=inst["positions"], quaternions=inst["quaternions"],
                            asset=np.array([a.decode() for a in inst["asset_id"]]), scale=inst["scale"],
                            mass=inst["mass"], friction=inst["friction"], restitution=inst["restitution"],
                            camera_positions=cam["positions"], camera_quaternions=cam["quaternions"],
                            focal_length=cam["focal_length"], sensor_width=cam["sensor_width"],
                            video=int(ex["metadata"]["video_name"]), distance=d[j])
        print(f"clip {ids[j]}: video {int(ex['metadata']['video_name'])}, {len(inst['asset_id'])} objects", flush=True)
print(f"=== shards {args.shards}: {videos} videos, {twins} old clips given objects ===", flush=True)
