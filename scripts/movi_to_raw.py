"""MOVi-F clips from Google's released shards (gs://kubric-public/tfds/movi_f)
into the raw layout our renderer writes (scripts/kubric_movi.pbs): <id>/<id>.npy
with 3D point tracks and each point's object, and <id>/frames/{t:03d}.png and
{t:03d}_depth.png. The tracks are computed as for our own renders
(scripts/kubric_tracks.py), from the shards' depth, segmentation, object
coordinates, normals, cameras and object boxes. Each clip is named mf<video>.

Runs in the TensorFlow Datasets environment (~/scratch/venvs/movi):
  python scripts/movi_to_raw.py --root ~/scratch/movi_f/256x256/1.0.0 --split train --shards 0 10 --out DIR
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kubric_tracks import point_tracks, quat_to_rot  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", required=True, help="the dataset version directory, with dataset_info.json and the shards")
p.add_argument("--split", default="train")
p.add_argument("--shards", type=int, nargs=2, required=True, metavar=("FIRST", "END"), help="shards FIRST to END-1")
p.add_argument("--out", required=True)
p.add_argument("--tracks", type=int, default=2048)
p.add_argument("--limit", type=int, default=None, help="stop after this many clips")
args = p.parse_args()

root, out = Path(args.root).expanduser(), Path(args.out).expanduser()
features = tfds.builder_from_directory(str(root)).info.features
files = sorted(root.glob(f"movi_f-{args.split}.tfrecord-*"))
files = [str(f) for f in files if args.shards[0] <= int(f.name.split("-")[-3]) < args.shards[1]]
ds = tf.data.TFRecordDataset(files).map(features.deserialize_example)

done = 0
for ex in tfds.as_numpy(ds):
    name = f"mf{int(ex['metadata']['video_name']):06d}"
    clip = out / name
    if (clip / f"{name}.npy").exists():
        continue
    video = ex["video"]                                                    # (T, H, W, 3) uint8
    T, H, W = video.shape[:3]
    lo, hi = ex["metadata"]["depth_range"]
    depth_u16 = ex["depth"][..., 0]
    depth = depth_u16.astype(np.float32) / 65535 * (hi - lo) + lo           # distance from the camera
    seg = ex["segmentations"][..., 0].astype(np.int32)
    obj_coords = ex["object_coordinates"].astype(np.float32) / 65535. - 0.5
    normals = ex["normal"].astype(np.float32) / 65535. * 2. - 1.
    cam = ex["camera"]
    f = float(cam["focal_length"] / cam["sensor_width"])
    intrinsics = np.array([[f, 0, -0.5], [0, -f, -0.5], [0, 0, -1]], np.float32)   # normalised, Kubric's signs
    matrix_world = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    matrix_world[:, :3, :3] = quat_to_rot(cam["quaternions"].astype(np.float64))
    matrix_world[:, :3, 3] = cam["positions"]
    inst = ex["instances"]
    boxes, quats = list(inst["bboxes_3d"]), list(inst["quaternions"])
    rng = np.random.RandomState(int(ex["metadata"]["video_name"]))
    tr = point_tracks(depth, seg, obj_coords, normals, boxes, quats, intrinsics, matrix_world, args.tracks, rng)

    # Each point projects back onto its own query pixel at its query frame.
    n = np.arange(len(tr["queries"]))
    err = np.linalg.norm(tr["coords"][n, tr["queries"][:, 0]] - (tr["queries"][:, [2, 1]] + 0.5), axis=-1)
    frames = clip / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for t in range(T):
        Image.fromarray(video[t]).save(frames / f"{t:03d}.png")
        Image.fromarray(depth_u16[t]).save(frames / f"{t:03d}_depth.png")
    np.save(clip / f"{name}.npy", {
        "coords": tr["coords"], "occluded": tr["occluded"], "traj_3d": tr["traj_3d"],
        "intrinsics": np.repeat(intrinsics[None], T, 0), "matrix_world": matrix_world,
        "depth_range": np.array([lo, hi], np.float32), "queries": tr["queries"], "segment": tr["segment"],
    })
    moving = (np.linalg.norm(tr["traj_3d"] - tr["traj_3d"][:, :1], axis=-1).max(1) > 0.01).mean()
    print(f"{name}: {len(n)} points, query reprojection error median {np.median(err):.3f} px max {err.max():.3f}, "
          f"{len(boxes)} objects, moving {moving:.2f}", flush=True)
    done += 1
    if args.limit and done >= args.limit:
        break
print(f"=== {done} clips written to {out} ===")
