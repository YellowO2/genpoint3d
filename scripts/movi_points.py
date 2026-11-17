"""Each old Kubric clip's own points from its MOVi-F twin: of a track at every
4th pixel of every frame (scripts/kubric_tracks.py), those the adding rule
keeps at half its spacing (`add_points`), so wherever training's rule finds a
hole the clip has a point in it. Writes <out>/<old id>.npz: coords (N, T, 2) in
256 px, occluded, traj_3d, queries, segment, centre_3d, matrix_world.

Runs in the TensorFlow Datasets environment (~/scratch/venvs/movi), with torch:
  python scripts/movi_points.py --root ~/scratch/movi_f/256x256/1.0.0 --shards 0 8 --objects ~/scratch/movi_objects --out DIR
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kubric_tracks import point_tracks, quat_to_rot  # noqa: E402
from genpoint3d.tools.traj_scale import REACH  # noqa: E402
from genpoint3d.train.tools.farthest_first import GAP, add_points, spacing  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--split", default="train")
p.add_argument("--shards", type=int, nargs=2, required=True, metavar=("FIRST", "END"))
p.add_argument("--objects", required=True, help="scripts/movi_objects.py's output: which old clips are which video")
p.add_argument("--out", required=True)
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()

root, out = Path(args.root).expanduser(), Path(args.out).expanduser()
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
twins = defaultdict(list)
for f in sorted(Path(args.objects).expanduser().glob("*.npz")):
    twins[int(np.load(f)["video"])].append(f.stem)

features = tfds.builder_from_directory(str(root)).info.features
files = sorted(root.glob(f"movi_f-{args.split}.tfrecord-*"))
files = [str(f) for f in files if args.shards[0] <= int(f.name.split("-")[-3]) < args.shards[1]]
done, counts = 0, []
for ex in tfds.as_numpy(tf.data.TFRecordDataset(files).map(features.deserialize_example)):
    ids = [i for i in twins.get(int(ex["metadata"]["video_name"]), []) if not (out / f"{i}.npz").exists()]
    if not ids:
        continue
    T, H, W = ex["video"].shape[:3]
    lo, hi = ex["metadata"]["depth_range"]
    depth = ex["depth"][..., 0].astype(np.float32) / 65535 * (hi - lo) + lo
    seg = ex["segmentations"][..., 0].astype(np.int32)
    obj_coords = ex["object_coordinates"].astype(np.float32) / 65535. - 0.5
    normals = ex["normal"].astype(np.float32) / 65535. * 2. - 1.
    cam = ex["camera"]
    f = float(cam["focal_length"] / cam["sensor_width"])
    intrinsics = np.array([[f, 0, -0.5], [0, -f, -0.5], [0, 0, -1]], np.float32)
    matrix_world = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    matrix_world[:, :3, :3] = quat_to_rot(cam["quaternions"].astype(np.float64))
    matrix_world[:, :3, 3] = cam["positions"]
    inst = ex["instances"]
    grid = np.stack(np.meshgrid(np.arange(T), np.arange(0, H, 4), np.arange(0, W, 4), indexing="ij"), -1).reshape(-1, 3)
    tr = point_tracks(depth, seg, obj_coords, normals, list(inst["bboxes_3d"]), list(inst["quaternions"]), intrinsics,
                      matrix_world, None, None, centre=0.25, centres=list(inst["positions"]), queries=grid)
    n = np.arange(len(tr["queries"]))
    qt = tr["queries"][:, 0]
    # Each point's depth along the camera's axis in every frame (Blender's camera looks down -z).
    homo = np.concatenate([tr["traj_3d"], np.ones_like(tr["traj_3d"][..., :1])], -1)
    z = -np.einsum("tij,ntj->nti", np.linalg.inv(matrix_world), homo)[..., 2]              # (N, T)
    uv, seen, z = torch.from_numpy(tr["coords"]), torch.from_numpy(~tr["occluded"]), torch.from_numpy(z).float()
    ok = seen[n, qt] & (z[n, qt] < REACH)

    def candidates(t, born):
        c = torch.nonzero(torch.from_numpy(qt == t) & ok)[:, 0]
        return uv[c, t], c, spacing(z[c, t], f * W, W, GAP / 2)

    def covering(t, born):
        b = torch.tensor([k for _, k in born], dtype=torch.long)
        return uv[b, t][seen[b, t]]

    keep = np.array(sorted(k for _, k in add_points(list(range(T)), len(n), candidates, covering)))
    for i in ids:
        np.savez_compressed(out / f"{i}.npz", matrix_world=matrix_world, **{k: v[keep] for k, v in tr.items()})
    counts.append(len(keep))
    done += 1
    if done % 20 == 0:
        print(f"  shards {args.shards}: {done} videos, points a clip median {int(np.median(counts))}", flush=True)
    if args.limit and done >= args.limit:
        break
print(f"=== shards {args.shards}: {done} videos given points; a clip: median {int(np.median(counts)) if counts else 0}, "
      f"least {min(counts, default=0)}, most {max(counts, default=0)} ===", flush=True)
