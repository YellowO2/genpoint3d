"""
Minimal Kubric 3D point tracking dataset.

Expects the zbww/tapip3d-kubric layout:
    <root>/000000/000000.npy      -> dict with coords, occluded, traj_3d,
                                      intrinsics, matrix_world, depth_range
    <root>/000000/frames/{i:03d}.png        -> RGB frame i
    <root>/000000/frames/{i:03d}_depth.png  -> 16-bit encoded distance map for frame i

Field shapes as stored in the .npy (N points, T frames):
    coords        (N, T, 2)  float32   2D pixel coordinates
    occluded      (N, T)     bool      True = point occluded at that frame
    traj_3d       (N, T, 3)  float32   3D point positions, world space
    intrinsics    (T, 3, 3)  float32   per-frame camera intrinsics, NORMALIZED to [0, 1]
    matrix_world  (T, 4, 4)  float32   per-frame camera-to-world transform (Kubric/Blender convention)
    depth_range   (2,)       float32   [min_distance, max_distance] used to decode the depth pngs

The depth pngs encode *distance from the camera*, not z-depth, so we convert
via the pinhole model after loading.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# Kubric/Blender camera axes -> OpenCV camera axes (X right, Y down, Z forward).
_AXIS_FLIP = np.diag(np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32))


@dataclass
class KubricSample:
    seq_id: str
    frames: np.ndarray        # (T, H, W, 3) uint8 RGB
    depths: np.ndarray        # (T, H, W) float32 z-depth, meters
    traj_3d: np.ndarray       # (N, T, 3) float32 -- world-space 3D trajectory
    coords_2d: np.ndarray     # (N, T, 2) float32 -- 2D pixel projections
    visibility: np.ndarray    # (N, T) bool -- True = visible (NOT occluded)
    intrinsics: np.ndarray    # (T, 3, 3) float32, in pixel units
    extrinsics: np.ndarray    # (T, 4, 4) float32, world-to-camera, OpenCV convention
    query_points_3d: np.ndarray  # (N, 3) float32 -- 3D position at each point's first frame


# A clip is 48 small files, and on a network filesystem (NSCC's Lustre) each
# read is a round trip that dominates the decode. Read them concurrently:
# threads are the right tool because the time is spent *waiting*, and both
# file reads and cv2.imdecode release the GIL.
#
# Measured on NSCC: serial reads gave 46 s/clip with the job 90% idle
# (cput 28 min against 4h49m walltime). Locally, where files are on a real
# disk, this changes nothing much -- it is the network latency that parallelises.
_READ_WORKERS = int(os.environ.get("KUBRIC_READ_WORKERS", "16"))


def _imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read and decode one image, raising a useful error on a truncated file."""
    buf = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise ValueError(f"could not decode {path} -- truncated or corrupt")
    return img


def _load_frames_and_depths(frames_dir: str, num_frames: int, depth_range: np.ndarray):
    def one(t: int):
        rgb = _imread(os.path.join(frames_dir, f"{t:03d}.png"))
        depth_png = _imread(os.path.join(frames_dir, f"{t:03d}_depth.png"), cv2.IMREAD_UNCHANGED)
        distance = depth_range[0] + depth_png.astype(np.float32) * (depth_range[1] - depth_range[0]) / 65535.0
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), distance

    # `map` preserves order, so frame t stays at index t.
    with ThreadPoolExecutor(min(_READ_WORKERS, num_frames)) as ex:
        pairs = list(ex.map(one, range(num_frames)))

    frames, depths = zip(*pairs)
    return np.stack(frames), np.stack(depths)  # depths here are still *distance*, converted to z-depth by caller


def _distance_to_depth(distances: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """distances: (T, H, W) distance-from-camera. intrinsics: (T, 3, 3) pixel-space. -> (T, H, W) z-depth."""
    h, w = distances.shape[-2:]
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    uv_homogeneous = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)  # (h, w, 3)

    K_inv = np.linalg.inv(intrinsics)  # (T, 3, 3)
    camera_rays = np.einsum("tij,hwj->thwi", K_inv, uv_homogeneous)  # (T, h, w, 3)
    depth = camera_rays[..., -1] * distances / np.linalg.norm(camera_rays, axis=-1)
    return depth


class KubricSequenceDataset:
    """
    root_dir/
        000000/
            000000.npy
            frames/
        000001/
            000001.npy
            frames/
        ...

    Pass an explicit list of sequence ids if you've only downloaded a subset
    (e.g. seq_ids=["000000", "000001"] for your n=2 starting point).
    """

    def __init__(self, root_dir: str, seq_ids: list[str] | None = None, num_query_points: int = 256):
        self.root_dir = root_dir
        if seq_ids is None:
            seq_ids = sorted(
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )
        self.seq_ids = seq_ids
        self.num_query_points = num_query_points

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx: int) -> KubricSample:
        seq_id = self.seq_ids[idx]
        seq_dir = os.path.join(self.root_dir, seq_id)
        npy_path = os.path.join(seq_dir, f"{seq_id}.npy")
        frames_dir = os.path.join(seq_dir, "frames")

        data = np.load(npy_path, allow_pickle=True).item()
        coords = data["coords"].transpose(1, 0, 2)          # (N,T,2) -> (T,N,2)
        occluded = data["occluded"].transpose(1, 0)          # (N,T)   -> (T,N)
        traj_3d = data["traj_3d"].transpose(1, 0, 2)          # (N,T,3) -> (T,N,3)
        depth_range = data["depth_range"]
        num_frames = data["intrinsics"].shape[0]

        frames, distances = _load_frames_and_depths(frames_dir, num_frames, depth_range)
        H, W = frames.shape[1:3]

        # intrinsics are normalized -> scale to pixel units
        intrinsics = np.abs(data["intrinsics"]).astype(np.float32)  # negative entries are a Kubric quirk
        intrinsics[:, 0, :] *= W
        intrinsics[:, 1, :] *= H

        depths = _distance_to_depth(distances, intrinsics)

        # matrix_world (camera-to-world, Blender axes) -> extrinsics (world-to-camera, OpenCV axes)
        extrinsics = np.linalg.inv(data["matrix_world"]).astype(np.float32)
        extrinsics = np.einsum("ij,tjk->tik", _AXIS_FLIP, extrinsics)

        visibility = ~occluded  # True = visible
        visibility &= (coords[..., 0] >= 0) & (coords[..., 0] < W) & (coords[..., 1] >= 0) & (coords[..., 1] < H)

        N_total = traj_3d.shape[1]
        if self.num_query_points is not None and self.num_query_points < N_total:
            visible_at_0 = np.where(visibility[0])[0]
            pool = visible_at_0 if len(visible_at_0) >= self.num_query_points else np.arange(N_total)
            chosen = np.random.choice(pool, size=self.num_query_points, replace=False)
        else:
            chosen = np.arange(N_total)

        traj_3d = traj_3d[:, chosen]
        coords = coords[:, chosen]
        visibility = visibility[:, chosen]
        query_points_3d = traj_3d[0]  # 3D position at first frame

        return KubricSample(
            seq_id=seq_id,
            frames=frames,
            depths=depths,
            traj_3d=traj_3d,
            coords_2d=coords,
            visibility=visibility,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            query_points_3d=query_points_3d,
        )


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "local/data/kubric_test"
    ds = KubricSequenceDataset(root, num_query_points=256)
    print(f"Found {len(ds)} sequences: {ds.seq_ids}")
    sample = ds[0]
    print("seq_id:", sample.seq_id)
    print("frames:", sample.frames.shape, sample.frames.dtype)
    print("depths:", sample.depths.shape, "range:", sample.depths.min(), sample.depths.max())
    print("traj_3d:", sample.traj_3d.shape)
    print("coords_2d:", sample.coords_2d.shape)
    print("visibility:", sample.visibility.shape, "fraction visible:", sample.visibility.mean())
    print("intrinsics[0]:\n", sample.intrinsics[0])
    print("extrinsics[0]:\n", sample.extrinsics[0])
