"""Each old Kubric point's object and that object's centre of mass, from its
MOVi-F twin (scripts/movi_objects.py): the object most often shown at the
point's pixel where it is seen. Checked by the object's own motion: a point of
an object keeps its place in that object's frame. Written into the raw clip
(`segment`, `centre_3d`, as our renderer writes them) and, with --caches, into
each cached clip's points.

  python scripts/label_old_kubric.py --old ~/scratch/kubric --objects DIR [--caches DIR ...] [--write]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kubric_tracks import quat_to_rot  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--old", required=True)
p.add_argument("--objects", required=True)
p.add_argument("--caches", nargs="*", default=[])
p.add_argument("--write", action="store_true", help="save the labels (without: only the check)")
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()
old, objects = Path(args.old).expanduser(), Path(args.objects).expanduser()

share, kept, clips = [], [], 0
for f in sorted(objects.glob("*.npz"))[: args.limit]:
    o = np.load(f)
    path = old / f.stem / f"{f.stem}.npy"
    raw = np.load(path, allow_pickle=True).item()
    xy, seen, traj = raw["coords"], ~raw["occluded"], raw["traj_3d"].astype(np.float64)   # (N, T, 2) px, (N, T), (N, T, 3)
    seg = o["seg"]                                                                        # (T, h, w)
    T, h, w = seg.shape
    side = 2 * w                                                                           # the old clips are the 512 px render
    # Pixel j of the 256 px video is pixel 2j of the 512 px render.
    u = np.clip(np.floor(xy[..., 0] / side * w).astype(int), 0, w - 1)
    v = np.clip(np.floor(xy[..., 1] / side * h).astype(int), 0, h - 1)
    shown = seg[np.arange(T)[None], v, u]                                                  # (N, T)
    K = len(o["positions"])
    votes = np.stack([((shown == k) & seen).sum(1) for k in range(K + 1)], 1)              # (N, K + 1)
    label = votes.argmax(1)
    share.append(votes.max(1)[seen.any(1)] / seen.sum(1)[seen.any(1)])
    # Its place in its object's frame, which a point of the object keeps (the background: the world).
    local = traj.copy()
    centre = np.full(traj.shape, np.nan, np.float32)
    for k in range(1, K + 1):
        R = quat_to_rot(o["quaternions"][k - 1].astype(np.float64))                        # (T, 3, 3)
        m = label == k
        local[m] = np.einsum("tji,ntj->nti", R, traj[m] - o["positions"][k - 1][None])
        centre[m] = o["positions"][k - 1][None]
    moved = np.linalg.norm(local - local.mean(1, keepdims=True), axis=-1).max(1)           # (N,) metres
    kept.append(moved)
    clips += 1
    if args.write:
        raw["segment"], raw["centre_3d"] = label.astype(np.int32), centre
        np.save(path, raw)
        for c in args.caches:
            pt = Path(c).expanduser() / f"{f.stem}.pt"
            if not pt.exists():
                continue
            clip = torch.load(pt, weights_only=False)
            get = (lambda k: clip[k]) if isinstance(clip, dict) else (lambda k: getattr(clip, k))
            e0 = np.linalg.inv(raw["matrix_world"][0])
            flip = np.diag([1.0, -1.0, -1.0])
            world_to_cam0 = lambda x: (x @ e0[:3, :3].T + e0[:3, 3]) @ flip.T              # Blender's axes to OpenCV's
            mine = get("traj").double()                                                    # (T, n, 3) frame-0 camera
            raw0 = torch.from_numpy(world_to_cam0(traj)).transpose(0, 1)                   # (T, N, 3)
            d = torch.cdist(mine.transpose(0, 1).flatten(1), raw0.transpose(0, 1).flatten(1))
            near, idx = d.min(1)
            if float(near.max()) > 1e-2:
                raise SystemExit(f"{pt}: a cached point matches no raw point ({float(near.max()):.3f})")
            segment = torch.from_numpy(label)[idx].long()
            cen = torch.from_numpy(world_to_cam0(centre.astype(np.float64))).transpose(0, 1)[:, idx].float()
            if isinstance(clip, dict):
                clip["segment"], clip["centre"] = segment, cen
            else:
                clip.segment, clip.centre = segment, cen
            torch.save(clip, pt)

share, kept = np.concatenate(share), np.concatenate(kept)
print(f"{clips} clips, {len(kept)} points")
print(f"  the frames a seen point is shown on its object: median {np.median(share):.3f}, under 0.8 for {(share < 0.8).mean():.3%} of points")
print(f"  how far a point moves in its object's frame, cm: median {np.median(kept) * 100:.2f}, "
      f"90% under {np.percentile(kept, 90) * 100:.2f}, 99% under {np.percentile(kept, 99) * 100:.2f}, "
      f"over 2 cm {(kept > 0.02).mean():.3%}")
