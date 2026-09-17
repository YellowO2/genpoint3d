"""
Backfill `norm` and `intrinsics` into a cache written before they were stored.

The cached trajectory is in model units, and the per-clip mean/scale that turn
it back into metres were computed during preprocessing and then thrown away.
Without them no distance metric means anything, because one threshold is a
different physical distance in every clip.

Rebuilding the whole cache would work, but it redoes the DINOv3 pass for
nothing and re-draws the random point subset, so the cache would no longer
match the one the current checkpoint was trained on. This only reads what the
statistics actually need:

    <seq>.npy        intrinsics, extrinsics, depth_range   (small)
    frames/*_depth.png                                     (24 files, no RGB)

so it is CPU-only and touches half the images preprocessing does.

Run:  python scripts/patch_cache.py --cache local/cache/kubric --root local/data/kubric
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from genpoint3d.data.cache import CachedClip
from genpoint3d.data.kubric import _AXIS_FLIP, _distance_to_depth, _imread
from genpoint3d.data.transform import (
    compute_norm_stats,
    relative_extrinsics,
)
from genpoint3d.geometry import batch_unproject

import cv2


def _depths_and_intrinsics(seq_dir: Path, seq_id: str, hw: tuple[int, int]):
    """Everything `compute_norm_stats` needs, without decoding a single RGB frame.

    `hw` comes from the cache, which is why the frame size does not have to be
    recovered by reading an image.
    """
    data = np.load(seq_dir / f"{seq_id}.npy", allow_pickle=True).item()
    depth_range = data["depth_range"]
    T = data["intrinsics"].shape[0]
    H, W = hw

    # Same scaling as KubricSequenceDataset: intrinsics are stored normalised.
    intrinsics = np.abs(data["intrinsics"]).astype(np.float32)
    intrinsics[:, 0, :] *= W
    intrinsics[:, 1, :] *= H

    frames_dir = seq_dir / "frames"

    def one(t: int) -> np.ndarray:
        png = _imread(str(frames_dir / f"{t:03d}_depth.png"), cv2.IMREAD_UNCHANGED)
        return depth_range[0] + png.astype(np.float32) * (
            depth_range[1] - depth_range[0]
        ) / 65535.0

    with ThreadPoolExecutor(min(16, T)) as ex:
        distances = np.stack(list(ex.map(one, range(T))))

    depths = _distance_to_depth(distances, intrinsics)

    extrinsics_world = np.linalg.inv(data["matrix_world"]).astype(np.float32)
    extrinsics_world = np.einsum("ij,tjk->tik", _AXIS_FLIP, extrinsics_world)
    return (torch.from_numpy(depths).float(),
            torch.from_numpy(intrinsics).float(),
            torch.from_numpy(extrinsics_world).float())


def patch_one(pt_path: Path, root: Path) -> str | None:
    """Add `norm` and `intrinsics` to one cached clip. Returns a reason if skipped."""
    clip = CachedClip.from_dict(torch.load(pt_path, weights_only=False))
    if clip.norm is not None and clip.intrinsics is not None:
        return "already patched"

    seq_dir = root / clip.seq_id
    if not seq_dir.is_dir():
        return "raw clip missing"

    depths, intrinsics, extrinsics_world = _depths_and_intrinsics(
        seq_dir, clip.seq_id, clip.hw
    )
    extrinsics = relative_extrinsics(extrinsics_world)

    # The cache was built in pure tracking mode, so every frame is observed and
    # the statistics see the whole clip -- matching how `transform` computed
    # them originally. Revisit alongside the random-cutoff work.
    pointmap = batch_unproject(depths, intrinsics, extrinsics)
    clip.norm = compute_norm_stats(pointmap)
    clip.intrinsics = intrinsics

    tmp = pt_path.with_name(f".{pt_path.stem}.tmp")
    torch.save(clip.to_dict(), tmp)
    tmp.rename(pt_path)
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="cache DIRECTORY to patch in place")
    p.add_argument("--root", required=True, help="raw Kubric clips")
    p.add_argument("--workers", type=int, default=8,
                   help="clips in flight; the work is dominated by file reads")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cache, root = Path(args.cache), Path(args.root)
    files = sorted(cache.glob("*.pt"))
    if not files:
        raise SystemExit(f"no cached clips in {cache}")
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} cached clips to check", flush=True)

    t0, done, skipped = time.time(), 0, {}

    def run(f: Path):
        try:
            return f, patch_one(f, root)
        except Exception as e:
            return f, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(args.workers) as ex:
        for n, (f, reason) in enumerate(ex.map(run, files), 1):
            if reason is None:
                done += 1
            else:
                skipped.setdefault(reason, []).append(f.stem)
            if n % 100 == 0 or n == len(files):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(files)}  {rate:.2f}s/clip"
                      f"  eta {rate * (len(files) - n) / 60:.1f} min", flush=True)

    print(f"\npatched {done}/{len(files)}", flush=True)
    for reason, ids in skipped.items():
        print(f"  {len(ids)} skipped -- {reason}: {' '.join(ids[:10])}"
              f"{' ...' if len(ids) > 10 else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
