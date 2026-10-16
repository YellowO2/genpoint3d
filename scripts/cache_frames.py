"""
Cache each clip's RGB frames, small, beside an existing feature cache.

The feature cache holds what DINOv3 made of the frames and not the frames. The
relative finder (`genpoint3d/train/model/FineFinder.py`) trains a CNN on the pixels, so
training needs them again -- and decoding 24 PNGs per clip per step off Lustre
is what the feature cache was built to avoid.

One `<clip>.npy` per clip of the source cache: (T, res, res, 3) uint8, the
native frame squashed to a square exactly as it is for DINOv3. 10.6 MB a clip
at 384 px, 37 GB for 3500. A plain array rather than a `.pt`, so it loads
without unpickling and its shape can be read without loading it.

Resized by area averaging, which is the right filter for shrinking: every
source pixel counts once. Either way the frame's edges stay its edges, so a
point at a fraction `u` of the way across is at the same fraction afterwards,
which is the only thing the model's projection assumes.

Beside it, `<clip>.depth.npy`: (T, res / 4, res / 4) float32, the z-depth in
metres at each cell of the CNN's map, for `--fine-xyz 1` (0.9 MB a clip at
384 px). Cell j is centred on pixel INDEX 4j of the small frame, which is
`(4j + 0.5) / res` across, and the depth is the native pixel holding that spot
-- one pixel read, nothing averaged, so a cell on an object's edge gets one of
the two surfaces and not a depth in mid-air between them (as
`VisualEncoder.patch_centre_xyz` does for the 16 px patches). The training
side turns it into a 3D position (`tools/batch.py`'s `fine_inputs`).

The feature cache is only listed, never opened or written: it says WHICH clips.

Resumable: clips already in the destination are skipped, and a clip that has
its frames but no depth (a cache from before the depth) only gets its depth.

Run:  python scripts/cache_frames.py --src CACHE --raw DATA --dst FRAMES --res 384
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np

from genpoint3d.prepare.KubricSequenceDataset import KubricSequenceDataset
from genpoint3d.train.data.ClipDataset import DEPTH

# Pixels of the small frame per cell of the CNN's map (`BasicEncoder.STRIDE`).
STRIDE = 4


def cell_depth(depth: np.ndarray, res: int) -> np.ndarray:
    """(T, H, W) native z-depth -> (T, res / 4, res / 4): the native pixel under
    the centre of each CNN cell of a res x res frame.

    Cell j is centred at `(4j + 0.5) / res` across, inside native pixel
    `floor((8j + 1) W / (2 res))`. Integer arithmetic, so the pick cannot
    flip with float rounding."""
    H, W = depth.shape[1:3]
    j = np.arange(res // STRIDE)
    rows = ((2 * STRIDE * j + 1) * H) // (2 * res)
    cols = ((2 * STRIDE * j + 1) * W) // (2 * res)
    return depth[:, rows][:, :, cols].astype(np.float32)


def _save(array: np.ndarray, dst: Path, name: str) -> None:
    # temp name first, so a kill mid-write cannot leave a half-file that
    # the resume check (or a training job) would take for finished
    tmp = dst / f".{name}.{os.getpid()}.tmp.npy"
    np.save(tmp, array)
    tmp.rename(dst / f"{name}.npy")


def cache_one(job: tuple[str, str, str, int, bool]) -> dict:
    seq, raw, dst, res, frames_too = job
    dst = Path(dst)
    try:
        shape = None
        if frames_too:
            frames = KubricSequenceDataset.load_frames(raw, seq)                            # (T, H, W, 3)
            how = cv2.INTER_AREA if res < min(frames.shape[1:3]) else cv2.INTER_LINEAR
            small = np.stack([cv2.resize(f, (res, res), interpolation=how) for f in frames])
            _save(small, dst, seq)
            shape = small.shape
        depth = cell_depth(KubricSequenceDataset.load_depths(raw, seq), res)
        _save(depth, dst, seq + DEPTH.removesuffix(".npy"))
        return {"seq": seq, "shape": shape, "depth": depth.shape}
    except Exception as e:  # one bad clip must not sink the other 3499
        return {"seq": seq, "error": f"{type(e).__name__}: {e}"}


def _listed(dst: Path) -> tuple[set[str], set[str]]:
    """The clips in `dst` with frames, and those with depth."""
    names = [f.name for f in dst.glob("*.npy") if not f.name.startswith(".")]
    depth = {n.removesuffix(DEPTH) for n in names if n.endswith(DEPTH)}
    frames = {n.removesuffix(".npy") for n in names if not n.endswith(DEPTH)}
    return frames, depth


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="feature cache directory; names the clips (read only)")
    p.add_argument("--raw", required=True, help="raw Kubric root the cache was built from")
    p.add_argument("--dst", required=True, help="frames directory to write")
    p.add_argument("--res", type=int, default=384,
                   help="side of the square frame, which is training's --fine-res")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--limit", type=int, default=None, help="only the first N clips still to do")
    args = p.parse_args()

    clips = sorted(c.stem for c in Path(args.src).glob("*.pt"))
    if not clips:
        raise SystemExit(f"no cached clips in {args.src}")
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    framed, depthed = _listed(dst)
    if framed:
        # Resuming at another size would leave a cache the trainer refuses.
        got = np.load(dst / f"{sorted(framed)[0]}.npy", mmap_mode="r").shape[1]
        if got != args.res:
            raise SystemExit(f"{dst} holds {got}px frames and --res is {args.res}."
                             " Use a different --dst.")
    done = framed & depthed
    todo = [c for c in clips if c not in done]
    print(f"{len(clips)} clips in {args.src}, {len(done & set(clips))} already in {dst},"
          f" {len(todo)} to write at {args.res}px"
          f" ({sum(c in framed for c in todo)} of them only their depth)", flush=True)
    if args.limit:
        todo = todo[: args.limit]

    jobs = [(c, args.raw, str(dst), args.res, c not in framed) for c in todo]
    workers = max(1, min(args.workers, len(jobs)))
    t0, failed = time.time(), []
    with mp.Pool(workers) as pool:
        for n, r in enumerate(pool.imap(cache_one, jobs), 1):
            if "error" in r:
                failed.append(r["seq"])
                print(f"  FAIL {r['seq']}: {r['error']}", flush=True)
            elif n == 1:
                frames = f"{tuple(r['shape'])} uint8 and " if r["shape"] else ""
                print(f"  {r['seq']}  {frames}{tuple(r['depth'])} float32 depth", flush=True)
            if n % 100 == 0 or n == len(jobs):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(jobs)}  {rate:.2f}s/clip"
                      f"  eta {rate * (len(jobs) - n) / 60:.1f} min", flush=True)

    if failed:
        print(f"\n{len(failed)} clips FAILED and were not written: {' '.join(failed[:20])}"
              f"{' ...' if len(failed) > 20 else ''}")

    framed, depthed = _listed(dst)
    n_dst = len(framed & depthed & set(clips))
    print(f"\n{n_dst} of {len(clips)} clips in {dst} with frames and depth")
    if n_dst != len(clips):
        print("INCOMPLETE: training with --fine 1 refuses a frames cache that is missing"
              " a clip, and --fine-xyz 1 one missing a clip's depth. Re-run to resume.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
