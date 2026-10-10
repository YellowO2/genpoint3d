"""
Cache each clip's RGB frames, small, beside an existing feature cache.

The feature cache holds what DINOv3 made of the frames and not the frames. The
relative finder (`genpoint3d/models/fine.py`) trains a CNN on the pixels, so
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

The feature cache is only listed, never opened or written: it says WHICH clips.

Resumable: clips already in the destination are skipped.

Run:  python scripts/cache_frames.py --src CACHE --raw DATA --dst FRAMES --res 384
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np

from genpoint3d.data.kubric import load_frames


def cache_one(job: tuple[str, str, str, int]) -> dict:
    seq, raw, dst, res = job
    dst = Path(dst)
    try:
        frames = load_frames(raw, seq)                                # (T, H, W, 3)
        how = cv2.INTER_AREA if res < min(frames.shape[1:3]) else cv2.INTER_LINEAR
        small = np.stack([cv2.resize(f, (res, res), interpolation=how) for f in frames])
        # temp name first, so a kill mid-write cannot leave a half-file that
        # the resume check (or a training job) would take for finished
        tmp = dst / f".{seq}.{os.getpid()}.tmp.npy"
        np.save(tmp, small)
        tmp.rename(dst / f"{seq}.npy")
        return {"seq": seq, "shape": small.shape}
    except Exception as e:  # one bad clip must not sink the other 3499
        return {"seq": seq, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="feature cache directory; names the clips (read only)")
    p.add_argument("--raw", required=True, help="raw Kubric root the cache was built from")
    p.add_argument("--dst", required=True, help="frames directory to write")
    p.add_argument("--res", type=int, default=384,
                   help="side of the square frame, which is train.py's --fine-res")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--limit", type=int, default=None, help="only the first N clips still to do")
    args = p.parse_args()

    clips = sorted(c.stem for c in Path(args.src).glob("*.pt"))
    if not clips:
        raise SystemExit(f"no cached clips in {args.src}")
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    done = {f.stem for f in dst.glob("*.npy") if not f.name.startswith(".")}
    if done:
        # Resuming at another size would leave a cache the trainer refuses.
        got = np.load(dst / f"{sorted(done)[0]}.npy", mmap_mode="r").shape[1]
        if got != args.res:
            raise SystemExit(f"{dst} holds {got}px frames and --res is {args.res}."
                             " Use a different --dst.")
    todo = [c for c in clips if c not in done]
    print(f"{len(clips)} clips in {args.src}, {len(done & set(clips))} already in {dst},"
          f" {len(todo)} to write at {args.res}px", flush=True)
    if args.limit:
        todo = todo[: args.limit]

    jobs = [(c, args.raw, str(dst), args.res) for c in todo]
    workers = max(1, min(args.workers, len(jobs)))
    t0, failed = time.time(), []
    with mp.Pool(workers) as pool:
        for n, r in enumerate(pool.imap(cache_one, jobs), 1):
            if "error" in r:
                failed.append(r["seq"])
                print(f"  FAIL {r['seq']}: {r['error']}", flush=True)
            elif n == 1:
                print(f"  {r['seq']}  {tuple(r['shape'])} uint8", flush=True)
            if n % 100 == 0 or n == len(jobs):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(jobs)}  {rate:.2f}s/clip"
                      f"  eta {rate * (len(jobs) - n) / 60:.1f} min", flush=True)

    if failed:
        print(f"\n{len(failed)} clips FAILED and were not written: {' '.join(failed[:20])}"
              f"{' ...' if len(failed) > 20 else ''}")

    have = {f.stem for f in dst.glob("*.npy") if not f.name.startswith(".")}
    n_dst = len(have & set(clips))
    print(f"\n{n_dst} of {len(clips)} clips in {dst}")
    if n_dst != len(clips):
        print("INCOMPLETE: train.py --fine 1 refuses a frames cache that is missing"
              " a clip. Re-run to resume.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
