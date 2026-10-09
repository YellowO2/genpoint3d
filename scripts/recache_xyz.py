"""
Rewrite `patch_xyz` in an existing cache, without re-running DINOv3.

Caches written before 2026-10-10 store each patch's 3D position as the AVERAGE
of the pointmap over the patch. At an object edge that average sits in mid-air
between foreground and background. `patch_centre_xyz` now reads the one pixel
at the patch centre instead, and this script applies that to clips that are
already cached -- the features in them took GPU-hours and have not changed.

Each clip is copied to a NEW directory with every field untouched except
`patch_xyz`. The source is only ever read: a training job may be using it.

What is needed from the raw data is the depth maps and nothing else. The
pointmap is rebuilt exactly as `preprocess.py` built it -- native-resolution
depth, unprojected with the intrinsics and extrinsics stored in the clip
itself, in metres -- and the grid size is read off the cached `patch_xyz`, so
24x24 and 48x48 caches both work with no flag.

Before a clip is written, the OLD rule is applied to the rebuilt pointmap and
compared with what the cache holds. If they disagree the pointmap is not the
one the cache was built from (wrong --raw, a re-downloaded clip) and the clip
is refused rather than written with positions from a different scene.

Nothing else in a clip depends on `patch_xyz`: the scale statistics are
measured from the full pointmap, and the ID card and trajectory never touch it.

Clip names are kept, and `train.py` splits by position in the sorted name list,
so a complete destination has the same train/val split as its source. An
INCOMPLETE one does not -- the script says so if the counts differ.

Resumable: clips already in the destination are skipped.

Run:  python scripts/recache_xyz.py --src CACHE --raw DATA --dst NEW_CACHE
"""

import argparse
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from genpoint3d.data.kubric import load_depths
from genpoint3d.geometry import batch_unproject
from genpoint3d.models.encoder import patch_centre_xyz


def _init() -> None:
    # One clip per process is the parallelism; torch threads on top of that
    # only fight each other for the same cores.
    torch.set_num_threads(1)


def recache_one(job: tuple[str, str, str, float]) -> dict:
    src, raw, dst, tol = job
    src, dst = Path(src), Path(dst)
    seq = src.stem
    try:
        d = torch.load(src, weights_only=False)
        old = d.get("patch_xyz")
        if old is None:
            raise ValueError("no patch_xyz in this clip -- nothing to rewrite")
        T, P, _ = old.shape
        grid = math.isqrt(P)
        if grid * grid != P:
            raise ValueError(f"{P} patches is not a square grid")

        depths = torch.from_numpy(load_depths(raw, seq)).float()          # (T, H, W) metres
        if depths.shape[0] != T or tuple(depths.shape[1:]) != tuple(d["hw"]):
            raise ValueError(f"raw depth is {tuple(depths.shape)}, cache expects"
                             f" ({T}, {d['hw'][0]}, {d['hw'][1]})")
        pointmap = batch_unproject(depths, d["intrinsics"].float(), d["extrinsics"].float())

        # The old rule on the rebuilt pointmap must give back what is cached.
        # This is the only proof that old and new differ by the rule alone.
        avg = F.adaptive_avg_pool2d(pointmap, grid).flatten(2).transpose(1, 2)
        rebuilt_err = (avg - old.float()).norm(dim=-1).max().item()
        if not rebuilt_err <= tol:
            raise ValueError(f"rebuilt pointmap does not reproduce the cached patch_xyz"
                             f" (worst patch {rebuilt_err:.4f} m off, tolerance {tol} m)"
                             " -- is --raw the data this cache was built from?")

        new = patch_centre_xyz(pointmap, grid).to(old.dtype).contiguous()
        moved = (new - old).float().norm(dim=-1)
        d["patch_xyz"] = new

        # temp name first, so a kill mid-write cannot leave a half-file that
        # the resume check (or a training job's glob) would take for finished
        tmp = dst.parent / f".{seq}.{os.getpid()}.tmp"
        torch.save(d, tmp)
        tmp.rename(dst)
        return {"seq": seq, "grid": grid, "median": moved.median().item(),
                "frac5": (moved > 0.05).float().mean().item(), "rebuilt_err": rebuilt_err}
    except Exception as e:  # one bad clip must not sink the other 3499
        return {"seq": seq, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="existing cache directory (read only)")
    p.add_argument("--raw", required=True, help="raw Kubric root the cache was built from")
    p.add_argument("--dst", required=True, help="NEW cache directory")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--report", type=int, default=5, help="print the old-vs-new check for the first N clips")
    p.add_argument("--limit", type=int, default=None, help="only the first N clips still to do")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="metres; how closely the rebuilt pointmap must reproduce the cached patch_xyz")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    clips = sorted(src.glob("*.pt"))
    if not clips:
        raise SystemExit(f"no cached clips in {src}")
    dst.mkdir(parents=True, exist_ok=True)
    if dst.resolve() == src.resolve():
        raise SystemExit("--dst is the source cache. This script never rewrites in place.")

    done = {f.name for f in dst.glob("*.pt")}
    todo = [c for c in clips if c.name not in done]
    print(f"{len(clips)} clips in {src}, {len(done)} already in {dst}, {len(todo)} to rewrite",
          flush=True)
    if args.limit:
        todo = todo[: args.limit]

    jobs = [(str(c), args.raw, str(dst / c.name), args.tol) for c in todo]
    workers = max(1, min(args.workers, len(jobs)))
    t0, ok, failed = time.time(), [], []
    if jobs:
        print(f"{workers} workers. First {args.report} clips, old (patch average) vs new (patch centre):",
              flush=True)
    with mp.Pool(workers, initializer=_init) as pool:
        # `imap` keeps order, so the clips reported are the same ones every run.
        for n, r in enumerate(pool.imap(recache_one, jobs), 1):
            if "error" in r:
                failed.append(r["seq"])
                print(f"  FAIL {r['seq']}: {r['error']}", flush=True)
            else:
                ok.append(r)
                if len(ok) <= args.report:
                    g = r["grid"]
                    print(f"  {r['seq']}  grid {g}x{g}  median move {r['median'] * 100:.2f} cm"
                          f"  moved >5 cm: {r['frac5'] * 100:.1f}% of patches"
                          f"  (old rule reproduces cache to {r['rebuilt_err'] * 1000:.3f} mm)",
                          flush=True)
            if n % 100 == 0 or n == len(jobs):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(jobs)}  {rate:.2f}s/clip"
                      f"  eta {rate * (len(jobs) - n) / 60:.1f} min", flush=True)

    if ok:
        med = torch.tensor([r["median"] for r in ok]).median().item()
        frac = torch.tensor([r["frac5"] for r in ok]).mean().item()
        print(f"\nover {len(ok)} clips: median of per-clip median move {med * 100:.2f} cm,"
              f" {frac * 100:.1f}% of patches moved >5 cm")
    if failed:
        print(f"\n{len(failed)} clips FAILED and were not written: {' '.join(failed[:20])}"
              f"{' ...' if len(failed) > 20 else ''}")

    n_dst = len(list(dst.glob("*.pt")))
    print(f"\n{n_dst} of {len(clips)} clips in {dst}")
    if n_dst != len(clips):
        print("INCOMPLETE: train.py splits by position in the sorted clip list, so this"
              " directory does NOT have the source's train/val split yet. Re-run to resume.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
