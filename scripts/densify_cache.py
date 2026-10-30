"""
A copy of a feature cache with every true track a clip has, not 256 of them.

Kubric gives 2048 tracks a clip; the cache kept 256 visible in frame 0. A
model that groups points into rigid pieces needs many on each piece
(docs/tasks/rigid-motion-2026-10-30.md). Each clip here keeps everything of
the source clip that does not depend on which points there are -- the DINOv3
patch tokens, the patch positions, the cameras, the scale -- and takes every
track visible in frame 0 from the raw data, read and reframed exactly as the
prepare step reads them (`KubricSequenceDataset`, `transform._to_frame0`).
Each new point's DINOv3 card is read from the cached frame-0 tokens with the
prepare step's own lookup, so DINOv3 does not run again.

Resumable: clips already in the destination are skipped.

Run:  python scripts/densify_cache.py --src CACHE --raw DATA --dst DENSE
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import torch

from genpoint3d.CachedClip import CachedClip
from genpoint3d.prepare.KubricSequenceDataset import KubricSequenceDataset
from genpoint3d.prepare.transform import _to_frame0
from genpoint3d.tools.feature_lookup import sample_at


def densify(job) -> str:
    src, raw, dst = job
    clip = CachedClip.from_dict(torch.load(src, weights_only=False))
    sample = KubricSequenceDataset(raw, [clip.seq_id], num_query_points=None)[0]
    keep = torch.from_numpy(sample.visibility[0]).bool()
    traj = _to_frame0(torch.from_numpy(sample.traj_3d).float(),
                      torch.from_numpy(sample.extrinsics).float())[:, keep]     # (T, N, 3) metres
    # The same reframing must give back the cached points exactly: a check
    # that this reads the clip the way the prepare step did.
    old = clip.anchor[None]
    gap = (traj[0][:, None] - old).norm(dim=-1).min(dim=0).values.max().item()
    if gap > 1e-3:
        raise ValueError(f"{clip.seq_id}: a cached point is {gap:.1e} m from every raw one")
    clip.traj = traj
    clip.anchor = traj[0].clone()
    clip.visibility = torch.from_numpy(sample.visibility).bool()[:, keep]
    clip.query_uv = torch.from_numpy(sample.coords_2d[0]).float()[keep]
    T, P, D = clip.context.shape
    g = int(round(P ** 0.5))
    feat0 = clip.context[:1].float().transpose(1, 2).reshape(1, D, g, g)
    clip.id_card = sample_at(feat0, clip.query_uv[None], clip.hw)[0].half()
    clip.points = int(keep.sum())
    tmp = dst / f".{clip.seq_id}.tmp"
    torch.save(clip.to_dict(), tmp)
    tmp.rename(dst / f"{clip.seq_id}.pt")
    return f"{clip.seq_id} {clip.points}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="the feature cache to copy")
    p.add_argument("--raw", required=True, help="raw Kubric root it was built from")
    p.add_argument("--dst", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    clips = sorted(Path(args.src).glob("*.pt"))
    if args.limit:
        clips = clips[: args.limit]
    todo = [(c, args.raw, dst) for c in clips if not (dst / c.name).exists()]
    print(f"{len(clips)} clips, {len(todo)} to do", flush=True)
    start, counts = time.time(), []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(densify, todo), 1):
            counts.append(int(r.split()[1]))
            if i % 200 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {(time.time() - start) / i:.2f} s a clip;"
                      f" points a clip: min {min(counts)} median {sorted(counts)[len(counts) // 2]}",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
