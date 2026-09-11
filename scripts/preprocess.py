"""
Cache the step-0 transform to a single tensor file.

Loading a clip means decoding 48 PNGs (24 RGB + 24 depth), ~1.3 s. For step 2
there are no images in the model at all, so all of that work is thrown away
except the trajectory, the anchor and the visibility flags -- a few hundred KB
per clip. Decoding them fresh every epoch would leave the GPU idle ~25x longer
than it computes.

So: do it once, write tensors, train from those.

Caveat: the cached normalisation uses statistics from the WHOLE clip, not just
the observed frames. That is a small future-leak, and it is fine only because
the model currently sees no images -- the cutoff T_C has no other effect. When
step 3 adds visual conditioning this must be revisited (cache per cutoff, or
cache metric coordinates and normalise on the fly).

Run:  python scripts/preprocess.py --root DATA --out cache/kubric.pt
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from genpoint3d.data.kubric import KubricSequenceDataset
from genpoint3d.data.transform import transform


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="cache/kubric.pt")
    p.add_argument("--points", type=int, default=256)
    args = p.parse_args()

    ds = KubricSequenceDataset(args.root, num_query_points=args.points)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    clips, t0 = [], time.time()
    for i in range(len(ds)):
        raw = ds[i]
        x = transform(raw, num_context_frames=raw.frames.shape[0])
        clips.append({
            "seq_id": x.seq_id,
            "traj": x.traj.clone(),             # (T, N, 3) normalised, the target
            "anchor": x.anchor.clone(),         # (N, 3) conditioning
            "visibility": x.visibility.clone(),  # (T, N) bool
            "query_uv": x.query_uv.clone(),     # (N, 2) kept for step 3
        })
        if (i + 1) % 25 == 0 or i + 1 == len(ds):
            rate = (time.time() - t0) / (i + 1)
            print(f"  {i + 1}/{len(ds)}  {rate:.2f}s/clip"
                  f"  eta {rate * (len(ds) - i - 1) / 60:.1f} min", flush=True)

    torch.save({"clips": clips, "points": args.points}, out)
    mb = out.stat().st_size / 1e6
    print(f"\n{len(clips)} clips -> {out}  ({mb:.1f} MB, {mb / max(len(clips),1):.2f} MB/clip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
