"""
Cache the step-0 transform to a single tensor file.

Loading a clip means decoding 48 PNGs (24 RGB + 24 depth), ~1.3 s. For step 2
there are no images in the model at all, so all of that work is thrown away
except the trajectory, the anchor and the visibility flags -- a few hundred KB
per clip. Decoding them fresh every epoch would leave the GPU idle ~25x longer
than it computes.

So: do it once, write tensors, train from those.

Caveat: the cached normalisation uses statistics from the WHOLE clip, not just
the observed frames. That is a small future-leak. It is acceptable right now
because the first experiment is pure TRACKING -- every frame keeps its image,
so T_C = T and there is no "future" being withheld. Revisit before training
with random cutoffs.

Visual features are cached in fp16 at 384px (the resolution the paper's own
ablations use, §3.4), which is ~10 MB per clip -- about 5 GB for 500 clips.
Re-encoding them every epoch would dominate training time.

Run:  python scripts/preprocess.py --root DATA --out cache/kubric.pt
      python scripts/preprocess.py --root DATA --out cache/k.pt --features
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from genpoint3d.data.kubric import KubricSequenceDataset
from genpoint3d.data.transform import scene_pointmap, transform


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="cache/kubric.pt")
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--features", action="store_true", help="also cache DINOv3 features")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--feat-dim", type=int, default=256, help="must match the model's dim")
    p.add_argument("--stub", action="store_true", help="random backbone, for testing without DINOv3 access")
    args = p.parse_args()

    ds = KubricSequenceDataset(args.root, num_query_points=args.points)

    encoder = None
    if args.features:
        import torch as _t
        from genpoint3d.models.encoder import VisualEncoder

        dev = _t.device("cuda" if _t.cuda.is_available() else "cpu")
        encoder = VisualEncoder(
            dim=args.feat_dim, image_size=args.image_size, stub=args.stub
        ).to(dev).eval()
        print(f"encoding on {dev} at {args.image_size}px"
              f"{' (STUB backbone)' if args.stub else ''}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    clips, t0 = [], time.time()
    for i in range(len(ds)):
        raw = ds[i]
        x = transform(raw, num_context_frames=raw.frames.shape[0])
        entry = {
            "seq_id": x.seq_id,
            "traj": x.traj.clone(),              # (T, N, 3) normalised, the target
            "anchor": x.anchor.clone(),          # (N, 3) conditioning
            "visibility": x.visibility.clone(),  # (T, N) bool
            "query_uv": x.query_uv.clone(),      # (N, 2) where to read the ID card
            "hw": tuple(x.frames.shape[1:3]),    # needed to map uv into the grid
        }

        if encoder is not None:
            with torch.no_grad():
                feat = encoder(x.frames.to(dev), scene_pointmap(x).to(dev))  # (T,D,g,g)
                # The ID card is sampled ONCE, at the query's own frame (frame 0),
                # per the paper's "unique starting context" -- not per frame.
                uv0 = x.query_uv[None].to(dev)
                id_card = encoder.sample_at(feat[:1], uv0, entry["hw"])[0]   # (N, D)
            entry["context"] = encoder.tokens(feat).half().cpu()             # (T, P, D)
            entry["id_card"] = id_card.half().cpu()                          # (N, D)

        clips.append(entry)
        if (i + 1) % 25 == 0 or i + 1 == len(ds):
            rate = (time.time() - t0) / (i + 1)
            print(f"  {i + 1}/{len(ds)}  {rate:.2f}s/clip"
                  f"  eta {rate * (len(ds) - i - 1) / 60:.1f} min", flush=True)

    torch.save({
        "clips": clips,
        "points": args.points,
        "feat_dim": args.feat_dim if args.features else None,
        "image_size": args.image_size if args.features else None,
    }, out)
    mb = out.stat().st_size / 1e6
    print(f"\n{len(clips)} clips -> {out}  ({mb:.1f} MB, {mb / max(len(clips),1):.2f} MB/clip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
