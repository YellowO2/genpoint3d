"""
Cache the step-0 transform to a single tensor file.

Loading a clip means decoding 48 PNGs (24 RGB + 24 depth), ~1.3 s. For step 2
there are no images in the model at all, so all of that work is thrown away
except the trajectory, the anchor and the visibility flags -- a few hundred KB
per clip. Decoding them fresh every epoch would leave the GPU idle ~25x longer
than it computes.

So: do it once, write tensors, train from those.

One file per clip, written as it goes. A run that dies partway keeps
everything finished so far and resumes on the next submission, and training
can start on a partial set.

Caveat: the cached normalisation uses statistics from the WHOLE clip, not just
the observed frames. That is a small future-leak. It is acceptable right now
because the first experiment is pure TRACKING -- every frame keeps its image,
so T_C = T and there is no "future" being withheld. Revisit before training
with random cutoffs.

Visual features are cached in fp16 at 384px (the resolution the paper's own
ablations use, §3.4), RAW at the backbone's 384-wide output. Re-encoding them
every epoch would dominate training time; projecting them before caching would
freeze a layer that has to learn. About 11 MB per clip, 37 GB for 3500.

Run:  python scripts/preprocess.py --root DATA --out local/cache/kubric --features
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from genpoint3d.data.cache import CachedClip, scale_stats
from genpoint3d.data.kubric import KubricSequenceDataset
from genpoint3d.data.transform import transform
from genpoint3d.geometry import batch_unproject


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="local/cache/kubric", help="DIRECTORY, one .pt per clip")
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--features", action="store_true", help="also cache DINOv3 features")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--stub", action="store_true", help="random backbone, for testing without DINOv3 access")
    p.add_argument("--limit", type=int, default=None, help="only the first N complete clips")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = KubricSequenceDataset(args.root, num_query_points=args.points)
    found = len(ds.seq_ids)

    # No up-front completeness scan. A half-downloaded clip raises when we try
    # to read it, and the loop below skips it -- so the cost is proportional to
    # the number of BROKEN clips, not to the size of the dataset. Scanning 3500
    # clips to find the zero-or-two bad ones was the wrong trade.
    cached = {p.stem for p in out.glob("*.pt")}
    ds.seq_ids = [s for s in ds.seq_ids if s not in cached]
    print(f"{found} clips on disk, {len(cached)} already cached,"
          f" {len(ds.seq_ids)} to encode", flush=True)

    if args.limit:
        ds.seq_ids = ds.seq_ids[: args.limit]
    if not ds.seq_ids:
        print("nothing to do")
        return 1

    encoder = None
    if args.features:
        import torch as _t
        from genpoint3d.models.encoder import VisualEncoder

        dev = _t.device("cuda" if _t.cuda.is_available() else "cpu")
        # No width argument: the cache takes the backbone's own width, because a
        # projection to the model's width would be an untrainable layer sitting
        # in front of the data. `train.py` reads `feat_dim` off the cache.
        encoder = VisualEncoder(image_size=args.image_size, stub=args.stub).to(dev).eval()
        print(f"encoding on {dev} at {args.image_size}px, {encoder.dim}-wide features"
              f"{' (STUB backbone)' if args.stub else ''}", flush=True)
    todo = list(range(len(ds)))

    t0, skipped = time.time(), []
    for n, i in enumerate(todo, 1):
        try:
            raw = ds[i]
        except Exception as e:
            # Almost always an incomplete download. Nothing is written for this
            # clip, so a later run retries it for free once the files arrive.
            skipped.append(ds.seq_ids[i])
            print(f"  SKIP {ds.seq_ids[i]}: {type(e).__name__}: {e}", flush=True)
            continue

        x = transform(raw, num_context_frames=raw.frames.shape[0])
        # METRES, not model units. Normalisation is applied when a clip is
        # loaded, so changing the scheme never invalidates the cache again.
        pointmap = batch_unproject(x.depths, x.intrinsics, x.extrinsics)
        clip = CachedClip(
            seq_id=x.seq_id,
            traj=x.traj_metric.clone(),       # (T, N, 3) metres <- TARGET
            anchor=x.traj_metric[0].clone(),  # (N, 3) frame-0 position, metres
            visibility=x.visibility.clone(),  # (T, N) bool
            query_uv=x.query_uv.clone(),      # (N, 2) where to read the ID card
            hw=tuple(x.frames.shape[1:3]),    # needed to map uv into the grid
            intrinsics=x.intrinsics.clone(),  # (T, 3, 3) pixel units
            extrinsics=x.extrinsics.clone(),  # (T, 4, 4) cam_0 -> cam_t
            stats=scale_stats(pointmap),      # every candidate scale, measured once
            points=args.points,
            feat_dim=encoder.dim if encoder is not None else None,
            image_size=args.image_size if args.features else None,
        )

        if encoder is not None:
            with torch.no_grad():
                feat = encoder(x.frames.to(dev))                              # (T,D,g,g)
                # The ID card is sampled ONCE, at the query's own frame (frame 0),
                # per the paper's "unique starting context" -- not per frame.
                uv0 = x.query_uv[None].to(dev)
                id_card = encoder.sample_at(feat[:1], uv0, clip.hw)[0]        # (N, D)
                # METRES, like `traj`. The load path normalises it with the same
                # statistics as the anchor, so patch positions and point
                # positions land in one shared space whatever scheme is chosen.
                xyz = encoder.patch_xyz(pointmap.to(dev))                     # (T, P, 3)
            clip.context = encoder.tokens(feat).half().cpu()                  # (T, P, D)
            clip.id_card = id_card.half().cpu()                               # (N, D)
            clip.patch_xyz = xyz.float().cpu()                                # (T, P, 3)

        # write to a temp name first so a kill mid-write cannot leave a
        # half-file that the resume check would mistake for finished
        tmp = out / f".{clip.seq_id}.tmp"
        torch.save(clip.to_dict(), tmp)
        tmp.rename(out / f"{clip.seq_id}.pt")

        if n % 25 == 0 or n == len(todo):
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)}  {rate:.2f}s/clip"
                  f"  eta {rate * (len(todo) - n) / 60:.1f} min", flush=True)

    if skipped:
        print(f"\nskipped {len(skipped)} unreadable clips: {' '.join(skipped[:20])}"
              f"{' ...' if len(skipped) > 20 else ''}", flush=True)

    files = sorted(out.glob("*.pt"))
    mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"\n{len(files)} clips cached in {out}"
          f"  ({mb:.1f} MB, {mb / max(len(files),1):.2f} MB/clip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
