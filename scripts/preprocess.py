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
ablations use, §3.4), which is ~10 MB per clip -- about 5 GB for 500 clips.
Re-encoding them every epoch would dominate training time.

Run:  python scripts/preprocess.py --root DATA --out cache/kubric --features
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from genpoint3d.data.kubric import KubricSequenceDataset
from genpoint3d.data.transform import scene_pointmap, transform


def _complete(root: Path, seq_ids: list[str]) -> list[str]:
    """Drop clips that are still downloading.

    A finished clip has its .npy plus a frames/ dir holding one RGB and one
    depth PNG per frame. Preprocessing a half-written clip crashes on a
    missing file, so this makes it safe to run while a download is in flight.

    Only ever call this on clips you actually intend to process -- verifying a
    clip costs one .npy read plus a directory listing, and doing that for clips
    that are already cached is pure waste on a resumed run.
    """
    import os

    import numpy as np

    ok = []
    for s in seq_ids:
        d = root / s
        npy, frames = d / f"{s}.npy", d / "frames"
        try:
            # One listing beats 2*T separate exists() calls -- on a network
            # filesystem each of those is a round trip.
            present = {e.name for e in os.scandir(frames)}
        except OSError:
            continue  # frames/ missing or unreadable
        if not npy.exists():
            continue
        try:
            # The .npy is the authority on how many frames the clip has.
            # Counting PNGs against each other is not enough: a clip stopped
            # halfway has equal RGB and depth counts and would pass.
            t = int(np.load(npy, allow_pickle=True).item()["intrinsics"].shape[0])
        except Exception:
            continue  # .npy itself truncated or unreadable
        if all(f"{i:03d}.png" in present and f"{i:03d}_depth.png" in present
               for i in range(t)):
            ok.append(s)
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="cache/kubric", help="DIRECTORY, one .pt per clip")
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--features", action="store_true", help="also cache DINOv3 features")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--feat-dim", type=int, default=256, help="must match the model's dim")
    p.add_argument("--stub", action="store_true", help="random backbone, for testing without DINOv3 access")
    p.add_argument("--limit", type=int, default=None, help="only the first N complete clips")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = KubricSequenceDataset(args.root, num_query_points=args.points)
    found = len(ds.seq_ids)

    # Drop already-cached clips BEFORE verifying the rest. Verification reads
    # each clip's .npy, so checking clips we are about to skip is the single
    # most expensive pointless thing this script could do -- and it gets worse
    # on every resume, exactly when there is least work left to justify it.
    cached = {p.stem for p in out.glob("*.pt")}
    pending = [s for s in ds.seq_ids if s not in cached]
    print(f"{found} clips on disk, {len(cached)} already cached,"
          f" verifying {len(pending)}...", flush=True)

    t_scan = time.time()
    ds.seq_ids = _complete(Path(args.root), pending)
    print(f"  {len(ds.seq_ids)} complete and pending"
          f"  ({time.time() - t_scan:.0f}s to verify)", flush=True)
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
        encoder = VisualEncoder(
            dim=args.feat_dim, image_size=args.image_size, stub=args.stub
        ).to(dev).eval()
        print(f"encoding on {dev} at {args.image_size}px"
              f"{' (STUB backbone)' if args.stub else ''}", flush=True)
    # ds.seq_ids already excludes everything cached, so this is all of it.
    todo = list(range(len(ds)))
    print(f"{len(todo)} to encode", flush=True)

    t0 = time.time()
    for n, i in enumerate(todo, 1):
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

        entry["points"] = args.points
        entry["feat_dim"] = args.feat_dim if args.features else None
        entry["image_size"] = args.image_size if args.features else None

        # write to a temp name first so a kill mid-write cannot leave a
        # half-file that the resume check would mistake for finished
        tmp = out / f".{entry['seq_id']}.tmp"
        torch.save(entry, tmp)
        tmp.rename(out / f"{entry['seq_id']}.pt")

        if n % 25 == 0 or n == len(todo):
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)}  {rate:.2f}s/clip"
                  f"  eta {rate * (len(todo) - n) / 60:.1f} min", flush=True)

    files = sorted(out.glob("*.pt"))
    mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"\n{len(files)} clips cached in {out}"
          f"  ({mb:.1f} MB, {mb / max(len(files),1):.2f} MB/clip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
