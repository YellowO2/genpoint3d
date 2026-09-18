"""
Score a trained checkpoint on a cache. No training, no checkpoint selection.

This is what a held-out TEST set is for. The val split inside `train.py` picks
`best.pt`, so its APD is mildly optimistic -- it is the score of whichever
checkpoint happened to suit those exact clips. A set that is never used to
choose anything does not have that problem, and its number is the one to report.

Sequences 3500+ were downloaded separately for this and have never been seen by
training.

Run:  python scripts/evaluate.py --ckpt outputs/run3493/best.pt \
                                 --cache ~/scratch/cache/kubric_test
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from genpoint3d.models.model import PointDiT
from genpoint3d.data.cache import CachedClip
from train import ClipDataset, evaluate


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="best.pt or ckpt.pt from a run")
    p.add_argument("--cache", required=True, help="cache DIRECTORY to score")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--points", type=int, default=None,
                   help="default: whatever the checkpoint trained with")
    p.add_argument("--sample-steps", type=int, default=50,
                   help="Euler steps when sampling; the published protocol is not"
                        " prescriptive, so report whatever you use")
    p.add_argument("--out", default=None, help="write the metrics as JSON here")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    # Every clip in the cache is the test set. split() is not reused here: it
    # always holds back at least one clip for validation, which is right for
    # training and would silently drop a test clip.
    clips = sorted(Path(args.cache).glob("*.pt"))
    if not clips:
        raise SystemExit(f"no cached clips in {args.cache}")
    probe = CachedClip.from_dict(torch.load(clips[0], weights_only=False, mmap=True))
    feat_dim, has_feats = probe.feat_dim, probe.context is not None
    points = args.points or targs["points"]
    print(f"{len(clips)} test clips | {device} | amp {'bf16' if amp else 'off'}"
          f" | points {points} | {args.sample_steps} sampling steps", flush=True)

    model = PointDiT(dim=targs["dim"], depth=targs["depth"],
                     num_heads=targs["heads"], cross_attn=has_feats,
                     feat_dim=feat_dim).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded step {ckpt['step']} from {args.ckpt}", flush=True)

    loader = DataLoader(
        # resample=False: a test score must not change between runs.
        ClipDataset(clips, points, resample=False,
                    norm_mode=targs.get("norm_mode", "median")),
        batch_size=args.batch, shuffle=False, num_workers=4,
    )

    t0 = time.time()
    m = evaluate(model, loader, device, steps=args.sample_steps, amp=amp)
    print(f"\nscored in {(time.time() - t0) / 60:.1f} min\n", flush=True)

    for k in ("average_pts_within_thresh", "apd_baseline", "val_loss"):
        print(f"  {k:<26} {m[k]:.4f}")
    print()
    for t in (1, 2, 4, 8, 16):
        print(f"  pts_within_{t:<15} {m[f'pts_within_{t}']:.4f}")

    if m["average_pts_within_thresh"] <= m["apd_baseline"]:
        print("\n  APD is at or below the mean-trajectory baseline -- nothing"
              " generalised to unseen clips.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"ckpt": args.ckpt, "cache": args.cache, "step": ckpt["step"],
             "sample_steps": args.sample_steps, "clips": len(clips), **m},
            indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
