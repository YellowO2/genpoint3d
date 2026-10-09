"""
Run a published 4D reconstruction model, 4RC, on OUR validation clips and score
its 3D tracks with OUR scorer.

Unlike TAPIP3D, 4RC is given the RGB video and nothing else: no depth, no
cameras, no 3D query points. It predicts its own depth and cameras, and for one
query frame a dense map: for every pixel of that frame, where its 3D point is
at each time. So:

  - the query frame is frame 0, and each of our queries reads that map at its
    pixel (nearest pixel, as 4RC's own eval/track/track_eval.py does);
  - the map lives in 4RC's world frame. Our tracks are stored in the frame-0
    camera, so every track is moved there with 4RC's OWN predicted frame-0
    camera;
  - its scale is not metric, so only the protocol's global median rescale is
    meaningful. There is no "no rescale" row here.

Three predictions are scored: `static` (the true start point, held), `4RC@t0`
(4RC's own frame-0 point, held -- its geometry with no motion at all) and `4RC`.

Runs inside 4RC's own environment, with their repo on the path; only our cache
format, raw loader and metric are imported from this repo.

Run (see scripts/ref_4rc.pbs):
  python scripts/ref_4rc.py --ref-repo ~/scratch/refs/4rc \\
      --clips ~/scratch/ref_eval/val100 --raw ~/scratch/kubric
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

OURS = Path(__file__).resolve().parent.parent
PATCH = 14


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-repo", required=True, help="clone of Luo-Yihang/4RC")
    p.add_argument("--ckpt", default="checkpoints/4RC",
                   help="directory holding model.safetensors, or a Hugging Face repo id")
    p.add_argument("--clips", required=True, help="directory of cached clips to score")
    p.add_argument("--raw", required=True, help="raw Kubric root (frames)")
    p.add_argument("--out", default=None, help="save predictions here, one .pt per clip")
    p.add_argument("--size", type=int, default=512,
                   help="long side fed to the model, rounded down to a multiple of 14")
    p.add_argument("--min-motion-px", type=float, default=4.0)
    args = p.parse_args()

    ref = Path(args.ref_repo).expanduser().resolve()
    os.chdir(ref)
    sys.path.insert(0, str(ref))
    sys.path.append(str(OURS))

    from arc.dust3r.inference_multiview import inference
    from arc.dust3r.utils.image import ImgNorm
    from arc.models.arc import Arc

    from genpoint3d.data.kubric import KubricSequenceDataset
    from genpoint3d.eval.metrics import THRESHOLDS, motion_px, tapvid3d_metrics

    dev = torch.device("cuda")
    model = Arc.from_pretrained(args.ckpt).to(dev).eval()
    print(f"4RC {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
          f" | long side {args.size} | bf16-mixed", flush=True)

    clips = sorted(Path(args.clips).expanduser().glob("*.pt"))
    out = Path(args.out).expanduser() if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    keys = [f"pts_within_{t}" for t in THRESHOLDS] + ["average_pts_within_thresh"]
    names = ("static", "4RC@t0", "4RC")
    labels = ("all points, protocol rescale", "moving points, protocol rescale")
    sums, moving_frac, secs, cam0_dev, scored = {}, 0.0, [], 0.0, 0

    for i, path in enumerate(clips, 1):
        c = torch.load(path, weights_only=False)
        seq = c["seq_id"]
        raw = KubricSequenceDataset(str(Path(args.raw).expanduser()), seq_ids=[seq],
                                    num_query_points=None)[0]
        H0, W0 = raw.frames.shape[1:3]

        # Their eval's preparation: a plain resize to a multiple of the patch size.
        s = args.size / max(H0, W0)
        h, w = (round(H0 * s) // PATCH) * PATCH, (round(W0 * s) // PATCH) * PATCH
        views = []
        for f in raw.frames:
            img = Image.fromarray(f).resize((w, h), Image.BICUBIC)
            views.append(dict(img=ImgNorm(img)[None], true_shape=np.int32([[h, w]]),
                              idx=len(views), instance=str(len(views)),
                              track_query_idx=torch.tensor([0])))

        torch.cuda.synchronize()
        t0 = time.time()
        res = inference(views, model, dev, dtype="bf16-mixed", verbose=False)
        torch.cuda.synchronize()
        secs.append(time.time() - t0)

        preds = res["preds"]
        track = torch.cat([q["track"] for q in preds]).float()      # (T, h, w, 3), their world
        conf = torch.cat([q["conf_track"] for q in preds]).float()  # (T, h, w)
        c2w0 = preds[0]["extrinsic"].float().reshape(4, 4)          # frame-0 camera -> world
        cam0_dev = max(cam0_dev, (c2w0 - torch.eye(4)).abs().max().item())
        w2c0 = torch.linalg.inv(c2w0)

        uv = c["query_uv"].float() * torch.tensor([w / W0, h / H0])
        x = uv[:, 0].long().clamp(0, w - 1)
        y = uv[:, 1].long().clamp(0, h - 1)
        pred = track[:, y, x] @ w2c0[:3, :3].T + w2c0[:3, 3]         # (T, N, 3) frame-0 camera
        if out:
            torch.save({"seq_id": seq, "coords": pred, "conf": conf[:, y, x],
                        "cam0_to_world": c2w0, "hw": (h, w)}, out / path.name)
        if not torch.isfinite(pred).all():
            print(f"  {seq}: non-finite prediction, clip skipped", flush=True)
            continue
        scored += 1

        gt = c["traj_metric"].float().to(dev)                       # (T, N, 3) frame-0 camera, metres
        vis = c["visibility"].to(dev)
        E = c["extrinsics"].float().to(dev)                         # cam_0 -> cam_t
        K256 = c["intrinsics"].clone().float()
        K256[..., :2, :] *= 256 / min(c["hw"])
        K256 = K256.to(dev)[None]
        g, v, e = gt[None], vis[None], E[None]
        pr = pred.to(dev)[None]
        moving = motion_px(g, v, K256, e) > args.min_motion_px
        moving_frac += moving.float().mean().item()
        cands = {"static": g[:, :1].expand_as(g), "4RC@t0": pr[:, :1].expand_as(pr), "4RC": pr}
        for label, mask in zip(labels, (v, v & moving[:, None])):
            if not mask.any():
                continue
            for name, q in cands.items():
                m = tapvid3d_metrics(q, g, mask, K256, e, scaling="median")
                t = sums.setdefault((label, name), [[0.0] * len(keys), 0])
                t[0] = [a + m[k] for a, k in zip(t[0], keys)]
                t[1] += 1
        if i % 10 == 0 or i == len(clips):
            print(f"  {i}/{len(clips)} | {np.mean(secs[1:] or secs):.2f} s/clip", flush=True)

    print(f"\n{scored} clips scored of {len(clips)} | {moving_frac / max(scored, 1):.0%} of points move >"
          f" {args.min_motion_px:g}px | {np.mean(secs[1:] or secs):.2f} s/clip at {h}x{w}"
          f" | peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
          f" | predicted frame-0 camera differs from identity by at most {cam0_dev:.2g}\n")
    for label in labels:
        print(f"--- {label} ({sums[(label, '4RC')][1]} clips) ---")
        print(f"{'':10}" + "".join(f"{'<' + str(t) + 'px':>8}" for t in THRESHOLDS) + f"{'APD':>8}")
        for name in names:
            tot, n = sums[(label, name)]
            print(f"{name:10}" + "".join(f"{x / n:>8.3f}" for x in tot))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
