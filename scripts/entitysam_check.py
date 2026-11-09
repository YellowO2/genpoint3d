"""How well EntitySAM's video masks (CVPR 2025, prompt-free, one id per object
through the clip) say which object each of our points is on, and how long a
clip takes. Raw clips that save each point's object (scripts/kubric_movi.pbs):
each point read at its first visible frame and pixel; two points with one
mask id are put together. Of the points put with a point, the share truly on
its object (right), and of those on it, the share put with it (found), by how
far the point moves in the image, as scripts/check_groups.py reports the
model's own groups.

Runs in EntitySAM's environment (scripts/entitysam_check.pbs):
  python scripts/entitysam_check.py --raw DIR --entitysam ~/scratch/entitysam [--clips 10]
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--raw", required=True, help="raw clips: <id>/<id>.npy with 'segment', <id>/frames/*.png")
p.add_argument("--entitysam", required=True, help="the EntitySAM checkout")
p.add_argument("--ckpt", default="checkpoints/vit-l/model_0009999.pth")
p.add_argument("--cfg", default="configs/sam2.1_hiera_l.yaml")
p.add_argument("--decoder-depth", type=int, default=8)
p.add_argument("--clips", type=int, default=10)
args = p.parse_args()

root = Path(args.entitysam).expanduser()
sys.path.insert(0, str(root))
from sam2.build_sam import build_sam2_video_query_iou_predictor  # noqa: E402
from eval.eval_vipseg_entity_seg import inference_video_vps_save_results  # noqa: E402

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
predictor = build_sam2_video_query_iou_predictor(args.cfg, str(root / args.ckpt), mask_decoder_depth=args.decoder_depth)

edges = (0, 1, 4, 8, 16, 32, 64, float("inf"))
rows = {e: ([], []) for e in edges[:-1]}
unlabelled, seconds = [], []
clips = sorted(d for d in Path(args.raw).iterdir() if (d / f"{d.name}.npy").exists())[: args.clips]
for clip in clips:
    d = np.load(clip / f"{clip.name}.npy", allow_pickle=True).item()
    uv, occ, seg = d["coords"], d["occluded"], d["segment"]          # (N, T, 2), (N, T), (N,)
    N, T = occ.shape
    with tempfile.TemporaryDirectory() as tmp:
        for t in range(T):
            Image.open(clip / "frames" / f"{t:03d}.png").convert("RGB").save(f"{tmp}/{t:05d}.jpg", quality=95)
        H, W = np.asarray(Image.open(f"{tmp}/00000.jpg")).shape[:2]
        torch.cuda.synchronize()
        t0 = time.time()
        # As EntitySAM's own evaluation: three warm-up passes over the first frame.
        state = predictor.init_state(video_path=tmp, progress_order=[0, 0, 0] + list(range(T)))
        predictor.reset_state(state)
        masks, ious = [], []
        for _, _, logits, eiou in predictor.propagate_in_video(state, start_frame_idx=0):
            masks.append(logits)
            ious.append(eiou)
        masks = torch.cat(masks, dim=1)[:, 3:]
        out = inference_video_vps_save_results(torch.stack(ious).mean(0), None, masks, (H, W))
        pan = out["pred_masks"].numpy()                                # (T, H, W) one id an object, 0 none
        torch.cuda.synchronize()
        seconds.append(time.time() - t0)

    vis = ~occ
    q = vis.argmax(1)                                                  # first frame seen
    n = np.arange(N)
    x = np.clip(np.floor(uv[n, q, 0]).astype(int), 0, W - 1)
    y = np.clip(np.floor(uv[n, q, 1]).astype(int), 0, H - 1)
    pid = pan[q, y, x]
    moved = np.where(vis, np.linalg.norm(uv - uv[n, q][:, None], axis=-1), 0).max(1)
    off = ~np.eye(N, dtype=bool)
    said = (pid[:, None] == pid[None]) & (pid[:, None] > 0) & off
    true = (seg[:, None] == seg[None]) & off
    unlabelled.append((pid == 0).mean())
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (moved >= lo) & (moved < hi) & said.any(1) & true.any(1) & vis.any(1)
        rows[lo][0].extend(((said & true).sum(1) / said.sum(1).clip(1))[m])
        rows[lo][1].extend(((said & true).sum(1) / true.sum(1).clip(1))[m])
    print(f"  {clip.name}: {T} frames {H}x{W}, {len(np.unique(pan)) - 1} masks, {seconds[-1]:.1f} s", flush=True)

print(f"\n{len(clips)} clips | EntitySAM {args.ckpt} | {np.mean(seconds):.1f} s a clip "
      f"({np.sum(seconds) / sum(1 for _ in clips) / 48 * 1000:.0f} ms a frame) | points on no mask {np.mean(unlabelled):.3f}")
print(f"  {'moved (px)':>11} {'points':>7} {'right':>7} {'found':>7}")
for lo, hi in zip(edges[:-1], edges[1:]):
    r, f = rows[lo]
    if r:
        print(f"  {f'{lo:g}-{hi:g}':>11} {len(r):>7} {np.mean(r):>7.3f} {np.mean(f):>7.3f}")
