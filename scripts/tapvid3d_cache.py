"""TAPVid-3D clips as clips the model reads, one per query frame and direction.

A TAPVid-3D query starts at any frame; the model tracks from frame 0. So each
query frame q of a clip becomes two clips: frames q, q+1, ... (forward) and
q, q-1, ... (backward, the video reversed), each holding the queries that start
at q. Their two halves make the query's whole track, which is how trackers
that start from one frame run on this benchmark. Only every --every-th query
frame is kept, for a quick check rather than the full benchmark.

What a real tracker gets and nothing more: the frames, the camera (fixed in
Panoptic Studio), and each query's pixel and frame. The depth is Depth
Anything 3's (`scripts/da3_depth.py`); each query starts at its pixel at that
depth, not at its true 3D point. The true tracks are kept only to be scored.

Written like the Kubric caches: `<clip>_q<q>_<f|b>.pt` in --dst (DINOv3 patch
tokens at 384 px, patch positions, the scale, the queries' DINOv3 cards), and
beside it in --frames-dst the frames at 384 px and the depth at each CNN cell.

  python scripts/tapvid3d_cache.py --src DIR --dst CACHE --frames-dst FRAMES [--every 8]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from cache_frames import cell_depth  # noqa: E402
from genpoint3d.CachedClip import CachedClip  # noqa: E402
from genpoint3d.ScaleStats import ScaleStats  # noqa: E402
from genpoint3d.prepare.VisualEncoder import VisualEncoder  # noqa: E402
from genpoint3d.prepare.tools.geometry import batch_unproject  # noqa: E402
from genpoint3d.tools.feature_lookup import sample_at  # noqa: E402

RES = 384

p = argparse.ArgumentParser()
p.add_argument("--src", required=True, help="TAPVid-3D .npz files with their .da3.npy depth")
p.add_argument("--dst", required=True)
p.add_argument("--frames-dst", required=True)
p.add_argument("--every", type=int, default=8, help="keep every n-th query frame of a clip")
p.add_argument("--depth-suffix", default=".da3.npy",
               help="the depth beside each clip: Depth Anything 3's, or the true one of a rendered clip")
args = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VisualEncoder(image_size=RES).to(dev).eval()
dst, fdst = Path(args.dst), Path(args.frames_dst)
dst.mkdir(parents=True, exist_ok=True)
fdst.mkdir(parents=True, exist_ok=True)

for src in sorted(Path(args.src).glob("*.npz")):
    d = np.load(src, allow_pickle=True)
    frames = torch.from_numpy(d["images"])                                    # (T, H, W, 3)
    depth = torch.from_numpy(np.load(src.with_suffix(args.depth_suffix)).astype(np.float32))
    T, H, W = frames.shape[:3]
    fx, fy, cx, cy = d["fx_fy_cx_cy"]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    tracks = torch.from_numpy(d["tracks_XYZ"]).float()                        # (T, N, 3) metres
    vis = torch.from_numpy(d["visibility"]).bool()
    q_xyt = torch.from_numpy(d["queries_xyt"]).float()
    q_t = q_xyt[:, 2].round().long()
    eye = torch.eye(4).expand(T, 4, 4)
    pointmap = batch_unproject(depth, K.expand(T, 3, 3), eye)                 # (T, 3, H, W)
    with torch.no_grad():
        feat = torch.cat([encoder(frames[i:i + 16].to(dev)).cpu() for i in range(0, T, 16)])
    small = F.interpolate(frames.permute(0, 3, 1, 2).float(), size=(RES, RES), mode="area")
    small = small.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    cells = cell_depth(depth.numpy(), RES)
    starts = sorted(q_t.unique().tolist())[:: args.every]
    for q in starts:
        idx = (q_t == q).nonzero()[:, 0]
        uv = q_xyt[idx, :2]
        # The query's start: its pixel at the estimated depth.
        z = depth[q, uv[:, 1].long().clamp(0, H - 1), uv[:, 0].long().clamp(0, W - 1)]
        anchor = torch.stack([(uv[:, 0] - cx) / fx * z, (uv[:, 1] - cy) / fy * z, z], -1)
        for name, order in (("f", list(range(q, T))), ("b", list(range(q, -1, -1)))):
            if len(order) < 2:
                continue
            o = torch.tensor(order)
            seq = f"{src.stem}_q{q:03d}_{name}"
            clip = CachedClip(
                seq_id=seq, traj=tracks[o][:, idx], anchor=anchor.clone(), visibility=vis[o][:, idx],
                query_uv=uv.clone(), hw=(H, W), intrinsics=K.expand(len(order), 3, 3).clone(),
                extrinsics=torch.eye(4).expand(len(order), 4, 4).clone(),
                stats=ScaleStats.measure(pointmap[o]),
                context=VisualEncoder.tokens(feat[o]).half(),
                id_card=sample_at(feat[q:q + 1], uv[None], (H, W))[0].half(),
                patch_xyz=VisualEncoder.patch_centre_xyz(pointmap[o], encoder.grid).float(),
                points=len(idx), feat_dim=encoder.dim, image_size=RES)
            torch.save(clip.to_dict(), dst / f"{seq}.pt")
            np.save(fdst / f"{seq}.npy", small[order])
            np.save(fdst / f"{seq}.depth.npy", cells[order])
    print(f"  {src.stem}: {len(starts)} query frames, {int(sum((q_t == q).sum() for q in starts))}"
          f" of {len(q_t)} queries", flush=True)
