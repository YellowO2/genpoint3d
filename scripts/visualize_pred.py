"""
Overlay a model's predicted tracks on the ground truth, in 3D and reprojected
into the image.

APD says THAT a model is wrong; it cannot say HOW. "Frozen at frame 0",
"collapsed to one averaged path" and "offset by a bad normalisation" can all
produce the same scalar and need completely different fixes. This separates
them by eye:

    green   ground truth
    red     prediction

Pair it with `train.py --overfit N`. A model that has seen two clips thousands
of times should reproduce them exactly; anything else visible here is a bug in
the pipeline rather than a shortage of data.

Run:  python scripts/visualize_pred.py --ckpt local/outputs/overfit2/best.pt \
                                       --cache local/cache/overfit2 \
                                       --root local/data/kubric_test
Then drag the .rrd into https://rerun.io/viewer (or `rerun out.rrd`).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

import rerun
import rerun.blueprint as rrb

from genpoint3d.eval.metrics import tapvid3d_metrics
from genpoint3d.models.model import PointDiT
from genpoint3d.models.flow import sample as flow_sample
from genpoint3d.data.cache import CachedClip
from genpoint3d.viz import rerun_log as rl
from train import ClipDataset, known_frame0, split, to_device, to_metres

GT, PRED = (0.2, 0.9, 0.3), (0.95, 0.25, 0.2)


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True, help="cache DIRECTORY")
    p.add_argument("--root", default=None,
                   help="raw Kubric dir, for the RGB and point cloud backdrop."
                        " The cache does not keep frames, so without this you"
                        " get trajectories on an empty canvas")
    p.add_argument("--clip", default=None, help="seq_id; default is the first")
    p.add_argument("--points", type=int, default=None)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--out", default="local/outputs/pred.rrd")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    # An --overfit checkpoint memorised (train + val)[:N] of a SEEDED
    # PERMUTATION, which is not the first N files on disk. Picking by filename
    # would draw clips the model never saw and call the result memorisation.
    if targs.get("overfit"):
        tr, va, _, _ = split(args.cache, targs["val_frac"], targs["seed"])
        clips = (tr + va)[: targs["overfit"]]
        print(f"overfit checkpoint: showing one of the {len(clips)} clips it"
              " was trained on", flush=True)
    else:
        clips = sorted(Path(args.cache).glob("*.pt"))
    if args.clip:
        clips = [c for c in clips if Path(c).stem == args.clip] or clips[:1]
    clips = clips[:1]
    probe = CachedClip.from_dict(torch.load(clips[0], weights_only=False, mmap=True))
    seq_id = probe.seq_id
    print(f"clip {seq_id} | step {ckpt['step']} | {device}"
          f" | target {targs.get('target', 'absolute')}", flush=True)

    model = PointDiT(dim=targs["dim"], depth=targs["depth"], num_heads=targs["heads"],
                     cross_attn=probe.context is not None,
                     feat_dim=probe.feat_dim,
                     locality=bool(targs.get("locality", 0))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # resample=False and the training normalisation, so what is drawn is what
    # was scored -- a different point subset would not be comparable.
    loader = DataLoader(
        ClipDataset(clips, args.points or targs["points"], resample=False,
                    norm_mode=targs.get("norm_mode", "median"),
                    target=targs.get("target", "absolute")),
        batch_size=1, shuffle=False,
    )
    b = to_device(next(iter(loader)), device)
    traj, anchor, vis, ctx, idc, pxyz = (b["traj"], b["anchor"], b["visibility"],
                                         b["context"], b["id_card"], b["patch_xyz"])
    vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=device) if ctx is not None else None
    kx0 = known_frame0(b) if targs.get("anchor_frame0", 0) else None

    pred = flow_sample(model, anchor, num_frames=traj.shape[1],
                       steps=args.sample_steps, known_x0=kx0,
                       context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)

    pred_m, gt_m = to_metres(pred.float(), b), to_metres(traj, b)
    score = lambda x: tapvid3d_metrics(x, gt_m, vis, b["intrinsics"], b["extrinsics"])
    apd = score(pred_m)["average_pts_within_thresh"]
    static = score(gt_m[:, :1].expand_as(gt_m))["average_pts_within_thresh"]
    # Distances in metres say the same thing as APD but without the threshold,
    # and separate "wrong everywhere" from "wrong on the few points that move".
    moved = (gt_m - gt_m[:, :1]).norm(dim=-1)
    err = (pred_m - gt_m).norm(dim=-1)
    print(f"  APD {apd:.4f} | static baseline {static:.4f}")
    print(f"  mean error {err[vis].mean():.4f} m"
          f" | mean GT motion from frame 0 {moved[vis].mean():.4f} m")
    print(f"  mean |pred - frame0| {(pred_m - gt_m[:, :1]).norm(dim=-1)[vis].mean():.4f} m"
          f"   <- near 0 means the model just predicts 'no motion'")

    rl.setup_visualizer(serve=False)
    I = b["intrinsics"][0].cpu().numpy().astype(np.float64)
    E = b["extrinsics"][0].cpu().numpy().astype(np.float64)

    if args.root:
        from genpoint3d.data.kubric import KubricSequenceDataset
        s = KubricSequenceDataset(args.root, seq_ids=[seq_id], num_query_points=16)[0]
        rl.log_video("clip", s.frames, I, E, s.depths)

    v = vis[0].cpu().numpy()
    rl.log_trajectory("clip", "gt", I, E, gt_m[0].cpu().numpy().astype(np.float64), v, color=GT)
    rl.log_trajectory("clip", "pred", I, E, pred_m[0].cpu().numpy().astype(np.float64), v, color=PRED)

    rerun.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/clip", name=f"{seq_id}  green=GT  red=pred"),
            rrb.Spatial2DView(origin="/clip/world/camera/image", name="reprojected"),
        ),
        collapse_panels=True,
    ))
    rl.save_recording(Path(args.out))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
