"""MolmoMotion as a forecasting reference: its inputs from our clips, and the scores.

MolmoMotion (allenai, 2026) forecasts the 3D tracks of 8 points from 3 RGB
frames, the points' 3D history and a caption. This puts our val clips in that
form, with our model's own forecast beside them, and scores both the same way.

  export   per clip, 8 points on one moving object (`rigid_buddies`), visible
           in the history frames: the frames, the points at the last seen
           frame in pixels, their true 3D history in that frame's camera, the
           truth after it, and our checkpoint's forecast with the frames from
           --cutoff on blank. One .npz per clip.
  score    reads MolmoMotion's predictions (scripts/molmo_predict.py) and
           scores it, our forecast, "stay where it was last seen" and "keep
           its last speed" with the APD on the blank frames, these 8 points.

MolmoMotion runs at 15 fps and Kubric at --fps; its history frames are the
clip's frames nearest 1/15 s apart and its forecast is resampled in time.

  python scripts/molmo_export.py export --ckpt outputs/fyO_joint_long/best.pt \\
      --cache <cache> --frames-cache <frames> --clips 50 --out molmo/in
  python scripts/molmo_export.py score --in molmo/in --pred molmo/pred
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

MOLMO_FPS = 15.0
P = 8


def export(args) -> int:
    from torch.utils.data import DataLoader

    from genpoint3d.train.Settings import Settings
    from genpoint3d.train.data.ClipDataset import ClipDataset
    from genpoint3d.train.data.split_clips import split_clips
    from genpoint3d.train.model.build_model import build_model
    from genpoint3d.train.score.Predictor import Predictor
    from genpoint3d.train.score.ScoreSettings import ScoreSettings
    from genpoint3d.train.tools.batch import rigid_buddies, to_metres
    from genpoint3d.train.tools.metrics import _to_camera_t, motion_px

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = split_clips(args.cache, targs.get("val_frac", 0.1), targs.get("seed", 0))
    clips = split.val[: args.clips]
    fine_xyz = bool(targs.get("fine_xyz", 0))
    ds = ClipDataset(clips, targs["points"], resample=False,
                     norm_mode=targs.get("norm_mode", "median"),
                     target=targs.get("target", "absolute"),
                     fix_idcard=bool(targs.get("fix_idcard", 0)),
                     frames=ClipDataset.frames_dir(args.frames_cache, clips,
                                                   targs.get("fine_res", 384), depth=fine_xyz),
                     features=bool(targs.get("dino", 1)), depth=fine_xyz)
    loader = DataLoader(ds, batch_size=targs.get("batch", 1), shuffle=False, num_workers=2)
    model = build_model(Settings(**targs), split.feat_dim, split.has_feats).to(device)
    model.load_state_dict(ckpt["model"])
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    predictor = Predictor(model, ScoreSettings(
        device=device, amp=amp, anchor_frame0=bool(targs.get("anchor_frame0", 0)),
        loss_type=targs.get("loss_type", "l21"), method=targs.get("method", "flow"),
        refine_iters=targs.get("refine_iters", 4), param=targs.get("param", "position"),
        carry=bool(targs.get("carry", 0)), window=targs.get("window", 0),
        overlap=targs.get("overlap", 0), cutoff=args.cutoff))

    last = args.cutoff - 1
    step = args.fps / MOLMO_FPS
    hist = [round(last - k * step) for k in (2, 1, 0)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k, kept = 0, 0
    for batch in loader:
        with torch.no_grad():
            pred = predictor(batch)
        b = pred.batch
        gt = to_metres(b["traj"], b).float()
        ours = to_metres(pred.tracks.float(), b)
        vis = b["visibility"].bool()
        moving = motion_px(gt, vis, b["intrinsics_256"], b["extrinsics"]) > 4
        buddies = rigid_buddies(gt)
        for i in range(gt.shape[0]):
            name = clips[k].stem
            k += 1
            seen = vis[i, hist].all(0) & moving[i] & vis[i, args.cutoff:].any(0)
            if seen.sum() < P:
                continue
            # The most moving such point, then the others on its object.
            first = int(torch.where(seen, (gt[i] - gt[i, :1]).norm(dim=-1).amax(0), -1.0).argmax())
            group = torch.where(seen & buddies[i, first])[0]
            if len(group) < P - 1:
                continue
            pick = torch.cat([torch.tensor([first], device=group.device),
                              group[torch.randperm(len(group), generator=torch.Generator().manual_seed(0))[: P - 1].to(group.device)]])
            E = b["extrinsics"][i]                                      # (T, 4, 4) frame 0 -> frame t
            to_last = E[last:last + 1].expand(gt.shape[1], -1, -1)
            cam = _to_camera_t(gt[i:i + 1, :, pick], to_last[None])[0]  # (T, P, 3) in the last seen camera
            h, w = b["hw"][i].tolist()
            K = b["intrinsics"][i].clone().float()
            K = K[0] if K.dim() == 3 else K
            frames = np.load(Path(args.frames_cache) / f"{name}.npy")
            S = frames.shape[1]
            K384 = K.clone()
            K384[0] *= S / w
            K384[1] *= S / h
            xyz = cam[last]
            uv = torch.stack([K384[0, 0] * xyz[:, 0] / xyz[:, 2] + K384[0, 2],
                              K384[1, 1] * xyz[:, 1] / xyz[:, 2] + K384[1, 2]], -1)
            np.savez_compressed(
                out / f"{name}.npz",
                frames=frames[hist], points_2d=uv.cpu().numpy(), history_3d=cam[hist].cpu().numpy(),
                K=K384.cpu().numpy(), gt=gt[i][:, pick].cpu().numpy(), ours=ours[i][:, pick].cpu().numpy(),
                visible=vis[i][:, pick].cpu().numpy(), extrinsics=E.cpu().numpy(),
                intrinsics_256=b["intrinsics_256"][i].cpu().numpy(), cutoff=args.cutoff, fps=args.fps)
            kept += 1
        if k >= len(clips):
            break
    predictor.done()
    print(f"{kept} of {len(clips)} clips have 8 points on one moving object -> {out}", flush=True)
    return 0


def score(args) -> int:
    from genpoint3d.train.tools.metrics import tapvid3d_metrics

    rows = {"MolmoMotion": [], "ours": [], "stay where last seen": [], "keep last speed": []}
    failed = 0
    for f in sorted(Path(args.inp).glob("*.npz")):
        p = Path(args.pred) / f.name
        if not p.exists():
            continue
        d, m = np.load(f), np.load(p)
        if not bool(m["parsed"]):
            failed += 1
            continue
        c, fps = int(d["cutoff"]), float(d["fps"])
        T = d["gt"].shape[0]
        E = torch.tensor(d["extrinsics"])
        R, t = E[c - 1, :3, :3], E[c - 1, :3, 3]
        # MolmoMotion's frame j is j / 15 s after the last seen frame; ours j / fps.
        fut = torch.tensor(m["future_3d"]).float()                     # (P, F, 3), last seen camera
        start = torch.tensor(d["history_3d"][-1]).float()[:, None]
        track = torch.cat([start, fut], 1)                             # frame 0 = last seen
        at = torch.arange(1, T - c + 1).float() * MOLMO_FPS / fps
        lo = at.floor().long().clamp(max=track.shape[1] - 1)
        hi = (lo + 1).clamp(max=track.shape[1] - 1)
        a = (at - lo.float())[None, :, None]
        cam = track[:, lo] * (1 - a) + track[:, hi] * a                # (P, T - c, 3)
        molmo = torch.einsum("ji,ptj->tpi", R, cam - t)                # back to frame-0 camera
        gt = torch.tensor(d["gt"]).float()
        future = slice(c, T)
        stay = gt[c - 1:c].expand(T - c, -1, -1)
        speed = gt[c - 1] - gt[c - 2]
        keep = gt[c - 1] + speed * torch.arange(1, T - c + 1).float()[:, None, None]
        vis = torch.tensor(d["visible"]).bool()[future][None]
        K = torch.tensor(d["intrinsics_256"]).float()
        K = K[None].expand(T, -1, -1) if K.dim() == 2 else K
        for name, pr in (("MolmoMotion", molmo), ("ours", torch.tensor(d["ours"]).float()[future]),
                         ("stay where last seen", stay), ("keep last speed", keep)):
            s = tapvid3d_metrics(pr[None], gt[future][None], vis, K[future][None], E[future][None],
                                 scaling="none")["average_pts_within_thresh"]
            rows[name].append(float(s))
    n = len(rows["ours"])
    print(f"\n{n} clips ({failed} more where MolmoMotion gave no readable tracks), "
          "8 points on one moving object each, the frames after the cut, APD:")
    for name, v in rows.items():
        print(f"  {name:22s} {np.mean(v):.3f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--cache", required=True)
    e.add_argument("--frames-cache", required=True)
    e.add_argument("--clips", type=int, default=50)
    e.add_argument("--cutoff", type=int, default=12)
    e.add_argument("--fps", type=float, default=24.0)
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--pred", required=True)
    args = p.parse_args()
    return export(args) if args.mode == "export" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
