"""How well a --rigid-fit run's groups match the truth, by how far points move.

For every pair of points, the run's affinity 2 sigmoid(-|v_i - v_j|^2) (above
0.5 counts as "same piece") against the truth at two tolerances: the pairs the
training labels call rigid (distance never off by more than 2 cm) and those
that are rigid to 5 mm. For each point, of the points the run puts with it,
the share that truly move with it at each tolerance, and how far apart in
motion they are: the gap between their displacements, in metric pixels at the
point's depth. Split by how far the point moves. A slow object grouped with the
still floor shows as a high share at 2 cm, a lower one at 5 mm, and a gap
close to the point's own motion.

  python scripts/check_groups.py --ckpt outputs/fyV_fit_rounds/best.pt --cache <val dir> \\
      --frames-cache <frames> [--points 500]
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.tools.batch import fine_inputs, known_frame0, to_device, to_metres
from genpoint3d.train.tools.metrics import _to_camera_t, motion_px
from genpoint3d.train.tools.regress import refine

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--frames-cache", required=True)
p.add_argument("--points", type=int, default=500)
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
targs = ckpt["args"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clips = sorted(Path(args.cache).glob("*.pt"))
ds = ClipDataset(clips, args.points, resample=False, norm_mode=targs.get("norm_mode", "median"),
                 target=targs.get("target", "displacement"), fix_idcard=bool(targs.get("fix_idcard", 0)),
                 frames=Path(args.frames_cache), features=bool(targs.get("dino", 1)), depth=True)
first = torch.load(clips[0], weights_only=False, mmap=True)
model = build_model(Settings(**targs), first["feat_dim"], first["context"] is not None).to(dev).eval()
model.load_state_dict(ckpt["model"])

edges = (0, 1, 4, 8, 16, 32, 64, float("inf"))
rows = {e: [[], [], [], []] for e in edges[:-1]}
with torch.no_grad():
    for batch in DataLoader(ds, batch_size=2, num_workers=2):
        b = to_device(batch, dev)
        vm = torch.ones(b["traj"].shape[:2], dtype=torch.bool, device=dev)
        cond = dict(context=b["context"], visual_mask=vm, id_card=b["id_card"], patch_xyz=b["patch_xyz"],
                    **fine_inputs(model, b))
        refine(model, b["anchor"], num_frames=b["traj"].shape[1], iters=targs.get("refine_iters", 4),
               known_x0=known_frame0(b) if targs.get("anchor_frame0") else None, param=targs.get("param", "position"),
               window=targs.get("window", 0), overlap=targs.get("overlap", 0), **cond)
        emb = model._fitted[1].float()                                       # the last window's
        a = 2 * torch.sigmoid(-(emb[:, :, None] - emb[:, None]).pow(2).sum(-1))
        same = a > 0.5
        gt = to_metres(b["traj"].float(), b)
        d = (gt[:, :, :, None] - gt[:, :, None]).norm(dim=-1)
        drift = (d - d[:, :1]).abs().amax(dim=1)
        rigid2, rigid05 = drift < 0.02, drift < 0.005
        cam = _to_camera_t(gt, b["extrinsics"])
        focal = (b["intrinsics_256"][..., 0, 0] * b["intrinsics_256"][..., 1, 1]).sqrt()
        mult = (cam[..., 2].abs() / focal[..., None]).clamp_min(1e-6)                # (B, T, N) m per px
        disp = gt - gt[:, :1]
        gap = ((disp[:, :, :, None] - disp[:, :, None]).norm(dim=-1) / mult[..., None]).amax(dim=1)   # px
        moved = motion_px(gt, b["visibility"].bool(), b["intrinsics_256"], b["extrinsics"])
        off = ~torch.eye(same.shape[1], dtype=torch.bool, device=dev)
        sel = same & off
        n = sel.sum(-1).clamp_min(1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (moved >= lo) & (moved < hi) & (sel.sum(-1) > 0)
            if m.any():
                rows[lo][0].append(((rigid2 & sel).sum(-1) / n)[m].cpu())
                rows[lo][1].append(((rigid05 & sel).sum(-1) / n)[m].cpu())
                rows[lo][2].append(((gap * sel).sum(-1) / n)[m].cpu())
                rows[lo][3].append(sel.sum(-1)[m].float().cpu())

print(f"{len(clips)} clips | {args.ckpt} | a pair counts as one piece above affinity 0.5")
print(f"  {'moved (px)':>11} {'points':>7} {'piece size':>10} {'rigid 2 cm':>11} {'rigid 5 mm':>11} {'motion gap px':>14}")
for lo, hi in zip(edges[:-1], edges[1:]):
    r = rows[lo]
    if r[0]:
        c = [torch.cat(x) for x in r]
        print(f"  {f'{lo:g}-{hi:g}':>11} {len(c[0]):>7} {c[3].median():>10.0f} {c[0].mean():>11.3f}"
              f" {c[1].mean():>11.3f} {c[2].median():>14.2f}")
