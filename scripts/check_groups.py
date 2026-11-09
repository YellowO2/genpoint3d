"""How well a --group-feedback run's groups match the truth, by
how far points move.

For every pair of points, the run's affinity 2 sigmoid(-|v_i - v_j|^2) (above
0.5 counts as "same piece") against the truth at two tolerances: the pairs the
training labels call rigid (distance never off by more than 2 cm) and those
that are rigid to 5 mm. For each point, of the points the run puts with it,
the share that truly move with it at each tolerance, and how far apart in
motion they are: the gap between their displacements, in metric pixels at the
point's depth. Split by how far the point moves. A slow object grouped with the
still floor shows as a high share at 2 cm, a lower one at 5 mm, and a gap
close to the point's own motion. And the other way round: of the points that
truly move with it to 2 cm, the share the run puts with it ("found"), which a
run grouping each point with almost nothing would fail. In clips that say
which object each point is on, the same against the objects, and of the wrong
mates the share on the background.

  python scripts/check_groups.py --ckpt outputs/<a --group-feedback run>/best.pt --cache <val dir> \\
      --frames-cache <frames> [--points 500]
"""

import argparse

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.score.Run import Run
from genpoint3d.train.tools.batch import fine_inputs, to_device, to_metres, visual_mask
from genpoint3d.train.tools.metrics import _to_camera_t, motion_px
from genpoint3d.train.tools.regress import Sweeps

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--frames-cache", required=True)
p.add_argument("--points", type=int, default=500)
args = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run = Run.load(args.ckpt, dev)
model, rs = run.model, run.settings
ds = run.dataset(args.cache, args.frames_cache, points=args.points)
clips = ds.clips
edges = (0, 1, 4, 8, 16, 32, 64, float("inf"))
rows = {e: [[], [], [], [], [], [], [], []] for e in edges[:-1]}
with torch.no_grad():
    for batch in DataLoader(ds, batch_size=2, num_workers=2):
        b = to_device(batch, dev)
        cond = dict(context=b["context"], visual_mask=visual_mask(b, None), id_card=b["id_card"],
                    patch_xyz=b["patch_xyz"], **fine_inputs(model, b))
        sweeps = Sweeps(model, b["anchor"], b["traj"].shape[1], rs.refine_iters, rs.schedule, rs.window,
                        rs.overlap, cond)
        for _ in sweeps:
            pass
        emb = sweeps.groups.float()  # each point's last
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
        moved = motion_px(gt, b["visibility"].bool(), b["intrinsics_256"], b["extrinsics"], b["query_t"])
        off = ~torch.eye(same.shape[1], dtype=torch.bool, device=dev)
        sel = same & off
        n = sel.sum(-1).clamp_min(1)
        # Against the objects, in clips that say which (`segment`): right, found,
        # and of the wrong mates, the share on the background (floor, backdrop).
        seg = b["segment"]
        labelled = (seg >= 0).all(-1)[:, None]
        obj = (seg[:, :, None] == seg[:, None]) & off
        wrong = sel & ~obj
        for lo, hi in zip(edges[:-1], edges[1:]):
            mates = rigid2 & off
            m = (moved >= lo) & (moved < hi) & (sel.sum(-1) > 0) & (mates.sum(-1) > 0)
            if m.any():
                rows[lo][0].append(((rigid2 & sel).sum(-1) / n)[m].cpu())
                rows[lo][1].append(((rigid05 & sel).sum(-1) / n)[m].cpu())
                rows[lo][2].append(((gap * sel).sum(-1) / n)[m].cpu())
                rows[lo][3].append(sel.sum(-1)[m].float().cpu())
                rows[lo][4].append(((mates & sel).sum(-1) / mates.sum(-1).clamp_min(1))[m].cpu())
                k = m & labelled
                if k.any():
                    rows[lo][5].append(((obj & sel).sum(-1) / n)[k].cpu())
                    rows[lo][6].append(((obj & sel).sum(-1) / obj.sum(-1).clamp_min(1))[k].cpu())
                    rows[lo][7].append(((wrong & (seg[:, None, :] == 0)).sum(-1)
                                        / wrong.sum(-1).clamp_min(1))[k & (wrong.sum(-1) > 0)].cpu())

print(f"{len(clips)} clips | {args.ckpt} | a pair counts as one piece above affinity 0.5")
print(f"  {'moved (px)':>11} {'points':>7} {'piece size':>10} {'rigid 2 cm':>11} {'rigid 5 mm':>11}"
      f" {'motion gap px':>14} {'found':>6} {'object right':>13} {'object found':>13} {'wrong on bg':>12}")
for lo, hi in zip(edges[:-1], edges[1:]):
    r = rows[lo]
    if r[0]:
        c = [torch.cat(x) if x else torch.zeros(0) for x in r]
        o = [f"{x.mean():>13.3f}" if len(x) else f"{'-':>13}" for x in c[5:7]] + \
            [f"{c[7].mean():>12.3f}" if len(c[7]) else f"{'-':>12}"]
        print(f"  {f'{lo:g}-{hi:g}':>11} {len(c[0]):>7} {c[3].median():>10.0f} {c[0].mean():>11.3f}"
              f" {c[1].mean():>11.3f} {c[2].median():>14.2f} {c[4].mean():>6.3f} " + " ".join(o))
