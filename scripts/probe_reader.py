"""Does the matcher alone know where the point is? Only a reader trains.

A trained run's matcher -- its CNN and TAPIP3D's 3D-neighbourhood comparison
-- stays frozen. A small reader learns from the matcher's output alone to say
how far the true position is from a guess: the guess is the truth in frame t
pushed off in 3D by up to --push pixels on each axis (at the point's own
depth, as the metric measures), the template is the query in frame 0, and the
answer is Kubric's own track. Visible moving points, frames 1 on.

  reader error small   the matcher's output holds the position, and a model
                       that lands off the point is not reading it
  reader error large   the matcher's output does not hold it

Scored on the run's own val clips the way the tracker uses it, four reads,
each from the last one's guess, in the metric's pixels, and split, at the
same distance moved, by how far the point turned since frame 0.

With --out, the matcher and its reader also track on their own: from the
query in frame 0, each frame starting where the last one ended, four reads a
frame, nothing else of the model. Drawn on the first --clips val clips, the
truth green and the matcher red.

  python scripts/probe_reader.py --ckpt outputs/fyO_joint_long/best.pt \\
      [--fine-weights weights/tapip3d_finder.pt] [--push 16] [--steps 3000] \\
      [--out renders/matcher --clips 6]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.Loaders import Loaders
from genpoint3d.train.data.apply_run_settings import apply_run_settings
from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.tools.batch import fine_inputs, to_device, to_metres
from genpoint3d.train.tools.camera import project
from render_tracks import colours, draw_frame, pixels, write_video
from genpoint3d.train.tools.metrics import (_to_camera_t, appearance_change, change_table, motion_px,
                                            motion_table)

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--fine-weights", default="", help="put these weights into the matcher instead")
p.add_argument("--push", type=float, default=16.0)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--out", default="", help="also track on its own and draw it here")
p.add_argument("--clips", type=int, default=6)
p.add_argument("--draw", type=int, default=32, help="points drawn, the most moving first")
args = p.parse_args()
FRAMES_PER_CLIP = 6
LOOKS = 4

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
settings = Settings(**{**ckpt["args"], "workers": 2, "batch": 8})
hardware = Hardware.pick(settings)
dev = hardware.device
clips = apply_run_settings(split_clips(settings.cache, settings.val_frac, settings.seed), settings, hardware)
model = build_model(settings, clips.feat_dim, clips.has_feats)
model.load_state_dict(ckpt["model"])
if args.fine_weights:
    missing, unexpected = model.fine.load_state_dict(torch.load(args.fine_weights, map_location="cpu"),
                                                     strict=False)
    if unexpected:
        raise ValueError(f"{args.fine_weights} does not fit this finder: {unexpected}")
model = model.to(dev).eval()
finder = model.fine
if not finder.neigh:
    raise ValueError("this probe reads the 3D-neighbourhood matcher (--fine-neigh)")


def points(batch, generator):
    """A few random frames per clip, with what the matcher needs there and
    what the scores need: the truth in metres and in the scene's own units
    (B, F, N, 3), which points count (B, F, N), each point's metric pixel in
    metres (B, F, N), how far it has moved in metric pixels and how far it
    has turned in degrees (B, F, N)."""
    b = to_device(batch, dev)
    with torch.no_grad():
        fine = fine_inputs(model, b)
        fmap, proj, xyz = fine["fine_map"].float(), fine["fine_proj"].float(), fine["fine_xyz"].float()
        B, T = fmap.shape[:2]
        pos = model._lookup_pos(b["traj"].float(), b["anchor"].float())       # (B, T, N, 3) scene
        gt = to_metres(b["traj"].float(), b)
        vis = b["visibility"].bool()
        cam = _to_camera_t(gt, b["extrinsics"])
        focal = (b["intrinsics_256"][..., 0, 0] * b["intrinsics_256"][..., 1, 1]).sqrt()
        mult = cam[..., 2].abs() / focal[..., None]                             # metres per px
        moved = (gt - gt[:, :1]).norm(dim=-1) / mult
        turned, _ = appearance_change(gt, b["extrinsics"])
        moving = motion_px(gt, vis, b["intrinsics_256"], b["extrinsics"]) > 4
        uv, _ = project(pos, proj)
        t = torch.stack([torch.randperm(T - 1, generator=generator)[:FRAMES_PER_CLIP] + 1
                         for _ in range(B)]).to(dev)                           # (B, F)
        at = lambda x: x.gather(1, t.view(B, -1, *[1] * (x.dim() - 2)).expand(-1, -1, *x.shape[2:]))
        keep = at(vis) & moving[:, None] & ((at(uv) > 0) & (at(uv) < 1)).all(-1)
        return dict(fmap=at(fmap), fmap0=fmap[:, 0], xyz=at(xyz), xyz0=xyz[:, 0], proj=at(proj),
                    uv0=project(b["anchor"].float()[:, None], proj[:, :1])[0][:, 0],
                    pos=at(pos), pos0=b["anchor"].float(), scale=b["norm_scale"].float(),
                    mult=at(mult), keep=keep, moved=at(moved), turned=at(turned))


def read(d, guess):
    """(B, F, N, out_dim): the matcher at `guess` (B, F, N, 3), scene units."""
    with torch.no_grad():
        uv, _ = project(guess, d["proj"])
        return finder(d["fmap"], d["fmap0"], uv, d["uv0"], xyz=d["xyz"], xyz0=d["xyz0"],
                      pos=guess, pos0=d["pos0"])


def pushed(d, generator):
    """The truth pushed off by up to --push metric pixels on each axis, and
    the push in metric pixels (B, F, N, 3)."""
    px = (torch.rand(d["pos"].shape, generator=generator).to(dev) * 2 - 1) * args.push
    return d["pos"] + px * (d["mult"] / d["scale"][:, None, None])[..., None], px


reader = nn.Sequential(nn.Linear(finder.out_dim, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(),
                       nn.Linear(256, 3)).to(dev)
loaders = Loaders(clips, settings)
opt = torch.optim.AdamW(reader.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=args.steps)
gen = torch.Generator().manual_seed(0)
print(f"frozen matcher from {args.fine_weights or args.ckpt} | push up to {args.push:g} px a side"
      f" | {args.steps} reader steps", flush=True)
for step, batch in enumerate(loaders.micro_batches(args.steps), 1):
    d = points(batch, gen)
    guess, px = pushed(d, gen)
    k = d["keep"]
    # The reader answers in metric pixels; the correction is minus the push.
    loss = (reader(read(d, guess)[k]) * args.push + px[k]).norm(dim=-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 250 == 0:
        print(f"  step {step:5d}  train error {loss.item():.2f} px", flush=True)

errs = [[] for _ in range(LOOKS + 1)]
moved, turned = [], []
reader.eval()
gen = torch.Generator().manual_seed(1)
with torch.no_grad():
    for batch in loaders.val:
        d = points(batch, gen)
        guess, _ = pushed(d, gen)
        k = d["keep"]
        to_px = (d["mult"] / d["scale"][:, None, None])[..., None]               # scene units per px
        off = lambda g: ((g - d["pos"]) / to_px).norm(dim=-1)[k].cpu()
        errs[0].append(off(guess))
        for look in range(1, LOOKS + 1):
            guess = guess + reader(read(d, guess)) * args.push * to_px
            errs[look].append(off(guess))
        moved.append(d["moved"][k].cpu()); turned.append(d["turned"][k].cpu())
row = lambda e: (f"median {e.median():5.2f} px  mean {e.mean():5.2f} px  "
                 + "  ".join(f"<{c}px {(e < c).float().mean():.3f}" for c in (1, 2, 4, 8)))
errs = [torch.cat(e) for e in errs]
moved, turned = torch.cat(moved), torch.cat(turned)
print(f"\nval, {len(errs[0])} visible moving point-frames, guess pushed up to {args.push:g} px a side")
print("  start (stay)   ", row(errs[0]))
for look in range(1, LOOKS + 1):
    print(f"  after look {look}   ", row(errs[look]))
print("\n  after the last look, against how far the point has moved")
print(motion_table(errs[-1], moved))
known = ~turned.isnan()
print("\n  after the last look, error / moved by how far the point turned since frame 0, degrees")
print(change_table(errs[-1][known], moved[known], turned[known], (0, 5, 15, 45, 180), "turned"))
print("\n  after the last look, median error in px by how far the point turned (any distance)")
for lo, hi in ((0, 5), (5, 15), (15, 45), (45, 180)):
    m = known & (turned >= lo) & (turned < hi)
    if m.sum():
        print(f"    turned {lo:g}-{hi:g}  n={int(m.sum()):<7} median {errs[-1][m].median():.2f}"
              f"  <1px {(errs[-1][m] < 1).float().mean():.3f}")


if not args.out:
    raise SystemExit(0)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
names = [c.stem for c in clips.val]
k = 0
with torch.no_grad():
    for batch in loaders.val:
        b = to_device(batch, dev)
        fine = fine_inputs(model, b)
        fmap, proj, xyz = fine["fine_map"].float(), fine["fine_proj"].float(), fine["fine_xyz"].float()
        B, T = fmap.shape[:2]
        pos = model._lookup_pos(b["traj"].float(), b["anchor"].float())
        gt = to_metres(b["traj"].float(), b)
        cam = _to_camera_t(gt, b["extrinsics"])
        focal = (b["intrinsics_256"][..., 0, 0] * b["intrinsics_256"][..., 1, 1]).sqrt()
        to_px = (cam[..., 2].abs() / focal[..., None] / b["norm_scale"].float()[:, None, None])[..., None]
        uv0 = project(b["anchor"].float()[:, None], proj[:, :1])[0][:, 0]
        track = [pos[:, 0]]
        for t in range(1, T):
            d = dict(fmap=fmap[:, t:t + 1], fmap0=fmap[:, 0], xyz=xyz[:, t:t + 1], xyz0=xyz[:, 0],
                     proj=proj[:, t:t + 1], uv0=uv0, pos0=b["anchor"].float())
            guess = track[-1][:, None]
            for _ in range(LOOKS):
                guess = guess + reader(read(d, guess)) * args.push * to_px[:, t:t + 1]
            track.append(guess[:, 0])
        track = torch.stack(track, dim=1)
        err = ((track - pos) / to_px).norm(dim=-1)                              # metric px
        pr_m = to_metres(model._target_units(track, b["anchor"].float()), b)
        vis = b["visibility"].bool()
        moving = motion_px(gt, vis, b["intrinsics_256"], b["extrinsics"])
        for i in range(B):
            name = names[k]
            k += 1
            one = {key: b[key][i:i + 1] for key in ("intrinsics", "extrinsics", "hw")}
            pick = moving[i].argsort(descending=True)[: args.draw]
            g, r = pixels(gt[i][:, pick], one), pixels(pr_m[i][:, pick], one)
            v = vis[i][:, pick].cpu().numpy()
            rgb = np.load(clips.frames / f"{name}.npy")
            cols = colours(len(pick))
            frames = [draw_frame(rgb[t], t, g, r, v, cols, 0) for t in range(T)]
            codec = write_video(frames, out / f"{name}.mp4")
            at = np.linspace(0, T - 1, 6).round().astype(int)
            cv2.imwrite(str(out / f"{name}.png"), np.hstack([frames[t] for t in at]))
            e = err[i][:, pick][vis[i][:, pick]]
            print(f"  {name}  drawn points, seen frames: median {e.median():.2f} px"
                  f"  <4px {(e < 4).float().mean():.3f}  last frame median"
                  f" {err[i, -1, pick][vis[i, -1, pick]].median():.2f} px -> {out / name}.mp4 [{codec}]",
                  flush=True)
        if k >= args.clips:
            break
