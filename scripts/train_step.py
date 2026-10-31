"""Train the step tracker (`genpoint3d/train/model/StepTracker.py`) on its own.

Each point's step from frame t-1 to frame t, from its true position in t-1,
for a few random frame pairs of each clip at once (all pairs are independent
given the true positions); and which points move as one piece. The CNN stays
as TAPIP3D trained it; the comparison starts from TAPIP3D's weights.

  loss = steps    the guess after each refinement against the true position
                  in t, for points seen in both frames, the last weighted most
       + seen     "seen in both frames", cross-entropy
       + groups   vectors against the pairs the truth shows moving as one

  python scripts/train_step.py --cache DENSE --frames-cache FRAMES \\
      --fine-init weights/tapip3d_finder.pt --out outputs/step_a [--steps 1500]
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.model.StepTracker import StepTracker
from genpoint3d.train.tools.batch import _cell_xyz, rigid_groups, to_device
from genpoint3d.train.tools.camera import image_projection, project
from genpoint3d.train.tools.regress import _group_error

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--frames-cache", required=True)
p.add_argument("--fine-init", required=True)
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--batch", type=int, default=2)
p.add_argument("--points", type=int, default=256)
p.add_argument("--pairs", type=int, default=8, help="frame pairs of each clip per step")
p.add_argument("--lr", type=float, default=3e-4)
args = p.parse_args()

dev = torch.device("cuda")
split = split_clips(args.cache, 0.1, 0)
ds = ClipDataset(split.train, args.points, resample=True, norm_mode="median", target="displacement",
                 fix_idcard=True, frames=Path(args.frames_cache), features=True, depth=True)
loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    persistent_workers=True)
model = StepTracker(split.feat_dim).to(dev)
sd = torch.load(args.fine_init, map_location="cpu")
missing, unexpected = model.finder.load_state_dict(sd, strict=False)
if unexpected:
    raise ValueError(f"{args.fine_init} does not fit the finder: {unexpected}")
model.finder.fnet.requires_grad_(False)
params = [q for q in model.parameters() if q.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
w = [0.8 ** (model.iters - 1 - i) for i in range(model.iters)]
w = [v / sum(w) for v in w]

step, t0, run = 0, time.time(), {"step": 0.0, "seen": 0.0, "group": 0.0, "err_px": 0.0}
while step < args.steps:
    for batch in loader:
        b = to_device(batch, dev)
        B, T = b["traj"].shape[:2]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            fmap = model.finder.encode(b["frames"])
        fmap = fmap.float()
        proj = image_projection(b["intrinsics"], b["extrinsics"], b["hw"], b["norm_scale"], b["norm_mean"]).float()
        xyz = _cell_xyz(b["depth"].float(), proj)
        pos = b["traj"].float() * b["norm_traj_scale"][:, None, None, None] + b["anchor"][:, None].float()
        vis = b["visibility"].bool()
        t = torch.stack([torch.randperm(T - 1)[: args.pairs] + 1 for _ in range(B)]).to(dev)      # (B, K)
        bi = torch.arange(B, device=dev)[:, None].expand_as(t)
        pick = lambda x, s: x[bi, t + s].flatten(0, 1)
        guesses, seen = model.steps(pick(fmap, -1), pick(fmap, 0), pick(xyz, -1), pick(xyz, 0),
                                    pick(proj, -1), pick(proj, 0), pick(pos, -1))
        target, both = pick(pos, 0), pick(vis, -1) & pick(vis, 0)                             # (M, N, 3), (M, N)
        step_loss = sum(wi * ((g - target).norm(dim=-1) * both).sum() / both.sum().clamp_min(1)
                        for wi, g in zip(w, guesses))
        seen_loss = F.binary_cross_entropy_with_logits(seen, both.float())
        uv0, _ = project(b["anchor"].float()[:, None], proj[:, :1])
        emb = model.groups(b["id_card"], fmap[:, 0], uv0[:, 0], b["anchor"])
        group_loss = _group_error(emb, rigid_groups(b))
        loss = step_loss + 0.1 * seen_loss + group_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        # The last guess's error in pixels of the 384 px frame, for reading.
        with torch.no_grad():
            u_g, _ = project(guesses[-1][:, None], pick(proj, 0)[:, None])
            u_t, _ = project(target[:, None], pick(proj, 0)[:, None])
            px = ((u_g - u_t)[:, 0] * 384).norm(dim=-1)[both].median().item() if both.any() else 0.0
        for k, v in (("step", step_loss.item()), ("seen", seen_loss.item()), ("group", group_loss.item()),
                     ("err_px", px)):
            run[k] += v
        if step % 100 == 0:
            print(f"  step {step:5d}  " + "  ".join(f"{k} {v / 100:.4f}" for k, v in run.items())
                  + f"  {(time.time() - t0) / step:.2f}s/it", flush=True)
            run = {k: 0.0 for k in run}
        if step % 500 == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "step": step, "feat_dim": split.feat_dim, "args": vars(args)},
                       out / "step.pt")
        if step >= args.steps:
            break
print("=== TRAIN STEP DONE ===")
