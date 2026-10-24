"""Can the score tables alone say how far a point moved? Only a reader trains.

The CNN stays frozen at its pretrained weights. A small reader learns from
the 49 x 49 tables, the four levels of them, to say how far the true
position is from a guess: the guess is the true position in frame t pushed
off by up to --push pixels, the template is the query in frame 0, and the
answer is Kubric's own track. Visible moving points, frames 1 on.

  reader error small   the features hold the motion, and a model that reads
                       them badly is a training problem, not a feature one
  reader error large   the tables themselves are too vague to read

Scored on the val clips against "stay at the guess", in pixels of the
frame the CNN sees.

  python scripts/probe_reader.py --cache <cache> --frames-cache <frames> \\
      --fine 1 --fine-init <weights.pt> --dino 0 [--push 16] [--probe-steps 3000]
"""

import sys

import torch
import torch.nn.functional as F
from torch import nn

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.Loaders import Loaders
from genpoint3d.train.data.apply_run_settings import apply_run_settings
from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.model.BasicEncoder import STRIDE
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.model.set_up_model import set_up_model
from genpoint3d.train.tools.batch import fine_inputs, to_device, to_metres
from genpoint3d.train.tools.camera import project
from genpoint3d.train.tools.metrics import motion_px


def _take(argv, name, default, kind):
    if name in argv:
        i = argv.index(name)
        value = kind(argv[i + 1])
        del argv[i:i + 2]
        return value
    return default


argv = sys.argv[1:]
PUSH = _take(argv, "--push", 16.0, float)
STEPS = _take(argv, "--probe-steps", 3000, int)
FRAMES_PER_CLIP = 6
settings = Settings.from_command_line(argv + ["--workers", "2", "--target", "displacement"])
hardware = Hardware.pick(settings)
dev = hardware.device
clips = apply_run_settings(split_clips(settings.cache, settings.val_frac, settings.seed), settings, hardware)
model = set_up_model(build_model(settings, clips.feat_dim, clips.has_feats), settings, hardware).eval()
finder = model.fine
LEVELS, K = finder.levels, (2 * finder.radius + 1) ** 2


def samples(batch, generator):
    """Tables, the push and the true offset for one batch's visible moving
    points in a few random frames: (S, LEVELS, K*K), (S, 2) px, (S, 2) px."""
    b = to_device(batch, dev)
    with torch.no_grad():
        fine = fine_inputs(model, b)
        fmap, proj = fine["fine_map"].float(), fine["fine_proj"].float()
        B, T, C, h, w = fmap.shape
        W = b["frames"].shape[3]
        pos = model._lookup_pos(b["traj"].float(), b["anchor"].float())
        uv, _ = project(pos, proj)                                             # (B, T, N, 2)
        moving = motion_px(to_metres(b["traj"], b), b["visibility"], b["intrinsics_256"],
                           b["extrinsics"]) > 4
        t = torch.stack([torch.randperm(T - 1, generator=generator)[:FRAMES_PER_CLIP] + 1
                         for _ in range(B)]).to(dev)                           # (B, F)
        uv_t = uv.gather(1, t[..., None, None].expand(-1, -1, *uv.shape[2:]))  # (B, F, N, 2)
        keep = b["visibility"].gather(1, t[..., None].expand(-1, -1, uv.shape[2])) & moving[:, None]
        keep &= ((uv_t > 0) & (uv_t < 1)).all(-1)
        push = (torch.rand(uv_t.shape, generator=generator).to(dev) * 2 - 1) * PUSH   # px
        guess = uv_t + push / W
        size = uv.new_tensor([w, h])
        f_t = fmap.gather(1, t[..., None, None, None].expand(-1, -1, C, h, w))
        tables = []
        for sup, win in zip(finder._windows(fmap[:, 0], uv[:, 0] * size - 0.5 / STRIDE),
                            finder._windows(f_t.flatten(0, 1), (guess * size - 0.5 / STRIDE).flatten(0, 1))):
            win = win.reshape(B, t.shape[1], -1, K, C)
            tables.append(torch.einsum("bnsc,bfnkc->bfnsk", sup, win).flatten(3) * C ** -0.5)
        tables = torch.stack(tables, dim=3)                                    # (B, F, N, L, K*K)
    return tables[keep], push[keep], -push[keep]


class Reader(nn.Module):
    """Each level's table through its own small MLP, as the finder reads it,
    then one more to an offset in pixels."""

    def __init__(self):
        super().__init__()
        self.levels = nn.ModuleList(nn.Sequential(nn.Linear(K * K, 256), nn.GELU(), nn.Linear(256, 128))
                                    for _ in range(LEVELS))
        self.head = nn.Sequential(nn.GELU(), nn.Linear(128 * LEVELS, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, tables):
        return self.head(torch.cat([m(tables[:, i]) for i, m in enumerate(self.levels)], -1)) * PUSH


loaders = Loaders(clips, settings)
reader = Reader().to(dev)
opt = torch.optim.AdamW(reader.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=STEPS)
gen = torch.Generator().manual_seed(0)
print(f"frozen CNN from {settings.fine_init} | push up to {PUSH:g} px | {STEPS} reader steps", flush=True)
for step, batch in enumerate(loaders.micro_batches(STEPS), 1):
    x, _, y = samples(batch, gen)
    loss = (reader(x) - y).norm(dim=-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 250 == 0:
        print(f"  step {step:5d}  train error {loss.item():.2f} px", flush=True)

err, stay = [], []
reader.eval()
gen = torch.Generator().manual_seed(1)
with torch.no_grad():
    for batch in loaders.val:
        x, push, y = samples(batch, gen)
        err.append((reader(x) - y).norm(dim=-1))
        stay.append(push.norm(dim=-1))
err, stay = torch.cat(err), torch.cat(stay)
row = lambda e: (f"median {e.median():.2f} px  mean {e.mean():.2f} px  "
                 + "  ".join(f"<{k}px {(e < k).float().mean():.3f}" for k in (1, 2, 4, 8)))
print(f"\nval, {len(err)} visible moving point-frames")
print("  stay at the guess  ", row(stay))
print("  reader             ", row(err))
