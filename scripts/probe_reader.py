"""Can the score tables alone say how far a point moved? Only a reader trains.

The CNN stays frozen at its pretrained weights. A small reader learns from
the 49 x 49 tables, the four levels of them, to say how far the true
position is from a guess: the guess is the true position in frame t pushed
off by up to --push pixels, the template is the query in frame 0, and the
answer is Kubric's own track. Visible moving points, frames 1 on.

  reader error small   the features hold the motion, and a model that reads
                       them badly is a training problem, not a feature one
  reader error large   the tables themselves are too vague to read

Scored on the val clips the way the tracker uses it, four reads, each from
the last one's guess, in pixels of the frame the CNN sees.

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


def points(batch, generator):
    """The frozen maps and the visible moving points of a few random frames
    per clip: frame 0's map, frame t's maps (B, F, C, h, w), the query in
    frame 0 (B, N, 2), the truth in frame t (B, F, N, 2) in [0, 1], which
    ones count (B, F, N), and the frame's width in pixels."""
    b = to_device(batch, dev)
    with torch.no_grad():
        fine = fine_inputs(model, b)
        fmap, proj = fine["fine_map"].float(), fine["fine_proj"].float()
        B, T, C, h, w = fmap.shape
        pos = model._lookup_pos(b["traj"].float(), b["anchor"].float())
        uv, _ = project(pos, proj)                                             # (B, T, N, 2)
        moving = motion_px(to_metres(b["traj"], b), b["visibility"], b["intrinsics_256"],
                           b["extrinsics"]) > 4
        t = torch.stack([torch.randperm(T - 1, generator=generator)[:FRAMES_PER_CLIP] + 1
                         for _ in range(B)]).to(dev)                           # (B, F)
        uv_t = uv.gather(1, t[..., None, None].expand(-1, -1, *uv.shape[2:]))
        keep = b["visibility"].gather(1, t[..., None].expand(-1, -1, uv.shape[2])) & moving[:, None]
        keep &= ((uv_t > 0) & (uv_t < 1)).all(-1)
        f_t = fmap.gather(1, t[..., None, None, None].expand(-1, -1, C, h, w))
    return fmap[:, 0], f_t, uv[:, 0], uv_t, keep, b["frames"].shape[3]


def tables_at(f0, f_t, uv0, guess):
    """(B, F, N, LEVELS, K*K): the four levels of tables, the template at the
    query in frame 0, the search window at `guess` in frame t."""
    B, F_, C, h, w = f_t.shape
    size = uv0.new_tensor([w, h])
    out = []
    with torch.no_grad():
        for sup, win in zip(finder._windows(f0, uv0 * size - 0.5 / STRIDE),
                            finder._windows(f_t.flatten(0, 1), (guess * size - 0.5 / STRIDE).flatten(0, 1))):
            win = win.reshape(B, F_, -1, K, C)
            out.append(torch.einsum("bnsc,bfnkc->bfnsk", sup, win).flatten(3) * C ** -0.5)
    return torch.stack(out, dim=3)


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
    f0, f_t, uv0, uv_t, keep, W = points(batch, gen)
    push = (torch.rand(uv_t.shape, generator=gen).to(dev) * 2 - 1) * PUSH          # px
    x = tables_at(f0, f_t, uv0, uv_t + push / W)[keep]
    loss = (reader(x) + push[keep]).norm(dim=-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 250 == 0:
        print(f"  step {step:5d}  train error {loss.item():.2f} px", flush=True)

# Scored the way the tracker uses it: from a guess pushed off, read, move the
# window to the new guess, read again.
LOOKS = 4
errs = [[] for _ in range(LOOKS + 1)]
reader.eval()
gen = torch.Generator().manual_seed(1)
with torch.no_grad():
    for batch in loaders.val:
        f0, f_t, uv0, uv_t, keep, W = points(batch, gen)
        push = (torch.rand(uv_t.shape, generator=gen).to(dev) * 2 - 1) * PUSH
        guess = uv_t + push / W
        errs[0].append(((guess - uv_t) * W).norm(dim=-1)[keep])
        for look in range(1, LOOKS + 1):
            tab = tables_at(f0, f_t, uv0, guess)
            guess = guess + reader(tab.flatten(0, 2)).view(*guess.shape) / W
            errs[look].append(((guess - uv_t) * W).norm(dim=-1)[keep])
row = lambda e: (f"median {e.median():5.2f} px  mean {e.mean():5.2f} px  "
                 + "  ".join(f"<{k}px {(e < k).float().mean():.3f}" for k in (1, 2, 4, 8)))
errs = [torch.cat(e) for e in errs]
print(f"\nval, {len(errs[0])} visible moving point-frames, guess pushed up to {PUSH:g} px")
print("  start (stay)   ", row(errs[0]))
for look in range(1, LOOKS + 1):
    print(f"  after look {look}   ", row(errs[look]))
