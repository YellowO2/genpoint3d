"""Does the finder see the right place on real clips? Scoring only, no training.

Two checks on a few real batches, with the true tracks:

  1. place    project each point's TRUE 3D position into its frame, read the
              3D position of the CNN cell there (from the depth), and compare.
              For a visible point they should agree to about a cell. If they
              do not in later frames, the camera bookkeeping puts every window
              in the wrong place, and only a whole-frame match could hide it.
  2. match    put the 7x7 window at the TRUE position and ask where the
              template's centre cell matches best. With the right features it
              is the centre, or next to it. Also how big the table's numbers
              are, which is what the network reading them sees.

Both by third of the clip, visible moving points only.

  python scripts/check_fine_inputs.py --cache <cache> --frames-cache <frames> \\
      --fine 1 --fine-xyz 1 --fine-init <cotracker3_fnet.pt> --dino 0
"""

import sys

import torch
import torch.nn.functional as F

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

settings = Settings.from_command_line(sys.argv[1:] + ["--workers", "0", "--target", "displacement"])
hardware = Hardware.pick(settings)
clips = apply_run_settings(split_clips(settings.cache, settings.val_frac, settings.seed), settings, hardware)
model = set_up_model(build_model(settings, clips.feat_dim, clips.has_feats), settings, hardware).eval()
finder = model.fine

dist_cells, centre_hit, near_hit, scale = [], [], [], []
with torch.no_grad():
    for n, batch in enumerate(Loaders(clips, settings).val):
        if n == 4:
            break
        b = to_device(batch, hardware.device)
        fine = fine_inputs(model, b)
        fmap, proj, xyz = fine["fine_map"].float(), fine["fine_proj"].float(), fine["fine_xyz"].float()
        B, T, C, h, w = fmap.shape
        pos = model._lookup_pos(b["traj"].float(), b["anchor"].float())        # true, scene-normalised
        uv, _ = project(pos, proj)
        moving = motion_px(to_metres(b["traj"], b), b["visibility"], b["intrinsics_256"], b["extrinsics"]) > 4
        keep = b["visibility"] & moving[:, None]                                 # (B, T, N)

        # 1. the cell's 3D position at the projected pixel vs the true position
        grid = (2 * uv - 1)                                                      # [0,1] -> [-1,1]
        cell = F.grid_sample(xyz.flatten(0, 1), grid.flatten(0, 1)[:, :, None], align_corners=False)
        cell = cell[..., 0].transpose(1, 2).reshape(B, T, -1, 3)
        spacing = (xyz[..., :, 1:] - xyz[..., :, :-1]).norm(dim=2).median()     # one cell, scene units
        dist_cells.append(((cell - pos).norm(dim=-1) / spacing, keep))

        # 2. the window at the true place: where does the template centre match?
        size = uv.new_tensor([w, h])
        sup = next(finder._windows(fmap[:, 0], uv[:, 0] * size - 0.5 / STRIDE))          # (B, N, 49, C)
        win = next(finder._windows(fmap.flatten(0, 1), (uv * size - 0.5 / STRIDE).flatten(0, 1)))
        win = win.reshape(B, T, -1, 49, C)
        sc = torch.einsum("bnc,btnkc->btnk", sup[:, :, 24], win) * C ** -0.5
        best = sc.argmax(-1)
        oy, ox = (best // 7 - 3).abs(), (best % 7 - 3).abs()
        centre_hit.append((best == 24, keep))
        near_hit.append(((oy <= 1) & (ox <= 1), keep))
        scale.append(sc[keep])

thirds = torch.arange(1, T).tensor_split(3)
def by_third(pairs, fn):
    out = []
    for fr in thirds:
        vals = torch.cat([v[:, fr][k[:, fr]].float() for v, k in pairs])
        out.append(fn(vals))
    return " / ".join(f"{x:.3f}" for x in out)

print("frames 1-8 / 9-16 / 17-23, visible moving points")
print("1. cell at projected pixel vs true 3D, median distance in cells:", by_third(dist_cells, torch.median))
print("   share within 1 cell:", by_third(dist_cells, lambda v: (v < 1).float().mean()))
print("2. template centre matches best at the true cell:", by_third(centre_hit, torch.mean))
print("   ... or a neighbouring cell:", by_third(near_hit, torch.mean))
s = torch.cat(scale)
print(f"   table numbers: mean {s.mean():.2f}  std {s.std():.2f}  min {s.min():.2f}  max {s.max():.2f}")
