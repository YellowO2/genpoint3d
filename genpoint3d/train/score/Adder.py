"""Points added at test time as training adds them (`add_points`), without the truth."""

from typing import Optional

import torch

from genpoint3d.tools.feature_lookup import id_cards
from genpoint3d.tools.traj_scale import REACH
from genpoint3d.train.tools.batch import whole_frames
from genpoint3d.train.tools.camera import project
from genpoint3d.train.tools.farthest_first import SURFACE_TOL, add_points, spacing


class Adder:
    """For one clip, the points to add in a frame, up to `points` in the clip:
    on the CNN cells of a frame seen whole whose surface has no point within
    its spacing. A point covers its pixel if the world has it at the depth
    shown there. An added point is given where its cell is, with the look
    read there; it has no truth and is never scored."""

    def __init__(self, b: dict, fine: dict, view: torch.Tensor, points: int, gap: float) -> None:
        if b["traj"].shape[0] != 1:
            raise ValueError("points are added one clip at a time (--batch 1)")
        self.points, self.gap = points, gap
        self.proj = fine["fine_proj"][0].float()                                         # (T, 3, 4)
        self.xyz = fine["fine_xyz"][0].flatten(2).transpose(1, 2)                         # (T, h*w, 3)
        self.h, self.w = fine["fine_xyz"].shape[-2:]
        self.depth = b["depth"][0].flatten(1).float()                                     # (T, h*w)
        self.H, self.W = b["hw"][0].tolist()
        self.focal = b["intrinsics"][0, :, 0, 0]
        self.whole = whole_frames(view[0])
        self.context = b["context"][0]
        dev = self.depth.device
        v, u = torch.meshgrid(torch.arange(self.h, device=dev), torch.arange(self.w, device=dev), indexing="ij")
        self.cell_px = torch.stack([(u.flatten() + 0.125) / self.w * self.W, (v.flatten() + 0.125) / self.h * self.H], -1).float()

    @torch.no_grad()
    def __call__(self, t: int, pos: torch.Tensor, alive: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """The points to add in frame `t`, the world having its points at `pos` (N, 3) there, those `alive`
        (N,): where each is given (M, 3) and its id card (M, F); None for none."""
        room = self.points - len(pos)
        if room <= 0 or not self.whole[t] or not alive.any():
            return None
        z = self.depth[t]
        cand = torch.nonzero(z < REACH)[:, 0]
        far = spacing(z[cand], float(self.focal[t]), self.W, self.gap)
        uv, at = project(pos[alive].float()[None, None], self.proj[None, t:t + 1])
        uv, at = uv[0, 0], at[0, 0]
        px = (uv[:, 1] * self.h).long().clamp(0, self.h - 1) * self.w + (uv[:, 0] * self.w).long().clamp(0, self.w - 1)
        on = (at - z[px]).abs() < SURFACE_TOL * z[px]
        cover = uv[on] * uv.new_tensor([self.W, self.H])
        cells = [c for _, c in add_points([t], room, lambda *_: (self.cell_px[cand], cand, far), lambda *_: cover)]
        if not cells:
            return None
        cells = torch.tensor(cells, device=pos.device)
        card = id_cards(self.context, torch.full_like(cells, t), self.cell_px[cells], (self.H, self.W))
        return self.xyz[t, cells], card
