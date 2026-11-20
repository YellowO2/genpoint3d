"""Points added as a clip goes where the model says the stored world needs one (`WorldModel.select`)."""

from typing import Optional

import torch

from genpoint3d.tools.feature_lookup import id_cards
from genpoint3d.tools.traj_scale import REACH
from genpoint3d.train.tools.batch import whole_frames
from genpoint3d.train.tools.select import apart


class Adder:
    """For one clip, the points to add in a frame seen whole, up to `points`
    given by frames in the clip: on the CNN cells the model says show a
    corner of a body's mesh with no stored point near, best first, none
    within `select.NEAR` cells of another. An added point is given where its
    cell is, with the look read there; it has no truth and is never scored."""

    def __init__(self, model, b: dict, fine: dict, view: torch.Tensor, points: int) -> None:
        if b["traj"].shape[0] != 1:
            raise ValueError("points are added one clip at a time (--batch 1)")
        self.model, self.points = model, points
        self.proj = fine["fine_proj"][0].float()                                         # (T, 3, 4)
        self.xyz = fine["fine_xyz"][0].flatten(2).transpose(1, 2)                         # (T, h*w, 3)
        self.h, self.w = fine["fine_xyz"].shape[-2:]
        self.depth = b["depth"][0].float()                                                # (T, h, w)
        self.H, self.W = b["hw"][0].tolist()
        self.whole = whole_frames(view[0])
        self.context = b["context"][0]
        dev = self.depth.device
        v, u = torch.meshgrid(torch.arange(self.h, device=dev), torch.arange(self.w, device=dev), indexing="ij")
        self.cell_px = torch.stack([(u.flatten() + 0.125) / self.w * self.W, (v.flatten() + 0.125) / self.h * self.H], -1).float()

    @torch.no_grad()
    def __call__(self, t: int, logits: torch.Tensor, cover: torch.Tensor, given: int) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """The points to add in frame `t`, from where the model says the world needs one, `logits` (h * w,), the
        cells its points explain `cover` (h, w), and how many points frames gave so far: where each is given
        (M, 3), its id card (M, F) and its cell (M,); None for none."""
        room = self.points - given
        if room <= 0 or not self.whole[t]:
            return None
        cand = (logits > 0) & ~cover.flatten() & (self.depth[t].flatten() < REACH)
        order = torch.nonzero(cand)[:, 0]
        cells = apart(order[logits[order].argsort(descending=True)], room, self.h, self.w)
        if not cells:
            return None
        cells = torch.tensor(cells, device=logits.device)
        card = id_cards(self.context, torch.full_like(cells, t), self.cell_px[cells], (self.H, self.W))
        return self.xyz[t, cells], card, cells
