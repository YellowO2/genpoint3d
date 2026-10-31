"""Follow points from each frame to the next, and turning surfaces as one piece.

The core of docs/tasks/scene-points-2026-10-31.md, on its own before it joins
the main model. Two parts:

    step     where each point went between frame t-1 and frame t: its 3D
             neighbourhood in frame t-1, at where it was, compared with the
             neighbourhood around a guess in frame t, by TAPIP3D's comparison
             (`FineFinder`, its CNN and its weights), a few times over, each
             from the last guess, as RAFT refines a flow. A step is small even
             on a surface that turns fast, where frame 0's look is long gone.
             With it, how sure it is the point is seen in both frames.
    groups   which points move as one rigid piece: a vector per point from
             what it is (DINOv3), how it looks (the CNN) and where it is,
             points with close vectors being one piece.

Chained over a clip (`track`), each piece's step is the one rigid motion that
best explains its seen members' steps, and every point, seen or hidden, is
carried by its piece's: a spinning ball is followed by all of its visible
surface at once, which a point alone cannot do (runs.md, the spinning ball).

Positions are in the scene-normalised units `anchor` has.
"""

import torch
import torch.nn.functional as F
from torch import nn

from genpoint3d.train.model.FineFinder import FineFinder, _posenc
from genpoint3d.train.tools.camera import project, unproject


class StepTracker(nn.Module):
    def __init__(self, feat_dim: int, emb: int = 16, iters: int = 3, neigh: int = 32) -> None:
        super().__init__()
        self.iters = iters
        self.finder = FineFinder(levels=4, xyz=True, neigh=neigh)
        d = self.finder.out_dim
        # A step correction (3) and a "seen in both frames" logit, from the
        # comparison and the step so far.
        self.step_head = nn.Sequential(nn.Linear(d + 99, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                       nn.Linear(256, 4))
        nn.init.zeros_(self.step_head[-1].weight)
        nn.init.zeros_(self.step_head[-1].bias)
        self.group_head = nn.Sequential(nn.Linear(feat_dim + 128 + 99, 256), nn.GELU(), nn.Linear(256, emb))

    def steps(self, fmap_a, fmap_b, xyz_a, xyz_b, proj_a, proj_b, pos_a):
        """One step for many frame pairs at once, M = pairs: fmap (M, C, h, w),
        xyz (M, 3, h, w), proj (M, 3, 4), pos_a (M, N, 3) where the points are
        in frame a -> their guess in frame b after each refinement, a list of
        (M, N, 3), and the logits of "seen in both", (M, N)."""
        uv_a, _ = project(pos_a[:, None], proj_a[:, None])
        guess, out = pos_a, []
        for _ in range(self.iters):
            uv_b, _ = project(guess[:, None], proj_b[:, None])
            c = self.finder(fmap_b[:, None], fmap_a, uv_b, uv_a[:, 0], xyz=xyz_b[:, None], xyz0=xyz_a,
                            pos=guess[:, None], pos0=pos_a)[:, 0]
            h = self.step_head(torch.cat([c, _posenc((guess - pos_a).float())], -1))
            # Corrections in units of a few centimetres of a unit-spread scene.
            guess = guess.detach() + 0.05 * h[..., :3]
            out.append(guess)
        return out, h[..., 3]

    def groups(self, id_card, fmap0, uv0, anchor):
        """(B, N, E) group vectors from what each point is (`id_card`, DINOv3,
        (B, N, D)), how it looks in frame 0 (the CNN map (B, C, h, w) at uv0
        (B, N, 2)) and where it is (`anchor`)."""
        grid = 2 * uv0 - 1
        look = F.grid_sample(fmap0.float(), grid[:, :, None], align_corners=False)[..., 0].transpose(1, 2)
        return self.group_head(torch.cat([id_card.float(), look, _posenc(anchor.float())], -1))

    @staticmethod
    def flow_steps(flow_f, flow_b, xyz_b, proj_a, proj_b, pos_a):
        """The same step from a cached optical flow instead of the comparison
        (`scripts/cache_flow.py`): flow_f (M, 2, h, w) frame a -> b and flow_b
        (M, 2, h, w) b -> a, fractions of the frame at the CNN's cells; xyz_b
        (M, 3, h, w) frame b's cells in 3D. Each point's pixel in a moved by
        the forward flow, at the depth of the cell it lands on; "seen in both"
        as a logit from the flow's own forward-backward check, high when going
        there and back returns within a cell."""
        uv_a, _ = project(pos_a[:, None], proj_a[:, None])
        uv_a = uv_a[:, 0]
        uv_b = uv_a + _at_cells(flow_f, uv_a, "bilinear")
        back = uv_b + _at_cells(flow_b, uv_b, "bilinear")
        h, w = flow_f.shape[-2:]
        gap = ((back - uv_a) * uv_a.new_tensor([w, h])).norm(dim=-1)                   # in cells
        _, z = project(_at_cells(xyz_b, uv_b, "nearest")[:, None], proj_b[:, None])
        return unproject(uv_b[:, None], z, proj_b[:, None])[:, 0], 4.0 * (1.0 - gap)

    @torch.no_grad()
    def track(self, fmap, xyz, proj, anchor, emb, rigid: bool = True, flow=None):
        """Chain the steps over a clip: fmap (B, T, C, h, w), xyz (B, T, 3, h, w),
        proj (B, T, 3, 4), anchor (B, N, 3), emb (B, N, E) -> (B, T, N, 3).
        With `rigid`, each point moves by the rigid step its piece's seen
        members make: the rotation and shift that best carry them from t-1 to
        where their steps put them, weighted by how sure each is it is seen and
        by how close its vector is; without, by its own step. With `flow`,
        (forward (B, T-1, 2, h, w), backward likewise), the steps come from it
        (`flow_steps`) instead of the comparison."""
        B, T = fmap.shape[:2]
        a = 2 * torch.sigmoid(-(emb[:, :, None] - emb[:, None]).pow(2).sum(-1))       # (B, N, N)
        pos = [anchor.float()]
        for t in range(1, T):
            if flow is None:
                guesses, seen = self.steps(fmap[:, t - 1], fmap[:, t], xyz[:, t - 1], xyz[:, t],
                                           proj[:, t - 1], proj[:, t], pos[-1])
                moved = guesses[-1]
            else:
                moved, seen = self.flow_steps(flow[0][:, t - 1], flow[1][:, t - 1], xyz[:, t],
                                              proj[:, t - 1], proj[:, t], pos[-1])
            if not rigid:
                pos.append(moved)
                continue
            # A floor under each weight, so a piece none of whose members is
            # sure it is seen still averages them rather than dividing by
            # almost nothing.
            w = a * (torch.sigmoid(seen) + 1e-3)[:, None]                                 # (B, i, j)
            tot = w.sum(-1, keepdim=True)
            src, dst = pos[-1], moved
            cs, cd = (w @ src) / tot, (w @ dst) / tot
            A, Bm = src[:, None] - cs[:, :, None], dst[:, None] - cd[:, :, None]
            U, _, Vh = torch.linalg.svd(torch.einsum("bij,bijx,bijy->bixy", w, A, Bm))
            flip = torch.ones_like(U[..., 0])
            flip[..., 2] = torch.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2)).sign()
            R = Vh.transpose(-1, -2) @ torch.diag_embed(flip) @ U.transpose(-1, -2)
            pos.append(torch.einsum("bixy,biy->bix", R, src - cs) + cd)
        return torch.stack(pos, 1)


def _at_cells(grid: torch.Tensor, uv: torch.Tensor, mode: str) -> torch.Tensor:
    """grid (M, C, h, w) on the CNN's cells, uv (M, N, 2) in [0, 1] -> (M, N, C).
    Cell j is centred at `(j + 1/8) / w` across (FineFinder's docstring)."""
    h, w = grid.shape[-2:]
    cell = uv * uv.new_tensor([w, h]) - 0.125
    g = 2 * cell / cell.new_tensor([w - 1, h - 1]) - 1
    return F.grid_sample(grid.float(), g[:, :, None], mode=mode, align_corners=True)[..., 0].transpose(1, 2)
