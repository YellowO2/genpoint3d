"""Check the stored world's bodies: is each object whole, the side never seen too, and does its description know what it is?"""

import torch

from genpoint3d.train.tools.batch import whole_frames
from genpoint3d.train.tools.world import World

EARLY, LATE = 4, 4   # the first and last frames with an image a clip is scored over, for what seeing more does


def shape_scores(b: dict, world: World, view: torch.Tensor, said: torch.Tensor) -> dict:
    """Per clip (1,) values, by name; empty for a clip with no answer (`mesh`).

    Each of an object's true corners against the nearest point the world has
    of that object in the same frame, as a share of the object's size (its
    two farthest corners apart), a mean over the corners, the objects and the
    frames. The world's points of an object are the clip's own points on it
    and the points imagined by a body formed about one of those.

    shape_back        the corners no frame with an image ever sees
    shape_back_seen   the same with the world's own points alone: nothing imagined
    shape_back_early  shape_back over the clip's first EARLY frames with an image
    shape_back_late   ... its last LATE
    shape_front       the corners some frame sees
    shape_made_up     each imagined point from its object's nearest corner
    imagined          imagined points in the clip

    And what each body's description says of its object after the clip (`said`
    (K, 7), `WorldModel.known`) against what Kubric says, a mean over the
    bodies formed about one of the clip's own points on an object:

    known_mass        how many times off its mass is (1: right)
    known_size        ... its longest side
    known_colour      how far off its colour is, a mean over red, green and blue from 0 to 1

    And the bars the bodies said between their imagined points, in the clip's
    last frame with an image, each end taken as its object's nearest corner:

    bars_right        the share of bars whose two corners the answer joins
    bars              bars an imagined point
    """
    mesh, of, seen = b["mesh"][0].float(), b["mesh_body"][0], b["mesh_seen"][0]
    if not mesh.shape[1]:
        return {}
    segment = b["segment"][0]
    own = len(segment)
    shown = torch.nonzero(whole_frames(view)[0])[:, 0]
    ever = seen[shown].any(0)                                                        # (V,)
    seed = world.seed[world.body.clamp_min(0)] if len(world.seed) else torch.full_like(world.body, own)
    thing = torch.where(world.imagined, torch.where(seed < own, segment[seed.clamp_max(own - 1)], -1),
                        torch.cat([segment, torch.full((len(world.body) - own,), -1, device=of.device)]))   # (N,) each point's object
    sums = {k: [] for k in ("shape_back", "shape_back_seen", "shape_back_early", "shape_back_late", "shape_front", "shape_made_up")}
    for o in of[of > 0].unique().tolist():                                         # the objects, not the floor
        mine, has = of == o, thing == o
        size = torch.cdist(mesh[0, mine], mesh[0, mine]).max().clamp_min(1e-6)
        for t in shown.tolist():
            there = has & world.alive(t)
            real = there & ~world.imagined
            if not real.any():
                continue
            gap = torch.cdist(mesh[t, mine], world.pos[t, there]) / size                       # (v, n)
            near, near_real = gap.min(1).values, gap[:, ~world.imagined[there]].min(1).values
            back = ~ever[mine]
            if back.any():
                sums["shape_back"].append(near[back].mean())
                sums["shape_back_seen"].append(near_real[back].mean())
                if t in shown[:EARLY]:
                    sums["shape_back_early"].append(near[back].mean())
                if t in shown[-LATE:]:
                    sums["shape_back_late"].append(near[back].mean())
            if (~back).any():
                sums["shape_front"].append(near[~back].mean())
            if (there & world.imagined).any():
                sums["shape_made_up"].append(gap[:, world.imagined[there]].min(0).values.mean())
    nan = torch.tensor(float("nan"), device=mesh.device)
    out = {k: (torch.stack(v).mean() if v else nan)[None] for k, v in sums.items()}
    out["imagined"] = world.imagined.sum()[None].float()
    last, bars = int(shown[-1]), world.bars
    if len(bars):
        V = max(len(of), 1)
        keys = b["mesh_bars"][0]
        keys = torch.cat([keys[:, 0] * V + keys[:, 1], keys[:, 1] * V + keys[:, 0]])
        ends = []
        for e in (bars[:, 0], bars[:, 1]):
            mine = of[None] == thing[e][:, None]                                         # (E, V) its object's corners
            gap = torch.cdist(world.pos[last, e], mesh[last]).masked_fill(~mine, float("inf"))
            ends.append(torch.where(mine.any(-1), gap.argmin(-1), -1))
        ok = (ends[0] >= 0) & (ends[1] >= 0) & (ends[0] != ends[1])
        out["bars_right"] = (ok & torch.isin(ends[0] * V + ends[1], keys)).float().mean()[None]
    else:
        out["bars_right"] = nan[None]
    out["bars"] = (len(bars) / world.imagined.sum().clamp_min(1)).float()[None]
    thing = torch.where(world.seed < own, segment[world.seed.clamp_max(own - 1)], -1)
    ok = (thing >= 1) & (thing <= b["things"].shape[1])
    off = (said.float() - b["things"][0].float()[(thing - 1).clamp_min(0)]).abs()[ok] if ok.any() else None
    for name, cols, as_times in (("known_mass", slice(0, 1), True), ("known_size", slice(3, 4), True), ("known_colour", slice(4, 7), False)):
        out[name] = (nan if off is None else off[:, cols].exp().mean() if as_times else off[:, cols].mean())[None]
    return out
