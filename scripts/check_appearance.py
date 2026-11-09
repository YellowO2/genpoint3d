"""Can DINOv3 alone tell which points are on one object? On cached clips that
say which object each point is on, each point's DINOv3 feature at its query
pixel (its id card) against every other point's: how well cosine similarity
separates pairs on one object from pairs on two (AUC), and for each point,
of its most similar points (as many as its object has), the share on its
object. Points on moving objects and on the background apart.

Run:  python scripts/check_appearance.py CACHE_DIR [--clips 60]
"""

import argparse
from pathlib import Path

import torch

from genpoint3d.CachedClip import CachedClip

p = argparse.ArgumentParser()
p.add_argument("cache")
p.add_argument("--clips", type=int, default=60)
args = p.parse_args()


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """The chance a same-object pair is more similar than a two-object pair."""
    scores = torch.cat([pos, neg])
    ranks = scores.argsort().argsort().float() + 1
    return ((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()


aucs, prec = [], {"moving": [], "background": [], "still object": []}
for path in sorted(Path(args.cache).glob("*.pt"))[: args.clips]:
    c = CachedClip.from_dict(torch.load(path, weights_only=False))
    if c.segment is None:
        continue
    f = torch.nn.functional.normalize(c.id_card.float(), dim=-1)
    seg = c.segment.long()
    sim = f @ f.T
    off = ~torch.eye(len(seg), dtype=torch.bool)
    same = (seg[:, None] == seg[None]) & off
    aucs.append(auc(sim[same], sim[~same & off]))
    moved = (c.traj - c.traj[:1]).norm(dim=-1).amax(0) > 0.01                  # (N,)
    sizes = same.sum(-1)
    top = sim.masked_fill(~off, -2).argsort(-1, descending=True)
    for i in range(len(seg)):
        k = int(sizes[i])
        if k == 0:
            continue
        hit = (seg[top[i, :k]] == seg[i]).float().mean().item()
        kind = "background" if seg[i] == 0 else ("moving" if moved[i] else "still object")
        prec[kind].append(hit)

print(f"{len(aucs)} clips from {args.cache}")
print(f"  same object vs two objects, by DINOv3 similarity: AUC {torch.tensor(aucs).mean():.3f} "
      f"(0.5 chance, 1 perfect)")
print("  of each point's most similar points (as many as its object has), the share on its object:")
for kind, v in prec.items():
    if v:
        print(f"    {kind:<13} {torch.tensor(v).mean():.3f}  ({len(v)} points)")
