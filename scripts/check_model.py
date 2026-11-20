"""The model and its loops at toy size, on the CPU in seconds: one training
step reaches every weight, and the tracks are unchanged by anything in the
frames or quarters of frames they have no image of: after a cutoff, in a
gap, or in a hidden quarter. Run after any change to the model or the loops:

  .venv/bin/python scripts/check_model.py
"""

import types

import torch

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.learn import learn
from genpoint3d.train.model.PointDiT import PointDiT
from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.score.track_scores import track_scores
from genpoint3d.train.score.visibility_scores import visibility_scores
from genpoint3d.train.tools.batch import black_out, fine_inputs, random_view, to_device
from genpoint3d.train.tools.regress import refine

torch.manual_seed(0)
B, T, N, P, F, R = 1, 16, 6, 16, 24, 64       # R: frame side, CNN map R/4; windows of 8 sharing 4: three


def batch():
    traj = torch.randn(B, T, N, 3) * 0.1
    traj[:, 0] = 0
    E = torch.eye(4).expand(B, T, 4, 4).clone()
    K = torch.tensor([[R, 0, R / 2], [0, R, R / 2], [0, 0, 1.0]]).expand(B, T, 3, 3).clone()
    anchor = torch.randn(B, N, 3) * 0.2 + torch.tensor([0, 0, 3.0])
    return {
        "traj": traj, "anchor": anchor, "visibility": torch.rand(B, T, N) > 0.3,
        "query_t": torch.tensor([[0, 0, 0, 2, 5, 9]]), "gravity": torch.tensor([1.0]),
        "down": torch.tensor([[0.0, 1.0, 0.0]]), "down_known": torch.tensor([True]),
        "context": torch.randn(B, T, P, F).half(), "id_card": torch.randn(B, N, F).half(),
        "patch_xyz": torch.randn(B, T, P, 3) * 0.3 + torch.tensor([0, 0, 3.0]),
        "frames": torch.randint(0, 255, (B, T, R, R, 3), dtype=torch.uint8),
        "depth": torch.full((B, T, R // 4, R // 4), 3.0),
        "intrinsics": K, "intrinsics_256": K, "extrinsics": E, "hw": torch.tensor([[R, R]]),
        "norm_mean": torch.zeros(B, 3), "norm_scale": torch.ones(B), "norm_traj_scale": torch.full((B,), 0.1),
        "norm_offset": anchor[:, None].clone(),
        "centre": torch.randn(B, T, N, 3) * 0.1, "centre_known": torch.tensor([[True, True, True, False, True, False]]),
        "view": random_view(T, 2, 4)[None],
    }


def changed(b: dict, frames: slice) -> dict:
    """`b` with everything the images give in `frames` replaced."""
    b2 = {k: v.clone() for k, v in b.items()}
    b2["context"][:, frames] = torch.randn_like(b2["context"][:, frames])
    b2["frames"][:, frames] = torch.randint(0, 255, b2["frames"][:, frames].shape, dtype=torch.uint8)
    b2["depth"][:, frames] = 5.0
    b2["patch_xyz"][:, frames] += 1.0
    return b2


def masked_tracks(m, b: dict, sched: str, view: torch.Tensor) -> torch.Tensor:
    """The tracks with `view` (B, T, 2, 2) the quarters seen, the hidden ones blacked out."""
    b = to_device(b, torch.device("cpu"))
    b["frames"] = black_out(b["frames"], view)
    cond = dict(context=b["context"], visual_mask=view, id_card=b["id_card"], patch_xyz=b["patch_xyz"],
                **fine_inputs(m, b))
    return refine(m, b["anchor"], T, window=8, overlap=4, schedule=sched, **cond)[0]


def model(**kw):
    return PointDiT(dim=32, depth=1, num_heads=2, mlp_mult=2, feat_dim=F, cross_window=4, fine_neigh=4, **kw)


full = dict(visibility=True, group_feedback=4, rigid_proposal=True, follow_fit=True, feedback_scale=10.0,
            fit_robust=3, fit_fresh=30.0, patch=4, centre=True)
for kw, sched in ((dict(), "ff"), (full, "fbfb"), (full, "ffbb")):
    m = model(**kw)
    with torch.no_grad():             # off the zero inits, so every reader is live
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    s = types.SimpleNamespace(hidden_loss=1.0, vel_loss=1.0, rigid_loss=1.0,
                              group_loss=1.0 if kw else 0.0, vis_loss=1.0 if kw else 0.0,
                              accum=1, world=0, window=8, overlap=4, schedule=sched, down_hidden=1.0, down_loss=1.0,
                              centre_loss=1.0 if kw else 0.0)
    m.train()
    losses = learn(m, batch(), s, Hardware(torch.device("cpu"), False))
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    print(f"{sched}: loss {losses.method:.4f} vis {losses.visibility:.4f}, "
          f"params without gradient: {dead}, with a non-finite one: {len(bad)} {bad[:3]} (want 0)")
    j = ScoreSettings(device=torch.device("cpu"), window=8, overlap=4, schedule=sched, cutoff=6)
    b = batch()
    pred = Predictor(m, j)(b)
    b2 = changed(b, slice(6, None))
    pred2 = Predictor(m, j)(b2)
    leak = (pred.tracks - pred2.tracks).abs().max().item()
    print(f"  tracks {tuple(pred.tracks.shape)}, vis {None if pred.visibility is None else tuple(pred.visibility.shape)}, "
          f"forecast change when frames >= 6 change: {leak:.2e} (want 0)")
    scores = {**track_scores(pred, j), **visibility_scores(pred)}
    print(f"  {len(scores)} scores: {' '.join(sorted(scores))}")
    js = ScoreSettings(device=torch.device("cpu"), window=8, overlap=4, schedule=sched, cutoff=6, fill=N + 8, spawn_gap=0.02)
    spawn = Predictor(m, js)(b)
    print(f"  with points added: tracks {tuple(spawn.tracks.shape)} (want {tuple(pred.tracks.shape)}), "
          f"{len({**track_scores(spawn, js), **visibility_scores(spawn)})} scores")
    view = torch.ones(B, T, 2, 2, dtype=torch.bool)
    view[:, 5:8] = False
    gap = (masked_tracks(m, b, sched, view) - masked_tracks(m, changed(b, slice(5, 8)), sched, view)).abs().max().item()
    print(f"  change anywhere when the frames of a gap (5-7) change: {gap:.2e} (want 0)")
    view = torch.ones(B, T, 2, 2, dtype=torch.bool)
    view[:, 5:8, 0, 0] = False
    b2 = {k: v.clone() for k, v in b.items()}
    b2["frames"][:, 5:8, :R // 2, :R // 2] = torch.randint(0, 255, b2["frames"][:, 5:8, :R // 2, :R // 2].shape,
                                                           dtype=torch.uint8)
    b2["depth"][:, 5:8, :R // 8, :R // 8] = 5.0
    b2["context"][:, 5:8] = torch.randn_like(b2["context"][:, 5:8])      # no DINOv3 in a partial frame
    b2["id_card"][:, 4] = torch.randn_like(b2["id_card"][:, 4])          # its point, first seen at 5, is never handed over
    part = (masked_tracks(m, b, sched, view) - masked_tracks(m, b2, sched, view)).abs().max().item()
    seen = (masked_tracks(m, b, sched, view) - masked_tracks(m, changed(b, slice(5, 8)), sched, view)).abs().max().item()
    print(f"  top-left quarter of 5-7 hidden: change when it changes {part:.2e} (want 0), "
          f"when the rest of those frames does {seen:.2e} (want > 0)")

# The stored world: the same checks on the model that tracks a frame at a time.
from genpoint3d.train.model.WorldModel import WorldModel
from genpoint3d.train.tools.track import track

m = WorldModel(dim=32, depth=1, num_heads=2, mlp_mult=2, feat_dim=F, cross_window=4, fine_neigh=4, group_dim=4, patch=4, slots=4)
with torch.no_grad():
    for p in m.parameters():
        p.add_(torch.randn_like(p) * 0.05)
s = types.SimpleNamespace(hidden_loss=1.0, rigid_loss=1.0, group_loss=1.0, vis_loss=1.0, accum=1, world=1, rounds=2,
                          ahead_loss=1.0, down_hidden=1.0, down_loss=1.0, centre_loss=1.0)
m.train()
b = batch()
b["traj"][:, :, :3] = b["traj"][:, :, :1]                 # three points that move as one: a body
b["segment"] = torch.tensor([[0, 0, 0, 1, 1, 1]])
losses = learn(m, b, s, Hardware(torch.device("cpu"), False))
dead = [n for n, p in m.named_parameters() if (p.grad is None or p.grad.abs().sum() == 0)
        and not n.startswith(("pair", "touch_out", "what", "known", "of_body", "slot", "bar_out", "select_out", "imagined_in"))]   # the touches' and the bodies' weights: checked below, where there are bodies
bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
print(f"world: loss {losses.method:.4f} vis {losses.visibility:.4f}, params without gradient: {dead}, "
      f"with a non-finite one: {len(bad)} {bad[:3]} (want 0)")
j = ScoreSettings(device=torch.device("cpu"), rounds=2, cutoff=6)
pred, pred2 = Predictor(m, j)(b), Predictor(m, j)(changed(b, slice(6, None)))
print(f"  tracks {tuple(pred.tracks.shape)}, forecast change when frames >= 6 change: "
      f"{(pred.tracks - pred2.tracks).abs().max().item():.2e} (want 0)")
print(f"  {len({**track_scores(pred, j), **visibility_scores(pred)})} scores")
js = ScoreSettings(device=torch.device("cpu"), rounds=2, cutoff=6, fill=N + 8, spawn_gap=0.02)
print(f"  with points added: tracks {tuple(Predictor(m, js)(b).tracks.shape)} (want {tuple(pred.tracks.shape)})")
from genpoint3d.train.score.Adder import Adder

with torch.no_grad():
    m.select_out[-1].bias.fill_(10.0)          # the model says every cell needs a point: the adder's own rules decide
from genpoint3d.train.tools.track import Tracker

bd = to_device(b, torch.device("cpu"))
whole = torch.ones(B, T, 2, 2, dtype=torch.bool)
fi = fine_inputs(m, bd)
run = Tracker(m, bd["anchor"], T, 2, dict(context=bd["context"], visual_mask=whole, id_card=bd["id_card"],
                                          patch_xyz=bd["patch_xyz"], **fi), Adder(m, bd, fi, whole, N + 8))
with torch.no_grad():
    for _ in run:
        pass
print(f"  points added as the clip goes: {run.anchor.shape[1] - N} (want 8), first given in frames "
      f"{sorted(set(run.cond['query_t'][0, N:].tolist()))}, tracks {tuple(run.tracks.shape)}")


def world_tracks(b: dict, view: torch.Tensor) -> torch.Tensor:
    b = to_device(b, torch.device("cpu"))
    b["frames"] = black_out(b["frames"], view)
    return track(m, b["anchor"], T, 2, context=b["context"], visual_mask=view, id_card=b["id_card"],
                 patch_xyz=b["patch_xyz"], **fine_inputs(m, b))[0]


from genpoint3d.train.tools.world import World

m.eval()
view = torch.ones(B, T, 2, 2, dtype=torch.bool)
view[:, 5:8] = False
gap = (world_tracks(b, view) - world_tracks(changed(b, slice(5, 8)), view)).abs().max().item()
# A frame once seen is not gone back over: the past changes later only with a body, as a point's frames from
# before it was given, or as the frames a body was not seen in, put right when it is seen again (in either run).
redone, put_right = [], World._put_right
def spy(self, t, turn, centre, backed):
    for k in torch.nonzero(backed & (self.backed < t - 1) & (self.first < t - 1))[:, 0].tolist():
        redone.append((self.body == k, int(self.backed[k]) + 1, t))
    return put_right(self, t, turn, centre, backed)
World._put_right = spy
late = (world_tracks(b, view) - world_tracks(changed(b, slice(9, None)), view)).abs().amax(-1)[0, :9]   # (9, N)
World._put_right = put_right
by_body = torch.zeros(9, N, dtype=torch.bool)
for mine, a, z in redone:
    by_body[a:min(z, 9), mine[:N]] = True
since = (torch.arange(9)[:, None] >= b["query_t"][0][None]) & ~by_body
print(f"  change anywhere when the frames of a gap (5-7) change: {gap:.2e} (want 0); when the frames from 9 on "
      f"change, in frames 0-8: {late[since].max().item():.2e} from when a point is given (want 0), "
      f"{late[~since].max().item():.2e} with its body (before it is given, or its body put right when seen again)")

# What the model says of two bodies that touch reaches the engine's next frame: a ball on a floor, made up.
from genpoint3d.train.tools.world import World

floor = torch.cat([torch.rand(20, 1) - 0.5, torch.zeros(20, 1), torch.rand(20, 1) + 3], -1)
ball = floor[:10] + torch.tensor([0.0, 0.05, 0.0]) + torch.randn(10, 3) * 0.02
pos = torch.cat([floor, ball])
world = World(pos, torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.bool), 3)
groups = torch.cat([torch.zeros(20, 4), torch.full((10, 4), 3.0)])
world.store(0, pos, torch.ones(30), groups, pos)
n, k = world.touching(0, torch.full((30,), 0.2))
at, slide = world.sliding(0, n, k)
h = torch.randn(30, 32)
m.zero_grad()
fall = torch.tensor([0.0, -0.01, 0.0])
said = m.touches(h, n, k, world.body[n], world.body[k], pos[k] - pos[n], slide, at, fall)
world.ahead(0, fall, torch.tensor([0.0, 1.0]), torch.ones(2), said).pow(2).sum().backward()
dead = [name for name, p in m.named_parameters() if name.startswith(("pair", "touch_out")) and (p.grad is None or p.grad.abs().sum() == 0)]
print(f"  {len(n)} pairs of points where the two may touch, one touch said; its weights without gradient: {dead} (want none)")

# A body imagines the whole of its points: every point one body (the group vectors made alike), its object's
# answer four corners; one training step reaches the weights that imagine, and the world then has the four.
from genpoint3d.train.tools.imagine import Answer
from genpoint3d.train.tools.track import track_loss

corners = torch.randn(4, 3) * 0.2 + torch.tensor([0, 0, 3.0])
one = Answer(corners[None], torch.ones(4, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool), torch.ones(3, dtype=torch.long), torch.zeros(1, 7), torch.ones(1, dtype=torch.bool), torch.tensor([[0, 1], [1, 2]]),
             torch.eye(3).expand(2, 1, 3, 3), torch.zeros(2, 1, 3), torch.ones(1, 4, 4, dtype=torch.long))
hit = one.match(0, 0, torch.cat([corners[[2, 0, 3, 1]] + 0.01, torch.full((2, 3), 9.0)]))
print(f"imagine: six imagined points, four at the answer's corners shuffled: kept {hit[0].tolist()} as corners {hit[1].tolist()} "
      f"(want [0, 1, 2, 3] as [2, 0, 3, 1])")
print(f"  a second body on the same object, a piece of it: {len(one.match(1, 0, corners + 0.01)[0])} true points (want 0); "
      f"corners joined, 1-0, 2-1, 0-2: {one.joined(torch.tensor([1, 2, 0]), torch.tensor([0, 1, 2])).tolist()} (want [True, True, False])")
from genpoint3d.tools.kubric_models import FLIP, floor, turns

o = {"camera_quaternions": torch.tensor([[0.9, 0.3, -0.2, 0.25]]).numpy(), "camera_positions": torch.tensor([[1.0, -6.0, 3.0]]).numpy()}
o["camera_quaternions"] /= (o["camera_quaternions"] ** 2).sum() ** 0.5
on_floor = torch.cat([torch.rand(50, 2).double() * torch.tensor([4.0, 3.0]).double() - 1, torch.zeros(50, 1).double()], -1).numpy()
turn0, at0 = turns(o["camera_quaternions"][0].astype("float64")), o["camera_positions"][0].astype("float64")
back = (floor(o, ((on_floor - at0) @ turn0) @ FLIP) @ FLIP) @ turn0.T + at0
print(f"  the floor's corners from points on it: height {abs(back[:, 2]).max():.1e} (want 0), x from {back[:, 0].min():.2f} to "
      f"{back[:, 0].max():.2f} (want {on_floor[:, 0].min():.2f} to {on_floor[:, 0].max():.2f})")
m = WorldModel(dim=32, depth=1, num_heads=2, mlp_mult=2, feat_dim=F, cross_window=4, fine_neigh=4, group_dim=4, patch=4, slots=6)
with torch.no_grad():
    for p in m.parameters():
        p.add_(torch.randn_like(p) * 0.05)
    m.group_out.weight.zero_()
b = batch()
b["segment"] = torch.ones(B, N, dtype=torch.long)
b["mesh"] = (corners[None] + torch.randn(T, 1, 3) * 0.05)[None]
b["mesh_body"], b["mesh_seen"], b["things"] = torch.ones(B, 4, dtype=torch.long), torch.rand(B, T, 4) > 0.5, torch.randn(B, 1, 7)
b["moves"] = torch.ones(B, 1, dtype=torch.bool)
b["mesh_bars"] = torch.tensor([[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]])           # four corners, every two joined
b["turns"], b["places"] = torch.eye(3).expand(B, 2, T, 3, 3).clone(), torch.zeros(B, 2, T, 3)     # the floor and one still object
b["cell_object"] = torch.ones(B, T, R // 4, R // 4, dtype=torch.long)
s.points = N + 8
s.imagine, s.imagine_loss, s.known_loss = 6, 1.0, 1.0
m.train()
losses = learn(m, b, s, Hardware(torch.device("cpu"), False))
dead = [n for n, p in m.named_parameters() if n.startswith(("slot", "bar_out", "select_out", "imagined_in", "what", "known", "of_body")) and (p.grad is None or p.grad.abs().sum() == 0)]
bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
print(f"  loss {losses.method:.4f}; the imagining weights without gradient: {dead} (want none), with a non-finite one: {len(bad)} (want 0)")
bd = to_device(b, torch.device("cpu"))
fi = fine_inputs(m, bd)
answer = Answer.of(bd)
with torch.no_grad():
    out = track_loss(m, bd["traj"], bd["anchor"], mask=bd["visibility"], rounds=2, answer=answer, imagine_weight=1.0, context=bd["context"],
                     visual_mask=whole, id_card=bd["id_card"], patch_xyz=bd["patch_xyz"], **fi)[1]
w = out["world"]
print(f"  the world has {int(w.imagined.sum())} imagined points (want 4) on body {w.body[w.imagined].tolist()}, "
      f"{len(w.bars)} bars among them (at most 6, each between two of them: "
      f"{bool(w.imagined[w.bars].all()) if len(w.bars) else True}); the clip's own tracks {tuple(out['tracks'].shape)} (want {(B, T, N, 3)})")

moving = Answer.of(bd)
moving.turns = torch.eye(3).expand(2, T, 3, 3).clone()
moving.places = torch.zeros(2, T, 3)
moving.places[1] = torch.arange(T)[:, None] * torch.tensor([0.1, 0.0, 0.0])                     # object 1 slides along x
path = moving.track(3, torch.tensor([0, 5]), torch.tensor([[0.0, 0.0, 3.0], [1.0, 1.0, 3.0]]))
print(f"  a point added in frame 3 on a sliding object: at frame 0 x {float(path[0, 0, 0]):.2f}, at 10 x {float(path[10, 0, 0]):.2f} "
      f"(want -0.30, 0.70)")
from genpoint3d.train.score.shape_scores import shape_scores

print("  the shapes' scores: " + ", ".join(f"{k} {float(v):.2f}" for k, v in shape_scores(bd, w, whole, m.known(out["what"])).items()))
m.eval()
run = Tracker(m, bd["anchor"], T, 2, dict(context=bd["context"], visual_mask=whole, id_card=bd["id_card"], patch_xyz=bd["patch_xyz"], **fi))
with torch.no_grad():
    for _ in run:
        pass
print(f"  with no answer, the points said to be there are kept: {int(run.world.imagined.sum())} of 6, tracks {tuple(run.tracks.shape)}")
