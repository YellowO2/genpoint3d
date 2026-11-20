"""The stored world (`tools/world.py`) on a made-up clip whose answer is known: a ball thrown and spinning over
a still floor, the truth handed over as what is seen. Each line says what it wants; a line that misses it ends
the run with an error. Tiny, CPU.

Run:  python scripts/check_world.py
"""

import torch

from genpoint3d.train.tools.engine import Body, Touch, step
from genpoint3d.train.tools.world import World

torch.manual_seed(0)
T, GRAVITY = 24, torch.tensor([0.0, -9.81 / 144, 0.0])     # metres a frame squared, y up
failed = []


def want(name: str, ok: bool, said: str) -> None:
    print(f"  {'ok  ' if ok else 'MISS'} {name}: {said}")
    if not ok:
        failed.append(name)


def clip(speed=(0.03, 0.25, 0.0), spin=(0.0, 0.0, 0.2), height=1.0):
    """The true places (T, N, 3), each point's body (0 the floor, 1 the ball) and the ball's true centre (T, 3)."""
    ball = torch.randn(60, 3)
    ball = 0.3 * ball / ball.norm(dim=-1, keepdim=True)
    ball = ball - ball.mean(0)
    floor = torch.cat([torch.rand(60, 1) * 4 - 2, torch.zeros(60, 1), torch.rand(60, 1) * 4 + 3], -1)
    b = Body(torch.tensor([[0.0, height, 5.0]]), torch.eye(3)[None], torch.tensor([speed]), torch.tensor([spin]))
    frames, centres = [], []
    for _ in range(T):
        frames.append(torch.cat([floor, b.points(ball[None])[0]]))
        centres.append(b.centre[0])
        b = step(b, GRAVITY, torch.ones(1))
    return torch.stack(frames), torch.cat([torch.zeros(60, dtype=torch.long), torch.ones(60, dtype=torch.long)]), torch.stack(centres)


def watch(true, label, centres, query, hidden=(), gravity=GRAVITY, touch=None):
    """The world after the clip: each frame, the true places of the points seen (the ball's are not in the
    frames `hidden`), the true bodies as group vectors, arrows to the true centres; then the engine's next frame."""
    N = true.shape[1]
    world = World(true[query, torch.arange(N)], query, torch.ones(N, dtype=torch.bool), T)
    groups = torch.nn.functional.one_hot(label, 2).float() * 5
    for t in range(T):
        seen = ((label == 0) | (t not in hidden)).float() * world.alive(t)
        pos = torch.where(seen[:, None] > 0, true[t], world.pos[t])
        tips = torch.where(label[:, None] == 1, centres[t], true[t].mean(0))
        world.store(t, pos, seen, groups, tips)
        if t + 1 < T:
            free = (label[world.seed] == 1).float()
            said = None if touch is None else touch(world, t)
            world.ahead(t, gravity, free, torch.ones_like(free), said)
    return world


cm = lambda a, b: float((a - b).norm(dim=-1).max()) * 100
true, label, centres = clip()
N = true.shape[1]
ball = label == 1
every = torch.zeros(N, dtype=torch.long)

world = watch(true, label, centres, every)
want("two bodies are found", len(world.seed) == 2 and (world.body == world.body[0])[~ball].all() and (world.body == world.body[-1])[ball].all(),
     f"{len(world.seed)} bodies")
want("the world holds the clip as it was seen", cm(world.pos, true) < 1e-2, f"{cm(world.pos, true):.1e} cm off at most")
k = int(world.body[-1])
want("the ball's centre of gravity is where its points' arrows point", cm(world.centre[:, k], centres) < 1e-2,
     f"{cm(world.centre[:, k], centres):.1e} cm from the true centre, every frame")

# The ball unseen in frames 6 to 13: the engine carries it, thrown and spinning, from how it went when last seen.
hidden = range(6, 14)
world = watch(true, label, centres, every, hidden)
want("a body unseen is carried by the engine", cm(world.pos[6:14], true[6:14]) < 0.1, f"{cm(world.pos[6:14], true[6:14]):.1e} cm off in its unseen frames")

# A point given late: its frames before are the body's.
late = every.clone()
late[-10:] = 16
world = watch(true, label, centres, late, hidden)
want("a point given late has its past from its body", cm(world.pos[:16, -10:], true[:16, -10:]) < 0.1,
     f"{cm(world.pos[:16, -10:], true[:16, -10:]):.1e} cm off in the 16 frames before it was given")

# The engine wrong while the ball is unseen (told no gravity): seen again, the unseen frames are put right.
blind = World(true[0], every, torch.ones(N, dtype=torch.bool), T)
world = watch(true, label, centres, every, hidden, gravity=torch.zeros(3))
groups = torch.nn.functional.one_hot(label, 2).float() * 5
for t in range(14):
    seen = ((label == 0) | (t not in hidden)).float()
    blind.store(t, torch.where(seen[:, None] > 0, true[t], blind.pos[t]), seen, groups, torch.where(ball[:, None], centres[t], true[t].mean(0)))
    blind.ahead(t, torch.zeros(3), (label[blind.seed] == 1).float())
before, after = cm(blind.pos[6:14, ball], true[6:14, ball]), cm(world.pos[6:14, ball], true[6:14, ball])
want("a body seen again has its unseen frames put right", after < before / 3, f"{before:.0f} cm off before it is seen again, {after:.0f} cm after")

# A ball at rest on the floor, unseen from frame 4 on: it stays if the model says the floor holds and grips it, and falls through if not.
rest, label, centres = clip(speed=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0), height=0.3)
rest[:] = rest[0]
centres[:] = centres[0]
on_floor = lambda w, t: Touch(w.body[-1:], w.body[:1], centres[t][None] - torch.tensor([[0.0, 0.3, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]), torch.ones(1), torch.ones(1), torch.zeros(1, 3))
world = watch(rest, label, centres, every, range(4, T), touch=on_floor)
want("a ball the floor holds and grips stays on it unseen", cm(world.pos[4:], rest[4:]) < 0.1, f"{cm(world.pos[4:], rest[4:]):.1e} cm off after 20 frames unseen")
world = watch(rest, label, centres, every, range(4, T))
want("with nothing said, it falls through", cm(world.pos[4:], rest[4:]) > 100, f"{cm(world.pos[4:], rest[4:]):.0f} cm")
n, m = world.touching(3, torch.full((N,), 0.5))
want("the points where the two may touch are found", len(n) > 0 and (world.body[n] != world.body[m]).all()
     and bool((world.pos[3, n] - world.pos[3, m]).norm(dim=-1).max() < 0.5), f"{len(n)} pairs, each across the two bodies, within reach")

# The far half of the ball is never given: the body imagines it when first found. Imagined right, the world
# carries it with the body, thrown and spinning; imagined wrong and said right in frame 8, it is right from then on.
true, label, centres = clip()
far = torch.arange(N) >= N - 30
for name, wrong in (("right", 0.0), ("wrong", 0.2)):
    world = World(true[0, ~far], torch.zeros(N - 30, dtype=torch.long), torch.ones(N - 30, dtype=torch.bool), T)
    groups = torch.nn.functional.one_hot(label[~far], 2).float() * 5
    for t in range(T):
        many = len(world.body)
        pos = torch.cat([true[t, ~far], world.pos[t, N - 30:]])
        pos = torch.cat([true[t, ~far], world.rigid(t, pos, torch.ones(many))[0][N - 30:]])    # said to be where its body's shape puts it
        if wrong and t == 8:
            pos[N - 30:] = true[t, far]
        tips = torch.where(label[~far][:, None] == 1, centres[t], true[t, ~far].mean(0))
        world.store(t, pos, torch.ones(many), torch.cat([groups, torch.zeros(many - len(groups), 2)]),
                    torch.cat([tips, pos[N - 30:]]))
        if t + 1 < T:
            world.ahead(t, GRAVITY, (label[world.seed] == 1).float())
        if not t:
            world.imagine(int(world.body[N - 31]), true[0, far] + wrong, 0)
    if not wrong:
        want("a body's imagined points are carried with it", cm(world.pos[:, N - 30:], true[:, far]) < 0.1 and bool(world.imagined[N - 30:].all()),
             f"{cm(world.pos[:, N - 30:], true[:, far]):.1e} cm off over the clip, the ball's pose fitted to its seen half alone")
    else:
        before, after = cm(world.pos[1:8, N - 30:], true[1:8, far]), cm(world.pos[8:, N - 30:], true[8:, far])
        want("an imagined point said elsewhere is there from then on", before > 10 and after < 0.1,
             f"{before:.0f} cm off before it is said right, {after:.1e} cm after")

if failed:
    raise SystemExit(f"missed: {failed}")
print("all as wanted")
