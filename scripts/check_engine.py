"""The engine's fixed parts (`tools/engine.py`) on made-up bodies whose answer is known. Each line says what
it wants; a line that misses it ends the run with an error. Tiny, CPU.

Run:  python scripts/check_engine.py
"""

import torch

from genpoint3d.train.tools.engine import Body, Touch, fitted, mass_spread, pulled, step
from genpoint3d.train.tools.kabsch import turn_matrix

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
GRAVITY = torch.tensor([0.0, -9.81 / 144, 0.0])      # metres a frame squared, y up
NONE = torch.zeros(3)
failed = []


def want(name: str, ok: bool, said: str) -> None:
    print(f"  {'ok  ' if ok else 'MISS'} {name}: {said}")
    if not ok:
        failed.append(name)


def ball(n: int = 200, radius: float = 0.3) -> torch.Tensor:
    p = torch.randn(n, 3)
    p = radius * p / p.norm(dim=-1, keepdim=True)
    return p - p.mean(0)


def bodies(*each) -> Body:
    """Bodies from (centre, speed, spin) each, unturned."""
    stack = lambda i: torch.stack([torch.as_tensor(e[i], dtype=torch.float64) for e in each])
    return Body(stack(0), torch.eye(3).expand(len(each), 3, 3).clone(), stack(1), stack(2))


UP = (0.0, 1.0, 0.0)


def touch(a, b, at, hold=0.0, grip=0.0, push=(0.0, 0.0, 0.0), into=UP) -> Touch:
    return Touch(torch.tensor([a]), torch.tensor([b]), torch.tensor([at]), torch.tensor([into]), torch.tensor([hold]),
                 torch.tensor([grip]), torch.tensor([push]))


shape = ball()
spread = mass_spread(shape, torch.ones(len(shape)))
FLOOR = ((0.0, 0.0, 5.0), NONE, NONE)                 # the fixed world, body 0 where it is in
two = lambda: dict(free=torch.tensor([0.0, 1.0]), heavy=torch.ones(2), spread=spread.expand(2, 3, 3))
BOTTOM = (0.0, 0.0, 5.0)                              # under a ball centred at (0, 0.3, 5)

# Thrown and spinning, nothing touching it: the centre falls as gravity says, the spin and the shape hold.
b = start = bodies(((0.0, 3.0, 5.0), (0.05, 0.1, 0.0), (0.0, 0.0, 0.15)))
first = b.points(shape)
for _ in range(20):
    b = step(b, GRAVITY, torch.ones(1))
true = start.centre + 20 * start.speed + GRAVITY * 20 * 21 / 2
want("a thrown body's centre falls as gravity says", (b.centre - true).norm() < 1e-9, f"{(b.centre - true).norm() * 100:.2e} cm off after 20 frames")
want("it keeps its spin", (b.spin - start.spin).norm() < 1e-12, f"{float(b.spin[0, 2]):.3f} radians a frame")
gaps = lambda p: (p[:, :, None] - p[:, None]).norm(dim=-1)
want("it keeps its shape", (gaps(b.points(shape)) - gaps(first)).abs().max() < 1e-9, f"{(gaps(b.points(shape)) - gaps(first)).abs().max() * 100:.2e} cm")

# The fixed world does not fall.
b = step(bodies(FLOOR), GRAVITY, torch.zeros(1))
want("the fixed world stays", b.speed.norm() == 0 and (b.centre - torch.tensor(FLOOR[0])).norm() == 0, "no move under gravity")

# A push through the centre moves the body and does not turn it.
rest = bodies(FLOOR, ((0.0, 0.3, 5.0), NONE, NONE))
after = step(rest, NONE, touch=touch(1, 0, BOTTOM, push=(0.0, 0.02, 0.0)), **two())
want("a push through the centre only moves it", after.spin.norm() < 1e-12 and (after.speed[1] - torch.tensor([0.0, 0.02, 0.0])).norm() < 1e-12,
     f"speed {after.speed[1].tolist()}, spin {float(after.spin.norm()):.1e}")

# A ball on the floor pushed sideways at its bottom (friction, toward -x): it slows toward -x and
# rolls forward, its top going toward +x.
after = step(rest, NONE, touch=touch(1, 0, BOTTOM, push=(-0.02, 0.0, 0.0)), **two())
top = torch.linalg.cross(after.spin[1], torch.tensor([0.0, 0.3, 0.0]))
want("a sideways push at the bottom moves it and spins it", after.speed[1, 0] < 0 and top[0] > 0,
     f"speed x {float(after.speed[1, 0]):.3f}, its top's x {float(top[0]):+.4f} a frame")

# One push between two free bodies, of mass 1 and 2: what one gains the other loses.
pair = bodies(((0.0, 0.3, 5.0), NONE, NONE), ((0.6, 0.3, 5.0), NONE, NONE))
heavy = torch.tensor([1.0, 2.0])
after = step(pair, NONE, torch.ones(2), heavy, spread.expand(2, 3, 3), touch(0, 1, (0.3, 0.3, 5.0), push=(0.03, 0.01, 0.0)))
total = (heavy[:, None] * after.speed).sum(0)
want("a push between two bodies keeps the pair's momentum", total.norm() < 1e-12 and after.speed[0].norm() > 0,
     f"{float(total.norm()):.1e}, the lighter going {float(after.speed[0].norm() / after.speed[1].norm()):.1f} times as fast")

# A ball at rest on the floor, held and gripped where it touches: gravity does not move it.
b = rest
for _ in range(20):
    b = step(b, GRAVITY, touch=touch(1, 0, BOTTOM, hold=1.0, grip=1.0), **two())
want("a ball the floor holds and grips stays at rest under gravity", (b.centre[1] - rest.centre[1]).norm() < 1e-8,
     f"{(b.centre[1] - rest.centre[1]).norm() * 100:.2e} cm after 20 frames")

# A ball sliding along the floor, held and gripped where it touches: it ends rolling, its bottom still on the floor.
slide = bodies(FLOOR, ((0.0, 0.3, 5.0), (0.1, 0.0, 0.0), NONE))
after = step(slide, GRAVITY, touch=touch(1, 0, BOTTOM, hold=1.0, grip=1.0), **two())
bottom = after.speed[1] + torch.linalg.cross(after.spin[1], torch.tensor([0.0, -0.3, 0.0]))
want("a sliding ball that is gripped ends rolling", bottom.norm() < 1e-9 and after.speed[1, 0] > 0.03 and after.spin[1, 2] < 0,
     f"its bottom's speed {float(bottom.norm()):.1e}, its centre's {float(after.speed[1, 0]):.3f}")

# The same ball held by the floor but not gripped (ice): it does not sink, and slides on as it was, unturned.
after = step(slide, GRAVITY, touch=touch(1, 0, BOTTOM, hold=1.0), **two())
want("a ball held but not gripped slides on", abs(float(after.speed[1, 1])) < 1e-9 and abs(float(after.speed[1, 0]) - 0.1) < 1e-9
     and after.spin.norm() < 1e-9, f"speed {[round(v, 4) for v in after.speed[1].tolist()]}, spin {float(after.spin.norm()):.1e}")

# Two bodies falling together, stuck where they touch: nothing pulls them apart, so no push.
fall = bodies(((0.0, 3.0, 5.0), NONE, NONE), ((0.6, 3.0, 5.0), NONE, NONE))
after = step(fall, GRAVITY, torch.ones(2), heavy, spread.expand(2, 3, 3), touch(0, 1, (0.3, 3.0, 5.0), hold=1.0, grip=1.0, into=(1.0, 0.0, 0.0)))
want("two bodies falling as one keep falling", (after.speed - GRAVITY).norm() < 1e-12, f"both at {after.speed[0].tolist()}")

# The pose that keeps the shape: the shape turned and moved, its points tracked with 1 cm of noise.
R = turn_matrix(torch.tensor([0.3, -0.5, 0.2]))
true = shape @ R.T + torch.tensor([1.0, 0.5, 4.0])
noisy = true + 0.01 * torch.randn_like(true)
place = Body(*reversed(fitted(shape, noisy, torch.ones(len(shape)))), None, None).points(shape)
off, raw = (place - true).norm(dim=-1).mean() * 100, (noisy - true).norm(dim=-1).mean() * 100
want("the place that keeps the shape is nearer the truth than the noisy points", off < raw / 4, f"{off:.2f} cm against the points' {raw:.2f} cm")
# ... and where it sits is said by the points that count: half the points wrong by 20 cm, given no weight.
sure = torch.arange(len(shape)) % 2 == 0
wrong = torch.where(sure[:, None], noisy, true + 0.2 * torch.randn_like(true))
place = Body(*reversed(fitted(shape, wrong, sure.double())), None, None).points(shape)
off = (place - true).norm(dim=-1)[~sure].mean() * 100
want("points that do not count are placed by those that do", off < 0.5, f"{off:.2f} cm for the wrong half")
# ... and a few wrong points that do count do not drag it.
few = torch.arange(len(shape)) % 10 == 0
wrong = torch.where(few[:, None], true + 0.3 * torch.randn_like(true), noisy)
off = (Body(*reversed(fitted(shape, wrong, torch.ones(len(shape)))), None, None).points(shape) - true).norm(dim=-1).mean() * 100
want("a few wrong points do not drag it", off < raw / 2, f"{off:.2f} cm with a tenth of the points 30 cm off")
half = pulled(noisy, place, torch.full((len(shape),), 0.5))
want("no stiffness, no pull; full stiffness, the rigid place",
     torch.equal(pulled(noisy, place, torch.zeros(len(shape))), noisy)
     and torch.allclose(pulled(noisy, place, torch.ones(len(shape))), place)
     and torch.allclose(half, (noisy + place) / 2), "0, 0.5 and 1")

# Where the points went teaches the model's part: a touch nobody is told, found from the points a frame on.
land = bodies(FLOOR, ((0.0, 0.3, 5.0), (0.02, -0.1, 0.0), NONE))
at = (0.05, 0.0, 5.0)
seen = step(land, GRAVITY, touch=touch(1, 0, at, hold=0.9, grip=0.4, push=(-0.01, 0.0, 0.004)), **two()).points(shape)[1]
hold, grip, push = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True), torch.zeros(1, 3, requires_grad=True)
opt = torch.optim.Adam([hold, grip, push], lr=0.02)
said = lambda: Touch(torch.tensor([1]), torch.tensor([0]), torch.tensor([at]), torch.tensor([UP]), torch.sigmoid(hold),
                     torch.sigmoid(grip), push)
for _ in range(4000):
    opt.zero_grad()
    (step(land, GRAVITY, touch=said(), **two()).points(shape)[1] - seen).pow(2).sum().backward()
    opt.step()
err = (step(land, GRAVITY, touch=said(), **two()).points(shape)[1] - seen).norm(dim=-1).max() * 100
want("the touch is learned from where the points went", err < 0.01,
     f"the points a frame on {err:.1e} cm off, never told the hold, the grip or the push; no turn's gradient is lost at no spin")

if failed:
    raise SystemExit(f"missed: {failed}")
print("all as wanted")
