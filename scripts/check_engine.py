"""The engine's fixed parts (`tools/engine.py`) on made-up bodies whose answer is known. Each line says what
it wants; a line that misses it ends the run with an error. Tiny, CPU.

Run:  python scripts/check_engine.py
"""

import torch

from genpoint3d.train.tools.engine import Body, mass_spread, pulled, rigid_place, step
from genpoint3d.train.tools.kabsch import turn_matrix

torch.manual_seed(0)
GRAVITY = torch.tensor([0.0, -9.81 / 144, 0.0], dtype=torch.float64)      # metres a frame squared, y up
NONE = torch.zeros(3, dtype=torch.float64)
failed = []


def want(name: str, ok: bool, said: str) -> None:
    print(f"  {'ok  ' if ok else 'MISS'} {name}: {said}")
    if not ok:
        failed.append(name)


def ball(n: int = 200, radius: float = 0.3) -> torch.Tensor:
    p = torch.randn(n, 3, dtype=torch.float64)
    p = radius * p / p.norm(dim=-1, keepdim=True)
    return p - p.mean(0)


def still(centre=(0.0, 0.3, 5.0)) -> Body:
    return Body(torch.tensor(centre, dtype=torch.float64), torch.eye(3, dtype=torch.float64), NONE.clone(), NONE.clone())


shape = ball()
spread = mass_spread(shape)

# Thrown and spinning, nothing touching it: the centre falls as gravity says, the spin and the shape hold.
b = Body(torch.tensor([0.0, 3.0, 5.0], dtype=torch.float64), torch.eye(3, dtype=torch.float64),
         torch.tensor([0.05, 0.1, 0.0], dtype=torch.float64), torch.tensor([0.0, 0.0, 0.15], dtype=torch.float64))
start, first = b, b.points(shape)
for _ in range(20):
    b = step(b, GRAVITY, spread)
true = start.centre + 20 * start.speed + GRAVITY * 20 * 21 / 2
want("a thrown body's centre falls as gravity says", (b.centre - true).norm() < 1e-9, f"{(b.centre - true).norm() * 100:.2e} cm off after 20 frames")
want("it keeps its spin", (b.spin - start.spin).norm() < 1e-12, f"{float(b.spin[2]):.3f} radians a frame")
gaps = lambda p: torch.cdist(p, p)
want("it keeps its shape", (gaps(b.points(shape)) - gaps(first)).abs().max() < 1e-6, f"{(gaps(b.points(shape)) - gaps(first)).abs().max() * 100:.2e} cm")

# A push through the centre moves the body and does not turn it.
b = still()
push = torch.tensor([[0.0, 0.02, 0.0]], dtype=torch.float64)
after = step(b, NONE, spread, push, b.centre[None] + torch.tensor([[0.0, -0.3, 0.0]], dtype=torch.float64))
want("a push through the centre only moves it", after.spin.norm() < 1e-12 and (after.speed - push[0]).norm() < 1e-12,
     f"speed {after.speed.tolist()}, spin {float(after.spin.norm()):.1e}")

# A ball on the floor pushed sideways at its bottom (friction, toward -x): it slows toward -x and
# rolls forward, its top going toward +x.
side = torch.tensor([[-0.02, 0.0, 0.0]], dtype=torch.float64)
after = step(b, NONE, spread, side, b.centre[None] + torch.tensor([[0.0, -0.3, 0.0]], dtype=torch.float64))
top = torch.linalg.cross(after.spin, torch.tensor([0.0, 0.3, 0.0], dtype=torch.float64))
want("a sideways push at the bottom moves it and spins it", after.speed[0] < 0 and top[0] > 0,
     f"speed x {float(after.speed[0]):.3f}, its top's x {float(top[0]):+.4f} a frame")

# The same force, opposite, on two bodies that touch: what one gains the other loses.
a, c = still((0.0, 0.3, 5.0)), still((0.6, 0.3, 5.0))
touch, f = torch.tensor([[0.3, 0.3, 5.0]], dtype=torch.float64), torch.tensor([[0.03, 0.01, 0.0]], dtype=torch.float64)
a2, c2 = step(a, NONE, spread, -f, touch, mass=1.0), step(c, NONE, spread, f, touch, mass=2.0)
total = 1.0 * a2.speed + 2.0 * c2.speed
want("equal and opposite forces keep the pair's momentum", total.norm() < 1e-12, f"{float(total.norm()):.1e}")

# The rigid place: the shape turned and moved, its points tracked with 1 cm of noise.
R = turn_matrix(torch.tensor([0.3, -0.5, 0.2], dtype=torch.float64))
true = shape @ R.T + torch.tensor([1.0, 0.5, 4.0], dtype=torch.float64)
noisy = true + 0.01 * torch.randn_like(true)
place = rigid_place(shape, noisy, torch.ones(len(shape), dtype=torch.float64))
off, raw = (place - true).norm(dim=-1).mean() * 100, (noisy - true).norm(dim=-1).mean() * 100
want("the rigid place is nearer the truth than the noisy points", off < raw / 4, f"{off:.2f} cm against the points' {raw:.2f} cm")
# ... and where it sits is said by the points that count: half the points wrong by 20 cm, given no weight.
sure = torch.arange(len(shape)) % 2 == 0
wrong = torch.where(sure[:, None], noisy, true + 0.2 * torch.randn_like(true))
place = rigid_place(shape, wrong, sure.double())
off = (place - true).norm(dim=-1)[~sure].mean() * 100
want("points that do not count are placed by those that do", off < 0.5, f"{off:.2f} cm for the wrong half")
half = pulled(noisy, place, torch.full((len(shape),), 0.5, dtype=torch.float64))
want("no stiffness, no pull; full stiffness, the rigid place",
     torch.equal(pulled(noisy, place, torch.zeros(len(shape), dtype=torch.float64)), noisy)
     and torch.allclose(pulled(noisy, place, torch.ones(len(shape), dtype=torch.float64)), place)
     and torch.allclose(half, (noisy + place) / 2), "0, 0.5 and 1")

# Where the points went teaches the force: a force nobody is told, found from the points a frame on.
b = Body(torch.tensor([0.0, 0.3, 5.0], dtype=torch.float64), torch.eye(3, dtype=torch.float64),
         torch.tensor([0.02, -0.1, 0.0], dtype=torch.float64), NONE.clone())
at = b.centre[None] + torch.tensor([[0.05, -0.3, 0.0]], dtype=torch.float64)
secret = torch.tensor([[-0.01, 0.18, 0.004]], dtype=torch.float64)
seen = step(b, GRAVITY, spread, secret, at).points(shape)
guess = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([guess], lr=0.01)
for _ in range(600):
    opt.zero_grad()
    (step(b, GRAVITY, spread, guess, at).points(shape) - seen).pow(2).sum().backward()
    opt.step()
err = float((guess.detach() - secret).norm() / secret.norm())
want("the force is learned from where the points went", err < 0.01, f"{err:.2%} off, never told the force")

if failed:
    raise SystemExit(f"missed: {failed}")
print("all as wanted")
