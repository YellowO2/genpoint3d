"""The object fit (`regress.object_places`) on made-up clips whose answer is known: one rigid object, turning
and thrown, its points' true places given as the guess (with tracking noise where said). Each line says what
it wants; a line that misses it ends the run with an error. Tiny, CPU.

Run:  python scripts/check_fit.py
"""

import torch

from genpoint3d.train.tools.kabsch import turn_matrix
from genpoint3d.train.tools.regress import object_places

T, N = 30, 60
FRAMES = torch.arange(T).float()
GRAVITY = torch.tensor([0.0, -9.81 / 144, 0.0])          # metres per frame squared
failed = []


def thrown(spin: float = 0.15, speed=(0.05, 0.1, 0.0), bounce: int = -1):
    """A 30 cm object's true points (T, N, 3) and its centre (T, 3): turning `spin` radians a frame, thrown at
    `speed` metres a frame under gravity (none for a still one), bouncing at frame `bounce`."""
    torch.manual_seed(0)
    shape = torch.randn(N, 3) * 0.3
    R = turn_matrix(FRAMES[:, None] * torch.tensor([0.0, 0.0, spin]))
    still = not any(speed) and not spin
    centre, p, v = torch.zeros(T, 3), torch.tensor([0.0, 3.0, 5.0]), torch.tensor(speed)
    for i in range(T):
        centre[i] = p
        v = v if still else v + GRAVITY
        if i == bounce:
            v = v * torch.tensor([1.0, -0.6, 1.0])
        p = p + v
    return (R[:, None] @ shape[None, :, :, None])[..., 0] + centre[:, None], centre


def fit(pos, q, seen, centre=None, fall=True):
    """Each point's place by its object's pose (T, N, 3), all points one object."""
    tips = None if centre is None else centre[:, None].expand(T, N, 3)[None]
    return object_places(pos[None], q[None], torch.zeros(1, N, 4), torch.ones(1, N, dtype=torch.bool), seen[None].clone(),
                         3, 30.0, tips, GRAVITY[None] if fall else None)[0][0]


def noisy(true, cm: float, seed: int):
    torch.manual_seed(seed)
    pos = true + cm / 100 * torch.randn_like(true)
    pos[0] = true[0]                                        # the query frame is given
    return pos


def want(name: str, cm: float, most: float) -> None:
    ok = cm <= most
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {cm:.1f} cm (want at most {most:g})")
    if not ok:
        failed.append(name)


def off(places, true, frames) -> float:
    return float((places - true).norm(dim=-1)[frames].mean() * 100)


zero = torch.zeros(N, dtype=torch.long)
true, centre = thrown()

print("a point asked late is placed before its query frame by its object")
q = zero.clone(); q[30:] = 20
seen = (FRAMES[:, None] >= q[None]).float()
pos = torch.where(seen[..., None] > 0, true, true[q, torch.arange(N)][None].expand(T, N, 3))
want("late points, frames 0-15", float((fit(pos, q, seen, centre) - true).norm(dim=-1)[:16, 30:].mean() * 100), 0.1)

print("an object whose points come two a frame still gets its pose")
q = (torch.arange(N) // 2).clamp(max=T - 1)
seen = (FRAMES[:, None] >= q[None]).float()
pos = torch.where(seen[..., None] > 0, true, true[q, torch.arange(N)][None].expand(T, N, 3))
places = fit(pos, q, seen, centre)
want("points before their query frame", float((places - true).norm(dim=-1)[seen == 0].mean() * 100), 0.5)

print("a throw seen 3 frames, carried on")
seen3 = (FRAMES[:, None] < 3).float().expand(T, N)
want("exact points, 12 frames on", off(fit(true, zero, seen3, centre), true, 14), 0.1)
e = sum(off(fit(noisy(true, 1.5, s), zero, seen3, centre), true, 14) for s in range(5)) / 5
want("1.5 cm of tracking noise, 12 frames on", e, 15)

print("a still object under tracking noise holds")
still, _ = thrown(spin=0.0, speed=(0.0, 0.0, 0.0))
e = sum(off(fit(noisy(still, 1.5, s), zero, seen3), still, 14) for s in range(5)) / 5
want("12 frames on", e, 1)

print("an object unseen in frames 8-17 and seen again, 1.5 cm of tracking noise")
gap = ((FRAMES < 8) | (FRAMES >= 18))[:, None].float().expand(T, N)
for name, clip in (("free flight", (true, centre)), ("a bounce at frame 12, unseen", thrown(bounce=12))):
    e = sum(off(fit(noisy(clip[0], 1.5, s), zero, gap, clip[1]), clip[0], slice(8, 18)) for s in range(5)) / 5
    want(name + ", in the gap", e, 5)

if failed:
    raise SystemExit(f"{len(failed)} failed: {', '.join(failed)}")
print("all as wanted")
