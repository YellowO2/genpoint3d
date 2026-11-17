"""Adding points (`add_points`) on a made-up wall, each case with the answer it wants. Tiny, CPU."""

import torch

from genpoint3d.train.tools.farthest_first import GAP, UNIT, add_points, spacing

F, W, CELLS = 560.0, 512, 128
cell = (torch.arange(CELLS) + 0.5) / CELLS * W
PX = torch.stack(torch.meshgrid(cell, cell, indexing="xy"), -1).reshape(-1, 2)
failed = []


def want(name: str, ok: bool, got: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got}")
    if not ok:
        failed.append(name)


def added(cameras: list[tuple[float, float]], room: int = 10 ** 6) -> list[int]:
    """Points added per frame on a wall facing a camera at (x, distance) in each frame."""
    world = {}

    def candidates(t, born):
        x, z = cameras[t]
        return PX, torch.arange(len(PX)), spacing(torch.full((len(PX),), z), F, W, GAP)

    def covering(t, born):
        x, z = cameras[t]
        for t0, i in born:
            x0, z0 = cameras[t0]
            world.setdefault((t0, i), (PX[i] - W / 2) * z0 / F + torch.tensor([x0, 0.0]))
        if not born:
            return PX.new_zeros(0, 2)
        px = (torch.stack([world[k] for k in born]) - torch.tensor([x, 0.0])) * F / z + W / 2
        return px[((px >= 0) & (px < W)).all(-1)]

    born = add_points(list(range(len(cameras))), room, candidates, covering)
    return [sum(t0 == t for t0, _ in born) for t in range(len(cameras))]


print("a wall 8 m away, the camera still")
n = added([(0.0, 8.0)] * 4)
want("the first frame is covered at the image spacing", 250 <= n[0] <= 600, f"{n[0]} points (a {1 / GAP:.0f} x {1 / GAP:.0f} grid is {int(1 / GAP) ** 2})")
want("nothing more is added to covered surface", sum(n[1:]) == 0, f"{n[1:]}")

print("the camera pans along the wall, half a metre a frame")
n = added([(0.5 * t, 8.0) for t in range(12)])
want("new surface gets points in every frame", min(n[1:]) > 0, f"{n}")

print("the same pan, with room for only as many points as the first frame takes")
n = added([(0.5 * t, 8.0) for t in range(12)], room=n[0])
want("the limit is the only thing that stops it", sum(n[1:]) == 0, f"{n}")

print("the camera comes closer to a wall already covered at the smallest piece")
near = UNIT * F / (GAP * W)
n = added([(0.0, near), (0.0, near / 2), (0.0, near / 4)])
want(f"next to nothing is added from {near:.2f} m in (cells are 4 px, so a few slip through)", sum(n[1:]) <= 0.05 * n[0], f"{n}")

print("the camera comes closer to a wall covered only coarsely")
n = added([(0.0, 8.0), (0.0, 4.0)])
want("the finer pieces now told apart get points", n[1] > 0, f"{n}")

if failed:
    raise SystemExit(f"{len(failed)} failed: {', '.join(failed)}")
print("all as wanted")
