"""Does a ready one-picture shape model (TripoSR) know the unseen side of Kubric's objects? Each object of a clip,
at the frame that shows most of it, is cut out as TripoSR wants it (its box, the rest grey), TripoSR says where in
its own box there is object, and that shape is laid on the object's true one (Kubric's fine model, placed by its
pose) by the best turn, size and shift: the best case, the truth is used for the laying. Then how far the true
surface is from TripoSR's, for the side facing the camera and for the side facing away, against imagining
nothing (the true facing side by itself, laid the same way). In shares of the object's longest side and in cells (8
across). No model of ours.

  python scripts/check_shape_prior.py --objects OBJECTS --frames RAW --assets DIR --triposr TRIPOSR --clips 40 --out DIR
  python scripts/check_shape_prior.py --made-up        the measure itself on shapes with a known answer
"""

import argparse
import itertools
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kubric_tracks import quat_to_rot  # noqa: E402

from genpoint3d.tools.kubric_models import FINE, shape  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0])  # Blender's axes to OpenCV's
CELLS = 8      # the target's detail: cells across an object's longest side
FILL = 0.85    # the share of the picture the object fills, as TripoSR's own preparing
GRID = 48      # places across TripoSR's box it is asked at
SOLID = 25.0   # TripoSR's own threshold: denser is object
POINTS = 1500  # places on a surface, for laying one on the other and for the distances
MANY = 30000   # places all over an object, to tell what is in front of what
SMALL = 200    # pixels: an object showing fewer is left out
TURNS = [torch.tensor(m, dtype=torch.float32) for m in
         (np.eye(3)[list(p)] * np.array(s)[:, None] for p in itertools.permutations(range(3)) for s in itertools.product([1, -1], repeat=3))
         if np.linalg.det(m) > 0]                                                       # the 24 turns of a cube, to start from


def surface(corners: np.ndarray, walls: list, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 3) places spread evenly over the walls."""
    a, b, c = (corners[np.array(walls)[:, k]] for k in range(3))
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    f = rng.choice(len(area), n, p=area / area.sum())
    u, v = rng.random((2, n, 1))
    u, v = np.where(u + v > 1, 1 - u, u), np.where(u + v > 1, 1 - v, v)
    return a[f] + u * (b[f] - a[f]) + v * (c[f] - a[f])


def facing(cam: np.ndarray, whole: np.ndarray, longest: float, bins: int = 32) -> np.ndarray:
    """(n,) whether each place (n, 3), in the camera's frame (z ahead), is on the side the camera sees: nothing of
    the object, `whole` (M, 3) many places all over it, in front of it by more than a thirtieth of its longest side."""
    uv, low = whole[:, :2] / whole[:, 2:], None
    low, wide = uv.min(0), np.ptp(uv, axis=0) + 1e-9
    key = lambda q: (lambda at: at[:, 0] * bins + at[:, 1])(np.clip(((q[:, :2] / q[:, 2:] - low) / wide * bins).astype(int), 0, bins - 1))  # noqa: E731
    nearest = np.full(bins * bins, np.inf)
    np.minimum.at(nearest, key(whole), np.linalg.norm(whole, axis=1))
    return np.linalg.norm(cam, axis=1) <= nearest[key(cam)] + longest / 30


def laid(src: torch.Tensor, dst: torch.Tensor, rounds: int = 30) -> tuple[torch.Tensor, torch.Tensor, int]:
    """`src` (P, 3) laid on `dst` (Q, 3) by the turn, size and shift that bring the two surfaces closest: the
    moved src, the turn (3, 3), and which of the 24 starts won."""
    best = None
    for k, turn in enumerate(TURNS):
        turn = turn.to(src)
        size = (dst.max(0).values - dst.min(0).values).norm() / (src.max(0).values - src.min(0).values).norm()
        shift = dst.mean(0) - size * src.mean(0) @ turn.T
        for _ in range(rounds):
            d = torch.cdist(size * src @ turn.T + shift, dst)
            x, y = torch.cat([src, src[d.argmin(0)]]), torch.cat([dst[d.argmin(1)], dst])          # each place with its nearest on the other
            mx, my = x.mean(0), y.mean(0)
            u, s, vt = torch.linalg.svd((y - my).T @ (x - mx) / len(x))
            sign = torch.ones(3, device=src.device)
            sign[2] = torch.sign(torch.linalg.det(u) * torch.linalg.det(vt))
            turn, size = (u * sign) @ vt, (s * sign).sum() / ((x - mx) ** 2).sum(1).mean()
            shift = my - size * mx @ turn.T
        moved = size * src @ turn.T + shift
        d = torch.cdist(moved, dst)
        off = float(d.min(1).values.mean() + d.min(0).values.mean())
        if best is None or off < best[0]:
            best = (off, moved, turn, k)
    return best[1], best[2], best[3]


def skin(solid: torch.Tensor) -> torch.Tensor:
    """(P, 3) the places, as cell numbers, of a solid block (G, G, G) that touch empty space."""
    pad = torch.nn.functional.pad(solid, (1, 1, 1, 1, 1, 1))
    inside = torch.stack([pad[2:, 1:-1, 1:-1], pad[:-2, 1:-1, 1:-1], pad[1:-1, 2:, 1:-1], pad[1:-1, :-2, 1:-1], pad[1:-1, 1:-1, 2:], pad[1:-1, 1:-1, :-2]]).all(0)
    return torch.nonzero(solid & ~inside).float()


def measure(true: np.ndarray, front: np.ndarray, said: torch.Tensor, longest: float) -> dict:
    """The distances, as shares of the longest side, between the true surface (n, 3), `front` its camera-facing
    places, and the said one (P, 3) once laid on it."""
    t = torch.from_numpy(true).float().to(said)
    moved, turn, start = laid(said, t)
    d, f = torch.cdist(t, moved), torch.from_numpy(front).to(said.device)
    alone = laid(t[f], t)[0]                                                                # the facing side by itself, laid the same way: nothing imagined
    return {"front": float(d[f].min(1).values.mean()) / longest, "back": float(d[~f].min(1).values.mean()) / longest,
            "nothing": float(torch.cdist(t[~f], alone).min(1).values.mean()) / longest,
            "made_up": float(d.min(0).values.mean()) / longest, "turn": turn.cpu(), "start": start}


def made_up() -> None:
    """The measure on shapes with a known answer."""
    rng = np.random.default_rng(0)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 0.6) for z in (0, 0.3)], float)       # a box, no side like another
    walls = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    true = surface(corners, walls, POINTS, rng)
    assert abs(np.abs(true - 0.5 * np.array([1, 0.6, 0.3])).max(0) - 0.5 * np.array([1, 0.6, 0.3])).max() < 1e-9     # the places are on the walls
    cam = true + np.array([-0.5, -0.3, 2.0])                                                                         # seen from in front of its z = 0 wall
    front = facing(cam, surface(corners, walls, MANY, rng) + np.array([-0.5, -0.3, 2.0]), 1.0)
    assert front[true[:, 2] < 1e-9].all() and not front[true[:, 2] > 0.29].any(), "the near wall faces the camera, the far one does not"
    a = 0.7
    turn = torch.tensor([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=torch.float32) @ TURNS[7]
    said = torch.from_numpy(surface(corners, walls, POINTS, rng)).float()
    whole = measure(cam, front, 2.5 * said @ turn.T + torch.tensor([3.0, -1.0, 4.0]), 1.0)                           # the same box, turned, sized and shifted
    assert max(whole["front"], whole["back"], whole["made_up"]) < 0.03, whole                                        # found again, to the spacing of the places
    half = measure(cam, front, torch.from_numpy(true[front]).float(), 1.0)                                           # only the facing side said
    assert half["back"] > 4 * whole["back"] and abs(half["nothing"] - half["back"]) < 0.02, half                         # as bad as nothing imagined, and no worse
    lump = measure(cam, front, torch.cat([said, said + torch.tensor([0.0, 0.0, 1.5])]), 1.0)                         # something said that is not there
    total = lambda m: m["front"] + m["back"] + m["made_up"]  # noqa: E731
    assert total(lump) > 3 * total(whole), lump                                                                      # the best laying resizes, so a lump shows in all three
    solid = torch.zeros(6, 6, 6, dtype=torch.bool)
    solid[1:5, 1:5, 1:5] = True
    assert len(skin(solid)) == 4 ** 3 - 2 ** 3
    print(f"made-up shapes: the same box found again (front {whole['front']:.3f}, back {whole['back']:.3f}); only its facing side said, "
          f"back {half['back']:.3f} (nothing imagined {half['nothing']:.3f}); with a lump that is not there, front {lump['front']:.3f}, back {lump['back']:.3f}, made up {lump['made_up']:.3f}")


def picture(frame: np.ndarray, mine: np.ndarray) -> Image.Image:
    """The object as TripoSR wants it: its box, square, the object filling FILL of it, all else grey."""
    ys, xs = np.nonzero(mine)
    side = int(np.ceil(max(np.ptp(ys), np.ptp(xs)) / FILL)) + 1
    top, left = (ys.min() + ys.max()) // 2 - side // 2, (xs.min() + xs.max()) // 2 - side // 2
    out = np.full((side, side, 3), 127, np.uint8)
    y, x = ys - top, xs - left
    out[y, x] = frame[ys, xs]
    return Image.fromarray(out).resize((512, 512), Image.BICUBIC)


p = argparse.ArgumentParser()
p.add_argument("--made-up", action="store_true")
p.add_argument("--objects")
p.add_argument("--frames", help="the raw clips (<id>/frames/000.png)")
p.add_argument("--assets")
p.add_argument("--triposr", help="a copy of github.com/VAST-AI-Research/TripoSR")
p.add_argument("--clips", type=int, default=40)
p.add_argument("--out", default="shape_prior")
args = p.parse_args()
if args.made_up:
    made_up()
    sys.exit()

sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=None)     # TripoSR's surface drawing, not used here
sys.path.insert(0, str(Path(args.triposr).expanduser()))
from tsr.system import TSR  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt").to(device).eval()
radius = float(model.renderer.cfg.radius)
ask = (torch.stack(torch.meshgrid(*[torch.arange(GRID)] * 3, indexing="ij"), -1).float().view(-1, 3) + 0.5) / GRID * 2 * radius - radius

out, rng = Path(args.out).expanduser(), np.random.default_rng(0)
out.mkdir(parents=True, exist_ok=True)
files = [f for f in sorted(Path(args.objects).expanduser().glob("*.npz")) if (Path(args.frames).expanduser() / f.stem / "frames").exists()]
files = files[:: max(len(files) // args.clips, 1)][: args.clips]
rows, shown, small, empty, seconds = [], [], 0, 0, []
for f in files:
    o = np.load(f)
    if "asset" not in o:
        continue
    T, H, W = o["seg"].shape
    for k, name in enumerate(o["asset"].tolist()):
        count = (o["seg"] == k + 1).reshape(T, -1).sum(1)
        t = int(count.argmax())
        frame = np.asarray(Image.open(Path(args.frames).expanduser() / f.stem / "frames" / f"{t:03d}.png").convert("RGB"))
        mine = np.asarray(Image.fromarray((o["seg"][t] == k + 1).astype(np.uint8) * 255).resize(frame.shape[1::-1], Image.NEAREST)) > 0
        if mine.sum() < SMALL * (frame.shape[0] / H) ** 2:
            small += 1
            continue
        longest = float(o["scale"][k])
        true = surface(*shape(name, Path(args.assets).expanduser(), longest, FINE), POINTS + MANY, rng)
        world = true @ quat_to_rot(o["quaternions"][k][t:t + 1].astype(np.float64))[0].T + o["positions"][k][t]
        cam = ((world - o["camera_positions"][t]) @ quat_to_rot(o["camera_quaternions"][t:t + 1].astype(np.float64))[0]) @ FLIP.T
        cam, whole = cam[:POINTS], cam[POINTS:]
        front = facing(cam, whole, longest)
        px = cam[front, :2] / cam[front, 2:] * float(o["focal_length"]) / float(o["sensor_width"]) * W + np.array([W, H]) / 2
        inside = (px >= 0).all(1) & (px[:, 0] < W) & (px[:, 1] < H)
        seen = float((o["seg"][t][px[inside, 1].astype(int), px[inside, 0].astype(int)] == k + 1).sum() / max(len(px), 1))   # of its facing side, the share in the picture
        image = picture(frame, mine)
        if len(shown) < 24:
            shown.append(image.resize((128, 128)))
        with torch.no_grad():
            if device == "cuda":
                torch.cuda.synchronize()
            began = time.time()
            code = model([image], device=device)[0]
            solid = (model.renderer.query_triplane(model.decoder, ask.to(device), code)["density_act"].view(GRID, GRID, GRID) > SOLID)
            if device == "cuda":
                torch.cuda.synchronize()
            seconds.append(time.time() - began)
            said = skin(solid)
            if len(said) < 50:
                empty += 1
                continue
            m = measure(cam, front, said[torch.randperm(len(said))[:POINTS]], longest)
        rows.append({**m, "seen": seen, "pixels": int(mine.sum())})
        print(f"clip {f.stem} object {k} {name[:30]}: {seen:.0%} of its facing side seen, {int(mine.sum())} pixels; front {m['front'] * CELLS:.2f} cells, "
              f"back {m['back'] * CELLS:.2f}, nothing imagined {m['nothing'] * CELLS:.2f}, made up {m['made_up'] * CELLS:.2f}; start {m['start']}", flush=True)

sheet = Image.new("RGB", (128 * 8, 128 * ((len(shown) + 7) // 8)))
for i, im in enumerate(shown):
    sheet.paste(im, (128 * (i % 8), 128 * (i // 8)))
sheet.save(out / "pictures.png")
torch.save(rows, out / "rows.pt")


def told(name: str, some: list) -> None:
    if not some:
        return
    med = {key: float(np.median([r[key] for r in some])) * CELLS for key in ("front", "back", "nothing", "made_up")}
    better = float(np.mean([r["back"] < r["nothing"] for r in some]))
    print(f"=== {name}: {len(some)} objects; in cells of {CELLS} across, median: front {med['front']:.2f}, back {med['back']:.2f}, "
          f"nothing imagined {med['nothing']:.2f}, made up {med['made_up']:.2f}; the back nearer than with nothing imagined for {better:.0%} ===", flush=True)


told("all", rows)
told("facing side at least 90% seen", [r for r in rows if r["seen"] >= 0.9])
told("50 to 90% seen", [r for r in rows if 0.5 <= r["seen"] < 0.9])
told("under 50% seen", [r for r in rows if r["seen"] < 0.5])
starts = np.bincount([r["start"] for r in rows], minlength=24)
# Is TripoSR's box tied to the view? Were it, every object's turn onto the camera's frame would be one turn.
mean = torch.stack([r["turn"] for r in rows if r["start"] == starts.argmax()]).mean(0)
u, _, vt = torch.linalg.svd(mean)
angle = [float(torch.rad2deg(torch.acos((((u @ vt).T @ r["turn"]).trace().clamp(-1, 3) - 1) / 2))) for r in rows]
print(f"=== TripoSR's box against the camera: the commonest turn's share {starts.max() / max(len(rows), 1):.0%}; "
      f"degrees from the one turn nearest all, median {np.median(angle):.0f}, 90% under {np.quantile(angle, 0.9):.0f} ===")
print(f"=== left out: {small} too small (under {SMALL} pixels), {empty} where TripoSR said nothing; "
      f"seconds an object, median {np.median(seconds):.2f} on {device} ===", flush=True)
