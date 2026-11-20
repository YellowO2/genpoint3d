"""The 3D models Kubric's clips are made of (Google Scanned Objects), kept whole in one directory."""

import io
import math
import tarfile
from functools import lru_cache
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

STORE = "https://storage.googleapis.com/kubric-public/assets/GSO/{}.tar.gz"
WHOLE = ".whole"                  # written once every file of a model is there
SHAPE = "collision_geometry.obj"  # the low-poly shape Kubric's physics engine used
FINE = "visual_geometry.obj"      # the fine one, the surface the video shows
CELLS = 8                         # the answer's detail: an object's corners in one cell of a grid this many across are one
SHAPE_TOL = 0.03                  # metres the low-poly shape may sit off the surface the video shows (90% within, world_shapes_against_tracks)


def fetch(name: str, home: Path) -> Path:
    """The model's directory under `home`, fetched whole from Kubric's public store if not there."""
    at = home / name
    if not (at / WHOLE).exists():
        at.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(STORE.format(urllib.parse.quote(name))) as r, tarfile.open(fileobj=io.BytesIO(r.read())) as tar:
            for m in tar.getmembers():
                if m.isfile():
                    (at / Path(m.name).name).write_bytes(tar.extractfile(m).read())
        (at / WHOLE).touch()
    return at


def shape(name: str, home: Path, longest: float, file: str = SHAPE) -> tuple[np.ndarray, list]:
    """A model's corners (V, 3) in metres and its walls (F, 3), sized as Kubric sizes it: `longest` its longest side."""
    corners, walls = [], []
    for line in (fetch(name, home) / file).read_text().splitlines():
        part = line.split()
        if part[:1] == ["v"]:
            corners.append([float(x) for x in part[1:4]])
        elif part[:1] == ["f"]:
            i = [int(x.split("/")[0]) - 1 for x in part[1:]]
            walls += [[i[0], i[k], i[k + 1]] for k in range(1, len(i) - 1)]
    corners = np.array(corners)
    return corners * longest / (corners.max(0) - corners.min(0)).max(), walls


def things(o, home: Path) -> np.ndarray:
    """(O, 7) what Kubric says of each of a clip's objects `o`: the log of its mass (kg), its grip (friction), its
    bounce (restitution), the log of its longest side (metres), and its colour, red, green and blue from 0 to 1."""
    return np.array([[math.log(float(o["mass"][k])), float(o["friction"][k]), float(o["restitution"][k]), math.log(float(o["scale"][k])),
                      *(c / 255 for c in colour(name, home))] for k, name in enumerate(o["asset"].tolist())], dtype=np.float32).reshape(-1, 7)


@lru_cache(maxsize=None)
def colour(name: str, home: Path) -> list:
    """A model's one colour, the mean of its texture."""
    return np.asarray(Image.open(fetch(name, home) / "texture.png").convert("RGB").resize((16, 16))).reshape(-1, 3).mean(0).round().astype(int).tolist()


def thinned(corners: np.ndarray, walls: list, cells: int) -> tuple[np.ndarray, list]:
    """The shape with its corners in one cell of a grid, `cells` across its longest side, made one (their middle),
    and the walls left with three different corners; whole for 0 cells."""
    if not cells:
        return corners, walls
    low = corners.min(0)
    cell = np.floor((corners - low) / ((corners.max(0) - low).max() / cells + 1e-9)).astype(int)
    one = np.unique(cell, axis=0, return_inverse=True)[1].reshape(-1)
    middle = np.stack([np.bincount(one, corners[:, k]) for k in range(3)], 1) / np.bincount(one)[:, None]
    kept = {tuple(sorted(f)): f for f in one[np.array(walls)].tolist() if len(set(f)) == 3}
    return middle, list(kept.values())


FLIP = np.diag([1.0, -1.0, -1.0])  # Blender's camera axes to OpenCV's


def turns(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternions, w first -> (..., 3, 3) the turns they are."""
    w, x, y, z = (q[..., k] for k in range(4))
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def fewest(corners: np.ndarray, most: int) -> np.ndarray:
    """Which `most` of the corners (V, 3) to keep, each the one farthest from those kept so far; all if no more."""
    if len(corners) <= most:
        return np.arange(len(corners))
    kept = [int(np.linalg.norm(corners - corners.mean(0), axis=1).argmax())]
    gap = np.linalg.norm(corners - corners[kept[0]], axis=1)
    while len(kept) < most:
        kept.append(int(gap.argmax()))
        gap = np.minimum(gap, np.linalg.norm(corners - corners[kept[-1]], axis=1))
    return np.array(kept)


MOVED = 0.01   # metres an object's centre must go over a clip to be known free: one that never moves may be fixed


def moves(o) -> np.ndarray:
    """(O,) which of a clip's objects `o` move over it: those known to be free to move (MOVi's objects are all
    dropped or thrown, but the npz does not say which were fixed)."""
    p = o["positions"]
    return (np.linalg.norm(p - p[:, :1], axis=-1).max(1) > MOVED) if len(p) else np.zeros(0, bool)


def floor(o, ground: np.ndarray) -> np.ndarray:
    """(4, 3) the floor as a game's flat body: the corners of the rectangle on Kubric's floor (z = 0) that holds
    `ground` (P, 3), the clip's floor points in metres seen from its first frame's camera, in that camera."""
    turn0, at0 = turns(o["camera_quaternions"][0].astype(np.float64)), o["camera_positions"][0].astype(np.float64)
    world = (ground.astype(np.float64) @ FLIP) @ turn0.T + at0
    (x0, y0), (x1, y1) = world[:, :2].min(0), world[:, :2].max(0)
    corners = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
    return ((corners - at0) @ turn0) @ FLIP


def answer(o, home: Path, cells: int, most: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What a clip's stored world should come to be: every object's low-poly shape, all sides, placed by Kubric's
    pose in every frame and seen from the first frame's camera (a cached clip's frame), in metres. `o` is the
    clip's objects (scripts/movi_objects.py's). The corners (T, V, 3) of all its objects, each one's object
    (V,), 1 the first as in the clip's segments, and the bars (E, 2) between them. An object's corners are
    `thinned` to `cells` across; one with more than `most` keeps those farthest apart, and the bars among them."""
    turn0, at0 = turns(o["camera_quaternions"][0].astype(np.float64)), o["camera_positions"][0].astype(np.float64)
    places, of, bars, count = [], [], [], 0
    for k, name in enumerate(o["asset"].tolist()):
        corners, walls = thinned(*shape(name, home, float(o["scale"][k])), cells)
        keep = fewest(corners, most)
        new = np.full(len(corners), -1)
        new[keep] = np.arange(len(keep))
        pairs = {tuple(sorted((int(new[a]), int(new[b])))) for f in walls for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))
                 if new[a] >= 0 and new[b] >= 0}
        world = np.einsum("tij,vj->tvi", turns(o["quaternions"][k].astype(np.float64)), corners[keep]) + o["positions"][k][:, None]
        places.append(((world - at0) @ turn0) @ FLIP)
        of.append(np.full(len(keep), k + 1))
        bars += [(a + count, b + count) for a, b in sorted(pairs)]
        count += len(keep)
    T = len(o["camera_positions"])
    if not places:
        return np.zeros((T, 0, 3)), np.zeros(0, int), np.zeros((0, 2), int)
    return np.concatenate(places, 1), np.concatenate(of), np.array(bars, int).reshape(-1, 2)
