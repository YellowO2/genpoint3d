"""The 3D models Kubric's clips are made of (Google Scanned Objects), kept whole in one directory."""

import io
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

STORE = "https://storage.googleapis.com/kubric-public/assets/GSO/{}.tar.gz"
WHOLE = ".whole"                  # written once every file of a model is there
SHAPE = "collision_geometry.obj"  # the low-poly shape Kubric's physics engine used; visual_geometry.obj is the fine one


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


def shape(name: str, home: Path, longest: float) -> tuple[np.ndarray, list]:
    """A model's corners (V, 3) in metres and its walls (F, 3), sized as Kubric sizes it: `longest` its longest side."""
    corners, walls = [], []
    for line in (fetch(name, home) / SHAPE).read_text().splitlines():
        part = line.split()
        if part[:1] == ["v"]:
            corners.append([float(x) for x in part[1:4]])
        elif part[:1] == ["f"]:
            i = [int(x.split("/")[0]) - 1 for x in part[1:]]
            walls += [[i[0], i[k], i[k + 1]] for k in range(1, len(i) - 1)]
    corners = np.array(corners)
    return corners * longest / (corners.max(0) - corners.min(0)).max(), walls


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
