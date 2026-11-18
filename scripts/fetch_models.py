"""Every 3D model the clips' objects are (scripts/movi_objects.py names them), fetched whole from Kubric's public store.

  python scripts/fetch_models.py --objects OBJECTS --assets DIR
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from genpoint3d.tools.kubric_models import fetch

p = argparse.ArgumentParser()
p.add_argument("--objects", required=True)
p.add_argument("--assets", required=True)
p.add_argument("--at_once", type=int, default=8)
args = p.parse_args()

home = Path(args.assets).expanduser()
names, clips = set(), 0
for f in sorted(Path(args.objects).expanduser().glob("*.npz")):
    d = np.load(f)
    if "asset" in d:
        names |= set(d["asset"].tolist())
        clips += 1
print(f"{clips} clips name {len(names)} models", flush=True)


def one(name: str) -> str:
    try:
        fetch(name, home)
        return ""
    except Exception as e:  # a model that cannot be fetched is reported, the rest go on
        return f"{name}: {e}"


with ThreadPoolExecutor(args.at_once) as pool:
    failed = [r for i, r in enumerate(pool.map(one, sorted(names))) if print(f"{i + 1} of {len(names)}", flush=True) or r]
print(f"=== {len(names) - len(failed)} of {len(names)} models in {home}; failed: {failed} ===", flush=True)
