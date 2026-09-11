"""
Download Kubric clips over plain HTTPS, bypassing huggingface_hub entirely.

Why not `snapshot_download`: HuggingFace now serves file data through Xet
(`us.aws.cdn.hf.co/xet-bridge-us/...`), and the `hf_xet` client it uses does
not get through on NSCC -- it hangs with no error and no progress. Plain HTTPS
to the same redirect works, which is what this does.

Resumable: files already on disk with a non-zero size are skipped, so rerunning
after an interruption only fetches what is missing.

Run:  python scripts/download_kubric.py --out ~/scratch/kubric --clips 500
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "zbww/tapip3d-kubric"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def get(url: str, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "genpoint3d"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def list_clip(seq: str) -> list[str]:
    """Paths inside one sequence folder, via the (non-Xet) API."""
    out = []
    for sub in ("", "/frames"):
        try:
            entries = json.loads(get(f"{API}/{seq}{sub}?recursive=false"))
        except urllib.error.HTTPError:
            continue
        out += [e["path"] for e in entries if e["type"] == "file"]
    return out


def fetch(path: str, root: Path) -> str | None:
    dest = root / path
    if dest.exists() and dest.stat().st_size > 0:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(get(f"{RAW}/{path}"))
    tmp.rename(dest)
    return path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--clips", type=int, default=500)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    root = Path(args.out).expanduser()
    seqs = [f"{i:06d}" for i in range(args.start, args.start + args.clips)]

    print(f"listing {len(seqs)} clips...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        files = [f for group in ex.map(list_clip, seqs) for f in group]
    print(f"  {len(files)} files ({time.time() - t0:.0f}s)", flush=True)
    if not files:
        print("NOTHING FOUND -- sequence naming is different than expected")
        return 1

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for got in ex.map(lambda f: fetch(f, root), files):
            done += 1
            if done % 200 == 0 or done == len(files):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(files)}  {rate:.0f} files/s"
                      f"  eta {(len(files) - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    print(f"DONE -> {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
