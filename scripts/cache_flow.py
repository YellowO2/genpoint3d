"""
Cache each clip's optical flow between neighbouring frames, beside its frames.

SEA-RAFT (github.com/princeton-vl/SEA-RAFT), pretrained and frozen, on the
frames cache's square frames: frame t to t+1 and back, every pair of the clip
in one batch. How each pixel moves between two frames needs no guess about any
point, so it is computed once here, like DINOv3's features, and not per step.

One `<clip>.flow.npy` per clip: (2, T-1, 2, res / 4, res / 4) float16 --
forward (t -> t+1) then backward (t+1 -> t), each (du, dv) as a fraction of
the frame across and down, so it does not depend on the frame's size. Read at
the CNN's cells as the depth is: cell j is pixel INDEX 4j of the small frame,
`(j + 1/8) / (res / 4)` across, one pixel read and nothing averaged, so a
cell on an object's edge moves with one of the two surfaces. 1.7 MB a clip at
384 px.

Resumable: clips already in the destination are skipped.

Run:  python scripts/cache_flow.py --frames FRAMES --dst FLOW \\
          --repo references/searaft/repo --weights <MemorySlices/... snapshot>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

FLOW = ".flow.npy"


def load_raft(repo: Path, weights: Path, cfg: str):
    """SEA-RAFT with its released weights. Its backbones would first fetch
    ImageNet weights from torchvision, which the checkpoint then replaces
    whole; that first step is skipped, so nothing is downloaded."""
    sys.path.insert(0, str(repo / "core"))
    import extractor
    from raft import RAFT
    from safetensors.torch import load_file
    extractor.ResNetFPN._init_weights = lambda self, args: None
    args = argparse.Namespace(**json.loads((repo / "config" / "eval" / cfg).read_text()))
    model = RAFT(args)
    # The checkpoint has no weights for the downsampling branches' BatchNorms,
    # and SEA-RAFT's own loader (non-strict) leaves them at their defaults, as
    # here. Anything else missing, or anything extra, is the wrong file.
    missing, unexpected = model.load_state_dict(load_file(weights / "model.safetensors"), strict=False)
    if unexpected or [k for k in missing if ".downsample.1." not in k]:
        raise ValueError(f"{weights} does not fit SEA-RAFT {cfg}: missing {missing}, unexpected {unexpected}")
    return model, args


@torch.no_grad()
def clip_flow(model, args, frames: np.ndarray, device) -> np.ndarray:
    """(T, R, R, 3) uint8 -> (2, T-1, 2, R/4, R/4) float16."""
    x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float()
    a = torch.cat([x[:-1], x[1:]])
    b = torch.cat([x[1:], x[:-1]])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        flow = model(a, b, iters=args.iters, test_mode=True)["flow"][-1].float()   # (2(T-1), 2, R, R) px
    R = frames.shape[1]
    flow = flow[:, :, ::4, ::4] / R
    return flow.reshape(2, -1, *flow.shape[1:]).cpu().numpy().astype(np.float16)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, help="frames cache (scripts/cache_frames.py)")
    p.add_argument("--dst", required=True)
    p.add_argument("--repo", required=True, help="a clone of SEA-RAFT")
    p.add_argument("--weights", required=True, help="a snapshot of its HuggingFace weights")
    p.add_argument("--cfg", default="spring-M.json")
    p.add_argument("--limit", type=int, default=0, help="only the first N clips")
    p.add_argument("--only", default="", help="only the clips of this feature cache directory")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_raft(Path(args.repo), Path(args.weights), args.cfg)
    model = model.to(device).eval()
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    clips = sorted(f for f in Path(args.frames).glob("*.npy") if not f.name.endswith(".depth.npy"))
    if args.only:
        keep = {c.stem for c in Path(args.only).glob("*.pt")}
        clips = [c for c in clips if c.stem in keep]
    if args.limit:
        clips = clips[: args.limit]
    todo = [c for c in clips if not (dst / f"{c.stem}{FLOW}").exists()]
    print(f"{len(clips)} clips, {len(todo)} to do | {device} | SEA-RAFT {args.cfg}", flush=True)
    start = time.time()
    for i, c in enumerate(todo, 1):
        out = clip_flow(model, margs, np.load(c), device)
        tmp = dst / f"{c.stem}.tmp.npy"
        np.save(tmp, out)
        tmp.rename(dst / f"{c.stem}{FLOW}")
        if i % 100 == 0 or i == len(todo):
            rate = (time.time() - start) / i
            print(f"  {i}/{len(todo)}  {rate:.2f} s a clip, {rate * (len(todo) - i) / 60:.0f} min left",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
