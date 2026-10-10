"""
Fetch CoTracker3's offline checkpoint and keep only its CNN.

The relative finder's CNN (`genpoint3d/models/fine.py::BasicEncoder`) is
CoTracker3's `fnet` under the same names, so its weights load directly. They
were trained for exactly this job -- features whose 7x7 windows correlate at
the right pixel -- on Kubric, at 4 px per cell. Everything else in the
checkpoint is CoTracker's own transformer, which we do not have.

The checkpoint is 102 MB; what is saved is the CNN alone, 10 MB.

Run once, on a machine with internet (the NSCC login node):

    python scripts/fetch_fnet.py --out ~/scratch/weights/cotracker3_fnet.pt
    python scripts/train.py ... --fine 1 --fine-init ~/scratch/weights/cotracker3_fnet.pt
"""

import argparse
import os
from pathlib import Path

# The Xet transfer client is what hangs on NSCC (docs/misc.md); plain HTTPS
# does not. Has to be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from huggingface_hub import hf_hub_download

from genpoint3d.models.fine import BasicEncoder

REPO, FILE = "facebook/cotracker3", "scaled_offline.pth"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="where to write the CNN's weights")
    args = p.parse_args()

    path = hf_hub_download(REPO, FILE)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    # The release is a bare state dict; a training checkpoint nests it.
    ckpt = ckpt.get("model", ckpt)
    fnet = {k.split("fnet.", 1)[1]: v for k, v in ckpt.items() if "fnet." in k}
    if not fnet:
        raise SystemExit(f"no fnet.* weights in {REPO}/{FILE}: {list(ckpt)[:5]} ...")

    # Strict, into the module that will use them: a renamed or reshaped layer
    # fails here, not in a training job.
    BasicEncoder().load_state_dict(fnet)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fnet, out)
    n = sum(v.numel() for v in fnet.values())
    print(f"wrote {out}: {len(fnet)} tensors, {n / 1e6:.2f}M weights"
          f" ({out.stat().st_size / 1e6:.1f} MB) from {REPO}/{FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
