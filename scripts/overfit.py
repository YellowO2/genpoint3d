"""
Step 2 smoke test: can the backbone memorise two clips?

No image conditioning at all -- no DINOv3, no cross-attention, no feature
cloud. Just noisy trajectory in, clean trajectory out. If this cannot overfit
two clips, the bug is in the backbone or the flow-matching code, and finding
that out with an encoder attached would be miserable.

`--regression` is the deliberate control: same network, no noise, no timestep,
predict `x1` directly. If regression overfits but flow matching does not, the
fault is in the flow-matching machinery rather than the transformer or the
data.

Run:  .venv/bin/python scripts/overfit.py [--steps 2000] [--regression]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from genpoint3d.data.kubric import KubricSequenceDataset
from genpoint3d.data.transform import transform
from genpoint3d.models.model import PointDiT
from genpoint3d.models.flow import flow_matching_loss, sample


def load_clips(root: str, num_points: int, device: torch.device):
    """Whole dataset as one batch -- it is two clips, so this fits trivially."""
    ds = KubricSequenceDataset(root, num_query_points=num_points)
    traj, anchor, vis = [], [], []
    for i in range(len(ds)):
        sample_i = ds[i]
        inputs = transform(sample_i, num_context_frames=sample_i.frames.shape[0])
        traj.append(inputs.traj)
        anchor.append(inputs.anchor)
        vis.append(inputs.visibility)
    return (
        torch.stack(traj).to(device),
        torch.stack(anchor).to(device),
        torch.stack(vis).to(device),
    )


def regress(model, anchor, num_frames: int) -> torch.Tensor:
    """Control mode: predict the trajectory directly, no noise, no timestep.

    The input tokens are the frame-0 anchor repeated across time. They cannot
    be zeros: AdaRMSNorm conditioning is purely *multiplicative* (it predicts a
    scale for the norm), and `rms_norm(0) = 0`, so a zero token stream stays
    zero through the entire network and no gradient flows at all.
    """
    x = anchor[:, None].expand(-1, num_frames, -1, -1)
    return model(x, torch.zeros(anchor.shape[0], device=anchor.device), anchor)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/kubric_test")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--points", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--regression", action="store_true", help="control run: no noise, predict x1 directly")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)

    x1, anchor, vis = load_clips(args.root, args.points, device)
    B, T, N, _ = x1.shape
    print(f"device {device} | {B} clips, T={T}, N={N}, target std {x1.std():.3f}")

    model = PointDiT(dim=args.dim, depth=args.depth, num_heads=args.heads).to(device)
    print(f"model  {model.num_parameters() / 1e6:.2f}M params"
          f" | mode {'regression' if args.regression else 'flow matching'}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.regression:
            loss = (regress(model, anchor, T) - x1).pow(2).mean(-1)[vis].mean()
        else:
            loss, _ = flow_matching_loss(model, x1, anchor, mask=vis)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % max(1, args.steps // 20) == 0 or step == 1:
            print(f"  step {step:>5}  loss {loss.item():>9.5f}  grad {grad:>7.3f}"
                  f"  {(time.time() - t0) / step:.2f}s/it")

    # Did it actually learn the trajectories, or just the mean?
    model.eval()
    print("\nevaluating:")
    if args.regression:
        pred = regress(model, anchor, T)
    else:
        pred = sample(model, anchor, num_frames=T, steps=50)

    err = (pred - x1).pow(2).mean(-1).sqrt()
    baseline = (x1.mean(dim=(1, 2), keepdim=True) - x1).pow(2).mean(-1).sqrt()
    print(f"  RMSE vs GT        {err[vis].mean():.4f}  (model units)")
    print(f"  RMSE of mean traj {baseline[vis].mean():.4f}  <- beat this, or nothing was learnt")
    print(f"  ratio             {err[vis].mean() / baseline[vis].mean():.3f}  (want << 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
