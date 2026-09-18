"""
Where does the training step's GPU memory actually go?

The batch sweep found batch 16 peaking at 23.9 GB for a 17.9M-parameter model,
and batch 32 out of memory. Counting the obvious tensors by hand accounts for
about 5 GB of that, so the rest was unexplained -- and a guess about attention
matrices does not survive the fact that `F.scaled_dot_product_attention` uses
FlashAttention kernels, which recompute the matrix in the backward pass rather
than storing it.

So: measure. This reports allocated memory at each stage of one step, then
PyTorch's own per-operator breakdown, so the largest consumer is named rather
than inferred.

Run:  python scripts/profile_memory.py --cache ~/scratch/cache/kubric --batch 16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from genpoint3d.models.flow import flow_matching_loss
from genpoint3d.models.model import PointDiT
from train import ClipDataset, split, to_device

GB = 2 ** 30


def gb() -> float:
    return torch.cuda.memory_allocated() / GB


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--points", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--trace", type=Path, default=None,
                   help="also write a chrome trace here")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU -- submit this as a job")

    device = torch.device("cuda")
    clips, _, feat_dim, has_feats = split(args.cache, 0.02, 0)
    loader = DataLoader(ClipDataset(clips, args.points), batch_size=args.batch,
                        shuffle=True, num_workers=4, drop_last=True)

    torch.cuda.reset_peak_memory_stats()
    print(f"{torch.cuda.get_device_name(0)} | batch {args.batch} "
          f"| points {args.points} | dim {args.dim} depth {args.depth}\n")

    model = PointDiT(dim=args.dim, depth=args.depth, num_heads=args.heads,
                     cross_attn=has_feats, feat_dim=feat_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp = not args.no_amp and torch.cuda.is_bf16_supported()
    print(f"amp {'bf16' if amp else 'off'}")
    print(f"  after model                 {gb():7.2f} GB")

    it = iter(loader)
    for phase in ("warmup", "measured"):
        batch = next(it)
        b = to_device(batch, device, amp)
        traj, anchor, vis = b["traj"], b["anchor"], b["visibility"]
        ctx, idc = b["context"], b["id_card"]
        if phase == "measured":
            print(f"  after batch on device       {gb():7.2f} GB"
                  f"   (context {ctx.numel() * ctx.element_size() / GB:.2f} GB"
                  f", {tuple(ctx.shape)} {ctx.dtype})")

        vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=device) if ctx is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, _ = flow_matching_loss(model, traj, anchor, mask=vis,
                                         context=ctx, visual_mask=vm, id_card=idc)
        if phase == "measured":
            print(f"  after forward               {gb():7.2f} GB"
                  f"   <- activations kept for backward")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if phase == "measured":
            print(f"  after backward              {gb():7.2f} GB")
        opt.step()
        if phase == "measured":
            print(f"  after optimiser step        {gb():7.2f} GB"
                  f"   (AdamW keeps 2 states per parameter)")
            print(f"\n  PEAK                        "
                  f"{torch.cuda.max_memory_allocated() / GB:7.2f} GB")

    # Per-operator breakdown: names the biggest consumer instead of inferring it.
    from torch.profiler import ProfilerActivity, profile

    batch = next(it)
    b = to_device(batch, device, amp)
    traj, anchor, vis, ctx, idc = (b["traj"], b["anchor"], b["visibility"],
                                   b["context"], b["id_card"])
    vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=device) if ctx is not None else None
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=True) as prof:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, _ = flow_matching_loss(model, traj, anchor, mask=vis,
                                         context=ctx, visual_mask=vm, id_card=idc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print("\ntop operators by CUDA memory allocated:\n")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))
    if args.trace:
        prof.export_chrome_trace(str(args.trace))
        print(f"\nwrote {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
