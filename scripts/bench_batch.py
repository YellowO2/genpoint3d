"""
Find the batch size that actually uses the GPU.

Doubling the batch is worth it for as long as the time per step does NOT double
with it -- that means calculators were sitting idle. Once time scales in
proportion, the GPU is saturated and a larger batch only costs memory. So the
number to watch is CLIPS/SEC, not seconds per step:

    batch 16 -> 0.79 s/it    20 clips/s
    batch 32 -> 0.85 s/it    38 clips/s   <- nearly free, take it
    batch 64 -> 1.60 s/it    40 clips/s   <- no gain, the wall

`nvidia-smi`'s GPU-Util is not a substitute: it reports the fraction of time at
least one kernel was running, so a tiny kernel running constantly reads as
100% while wasting most of the chip. Throughput cannot be fooled that way.

Timed directly rather than through train.py, whose printed s/it is a cumulative
average including startup and validation pauses -- far too sluggish to read a
doubling off.

Run:  python scripts/bench_batch.py --cache ~/scratch/cache/kubric
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from genpoint3d.models.flow import flow_matching_loss
from genpoint3d.models.model import PointDiT
from train import ClipDataset, split, to_device


def time_one(clips, feat_dim, has_feats, args, batch, workers) -> dict:
    """One configuration: median seconds per step, and peak memory."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointDiT(dim=args.dim, depth=args.depth, num_heads=args.heads,
                     cross_attn=has_feats, feat_dim=feat_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    loader = DataLoader(ClipDataset(clips, args.points), batch_size=batch,
                        shuffle=True, num_workers=workers, drop_last=True,
                        persistent_workers=workers > 0)

    if len(loader) == 0:
        raise ValueError(f"only {len(clips)} clips, fewer than batch {batch}")

    times, it = [], iter(loader)
    for i in range(args.warmup + args.steps):
        batch_data = next(it, None)
        if batch_data is None:          # wrap around; the loader is finite
            it = iter(loader)
            batch_data = next(it)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        b = to_device(batch_data, device)
        traj, anchor, vis = b["traj"], b["anchor"], b["visibility"]
        ctx, idc = b["context"], b["id_card"]
        vm = torch.ones(traj.shape[:2], dtype=torch.bool, device=device) if ctx is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, _ = flow_matching_loss(model, traj, anchor, mask=vis,
                                         context=ctx, visual_mask=vm, id_card=idc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= args.warmup:            # the first steps include allocation
            times.append(time.time() - t0)

    del model, opt, loader
    s_it = float(torch.tensor(times).median())
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    return {"s_it": s_it, "clips_s": batch / s_it, "peak_gb": peak}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    p.add_argument("--workers", type=int, nargs="+", default=[4])
    p.add_argument("--points", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=30, help="timed steps per setting")
    p.add_argument("--warmup", type=int, default=5, help="untimed steps first")
    args = p.parse_args()

    clips, _, feat_dim, has_feats = split(args.cache, 0.02, 0)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"{dev} | {len(clips)} clips | points {args.points} | "
          f"dim {args.dim} depth {args.depth}\n", flush=True)
    print(f"{'workers':>7} {'batch':>6} {'s/it':>8} {'clips/s':>9} {'peak GB':>9}"
          f"  {'vs prev':>8}", flush=True)

    for workers in args.workers:
        prev = None
        for batch in args.batches:
            try:
                r = time_one(clips, feat_dim, has_feats, args, batch, workers)
            except torch.OutOfMemoryError:
                print(f"{workers:>7} {batch:>6}   OUT OF MEMORY", flush=True)
                torch.cuda.empty_cache()
                break
            except ValueError as e:
                print(f"{workers:>7} {batch:>6}   skipped -- {e}", flush=True)
                continue
            # Throughput gain over the previous batch size. Near 2.0 means the
            # GPU was idle and doubling was nearly free; near 1.0 is the wall.
            gain = f"{r['clips_s'] / prev:.2f}x" if prev else "--"
            print(f"{workers:>7} {batch:>6} {r['s_it']:>8.3f} {r['clips_s']:>9.1f}"
                  f" {r['peak_gb']:>9.2f}  {gain:>8}", flush=True)
            prev = r["clips_s"]
        print(flush=True)

    print("Take the largest batch whose gain is still clearly above 1.0.")
    print("Then raise --lr with it: a bigger batch means fewer, better-aimed steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
