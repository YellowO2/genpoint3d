"""
What does a training step cost? Three questions, one harness.

    --what speed    which batch size to use
    --what memory   where the GPU memory goes
    --what sweep    which dimension drives it

One harness, three reports: each needs the same setup -- split the cache, build
the model, run one step -- and differs only in what it measures.

Run:  python scripts/bench.py --cache ~/scratch/cache/kubric --what speed
      python scripts/bench.py --cache ~/scratch/cache/kubric --what memory --batch 16
"""

import argparse
import sys
import time
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


# ------------------------------------------------------------------ harness

def build(args, clips, feat_dim, has_feats, device, batch, depth, points, workers):
    torch.manual_seed(0)
    model = PointDiT(dim=args.dim, depth=depth, num_heads=args.heads,
                     cross_attn=has_feats, feat_dim=feat_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = DataLoader(ClipDataset(clips, points), batch_size=batch,
                        shuffle=True, num_workers=workers, drop_last=True,
                        persistent_workers=workers > 0)
    if len(loader) == 0:
        raise ValueError(f"only {len(clips)} clips, fewer than batch {batch}")
    amp = (not args.no_amp) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    return model, opt, loader, amp


def unpack(batch, device, amp, patches=None):
    b = to_device(batch, device, amp)
    ctx, pxyz = b["context"], b["patch_xyz"]
    # Slicing the patch count emulates a smaller --image-size at preprocess
    # time. Features and positions are paired per patch, so both must be cut or
    # the model's `frame_proj(ctx) + patch_pos(pxyz)` gets mismatched shapes.
    if ctx is not None and patches is not None and patches < ctx.shape[2]:
        ctx, pxyz = ctx[:, :, :patches].contiguous(), pxyz[:, :, :patches].contiguous()
    vm = (torch.ones(b["traj"].shape[:2], dtype=torch.bool, device=device)
          if ctx is not None else None)
    return dict(traj=b["traj"], anchor=b["anchor"], mask=b["visibility"],
                context=ctx, visual_mask=vm, id_card=b["id_card"], patch_xyz=pxyz)


def step(model, opt, kw, amp):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        loss, _ = flow_matching_loss(model, kw.pop("traj"), kw.pop("anchor"), **kw)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss


def peak_for(args, clips, feat_dim, has_feats, device, batch, depth, points, patches) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, opt, loader, amp = build(args, clips, feat_dim, has_feats, device,
                                    batch, depth, points, workers=4)
    it = iter(loader)
    for _ in range(2):                       # first step allocates, second is real
        step(model, opt, unpack(next(it), device, amp, patches), amp)
    peak = torch.cuda.max_memory_allocated() / GB
    del model, opt, loader
    return peak


# -------------------------------------------------------------- what: speed

def run_speed(args, clips, feat_dim, has_feats, device) -> int:
    """Doubling the batch is worth it while the time per step does NOT double --
    that means calculators were idle. So watch CLIPS/SEC, not seconds per step.

    nvidia-smi's GPU-Util is no substitute: it reports the fraction of time at
    least one kernel ran, so a tiny kernel running constantly reads as 100%.

    Timed here rather than read off train.py, whose printed s/it is a cumulative
    average including startup and validation pauses.
    """
    print(f"{'workers':>7} {'batch':>6} {'s/it':>8} {'clips/s':>9} {'peak GB':>9}"
          f"  {'vs prev':>8}", flush=True)

    for workers in args.workers:
        prev = None
        for batch in args.batches:
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                model, opt, loader, amp = build(args, clips, feat_dim, has_feats,
                                                device, batch, args.depth,
                                                args.points, workers)
                times, it = [], iter(loader)
                for i in range(args.warmup + args.steps):
                    bd = next(it, None)
                    if bd is None:                    # wrap; the loader is finite
                        it = iter(loader)
                        bd = next(it)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                    step(model, opt, unpack(bd, device, amp), amp)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    if i >= args.warmup:              # first steps allocate
                        times.append(time.time() - t0)
                del model, opt, loader
            except torch.OutOfMemoryError:
                print(f"{workers:>7} {batch:>6}   OUT OF MEMORY", flush=True)
                torch.cuda.empty_cache()
                break
            except ValueError as e:
                print(f"{workers:>7} {batch:>6}   skipped -- {e}", flush=True)
                continue

            s_it = float(torch.tensor(times).median())
            clips_s = batch / s_it
            peak = torch.cuda.max_memory_allocated() / GB if device.type == "cuda" else 0.0
            gain = f"{clips_s / prev:.2f}x" if prev else "--"
            print(f"{workers:>7} {batch:>6} {s_it:>8.3f} {clips_s:>9.1f}"
                  f" {peak:>9.2f}  {gain:>8}", flush=True)
            prev = clips_s
        print(flush=True)

    print("Take the largest batch whose gain is still clearly above 1.0.")
    print("Then raise --lr with it: a bigger batch means fewer, better-aimed steps.")
    return 0


# ------------------------------------------------------------- what: memory

def run_memory(args, clips, feat_dim, has_feats, device) -> int:
    """Allocated memory at each stage, then PyTorch's per-operator breakdown, so
    the largest consumer is named rather than inferred. Counting tensors by hand
    does not work here: F.scaled_dot_product_attention uses FlashAttention, which
    recomputes the attention matrix in the backward pass instead of storing it.
    """
    torch.cuda.reset_peak_memory_stats()
    model, opt, loader, amp = build(args, clips, feat_dim, has_feats, device,
                                    args.batch, args.depth, args.points, workers=4)
    print(f"amp {'bf16' if amp else 'off'}")
    print(f"  after model                 {gb():7.2f} GB")

    it = iter(loader)
    for phase in ("warmup", "measured"):
        kw = unpack(next(it), device, amp)
        ctx = kw["context"]
        if phase == "measured" and ctx is not None:
            print(f"  after batch on device       {gb():7.2f} GB"
                  f"   (context {ctx.numel() * ctx.element_size() / GB:.2f} GB"
                  f", {tuple(ctx.shape)} {ctx.dtype})")
        step(model, opt, kw, amp)
        if phase == "measured":
            print(f"  after step                  {gb():7.2f} GB"
                  f"   (AdamW keeps 2 states per parameter)")
            print(f"\n  PEAK                        "
                  f"{torch.cuda.max_memory_allocated() / GB:7.2f} GB")

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=True) as prof:
        step(model, opt, unpack(next(it), device, amp), amp)

    print("\ntop operators by CUDA memory allocated:\n")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))
    if args.trace:
        prof.export_chrome_trace(str(args.trace))
        print(f"\nwrote {args.trace}")
    return 0


# -------------------------------------------------------------- what: sweep

def run_sweep(args, clips, feat_dim, has_feats, device) -> int:
    """Halve one dimension at a time; whatever memory follows is the driver.

    More direct than cumulative allocation totals, which count every temporary
    ever made rather than what is resident at the peak.
    """
    base = dict(batch=args.batch, depth=args.depth, points=args.points, patches=576)
    print(f"{'config':<28} {'peak GB':>9} {'vs base':>9}")
    ref = peak_for(args, clips, feat_dim, has_feats, device, **base)
    print(f"{'base ' + str(base):<28} {ref:>9.2f} {'--':>9}")

    for key in base:
        cfg = dict(base, **{key: max(1, base[key] // 2)})
        try:
            pk = peak_for(args, clips, feat_dim, has_feats, device, **cfg)
        except torch.OutOfMemoryError:
            print(f"{'half ' + key:<28}   OOM")
            torch.cuda.empty_cache()
            continue
        print(f"{'half ' + key + f' -> {cfg[key]}':<28} {pk:>9.2f} {pk / ref:>8.2f}x")

    print("\n~0.50x means memory is linear in that dimension and it is a driver.")
    print("~1.00x means it is not where the memory is.")
    return 0


WHAT = {"speed": run_speed, "memory": run_memory, "sweep": run_sweep}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--what", choices=sorted(WHAT), default="speed")
    p.add_argument("--batch", type=int, default=16, help="memory and sweep")
    p.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256],
                   help="speed only")
    p.add_argument("--workers", type=int, nargs="+", default=[4], help="speed only")
    p.add_argument("--points", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--trace", type=Path, default=None, help="memory only: chrome trace")
    # A median over 10 steps is already stable, and when a step takes ten seconds
    # the extra samples cost more than the precision is worth.
    p.add_argument("--steps", type=int, default=10, help="timed steps per setting")
    p.add_argument("--warmup", type=int, default=3, help="untimed steps first")
    args = p.parse_args()

    if args.what in ("memory", "sweep") and not torch.cuda.is_available():
        raise SystemExit(f"--what {args.what} needs a GPU -- submit this as a job")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clips, _, feat_dim, has_feats = split(args.cache, 0.02, 0)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"{dev} | {len(clips)} clips | points {args.points} | "
          f"dim {args.dim} depth {args.depth}\n", flush=True)
    return WHAT[args.what](args, clips, feat_dim, has_feats, device)


if __name__ == "__main__":
    raise SystemExit(main())
