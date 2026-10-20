"""
Check that forecasting leaks nothing: with a cutoff, the model's prediction
for EVERY frame is bit-identical whatever the frames from the cutoff on hold.

  1. blind      every input of frames >= cutoff replaced by noise -- DINOv3
                features, patch positions, RGB frames, depth, cameras -- and the
                tracks, the loss, the match loss and the visibility head's
                logits are bit-identical
  2. sees       the same change on one observed frame does move the tracks,
                so the check is not passing because nothing is read

Run through the real scoring path (`Predictor`: `to_device`, `fine_inputs`,
`visual_mask`, the sampler or the refiner) on a real cached batch, for flow
and regress, with and without the relative finder (and its depth), with the
match trained and the visibility head on, and fed back.

Every weight is pushed off its initial value first. Several readers start at
exactly zero (the output head, the cost volume's gate, the finder's readers)
so that step 0 is the model without them, and a check on such a model would
pass because nothing is read at all.

Run:  .venv/bin/python scripts/check_forecast.py [--cache local/cache/stub2]
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import match_targets, visual_mask
from genpoint3d.train.tools.matching import match_ce

CUT = 5     # frames [0, CUT) keep their images; the rest are forecast
RES = 64    # side of the random RGB frames the finder is given
CONFIGS = [
    ["--method", "flow"],
    ["--method", "regress"],
    ["--method", "flow", "--fine", "1"],
    ["--method", "regress", "--fine", "1"],
    # The other ways a frame's inputs reach the model: more candidates, an
    # upsampled grid, and attention to later frames.
    ["--method", "flow", "--fine", "1", "--match-topk", "3", "--upsample", "2",
     "--causal", "0"],
    # The finder reading each cell's 3D position from the depth.
    ["--method", "flow", "--fine", "1", "--fine-xyz", "1"],
    ["--method", "regress", "--fine", "1", "--fine-xyz", "1", "--causal", "0"],
    # Each iteration given the visibility the one before it predicted.
    ["--method", "regress", "--vis-feedback", "1"],
    ["--method", "regress", "--fine", "1", "--fine-xyz", "1", "--causal", "0",
     "--vis-feedback", "1"],
    # Corrections summed along time into the displacement.
    ["--method", "regress", "--param", "acceleration"],
    ["--method", "regress", "--fine", "1", "--fine-xyz", "1", "--causal", "0",
     "--vis-feedback", "1", "--param", "acceleration"],
    # Each token also given how far its guess moved since the frame before.
    ["--method", "regress", "--param", "velocity", "--vel-input", "1"],
    # Each point told where the points that move with it say it is.
    ["--method", "regress", "--param", "velocity", "--buddies", "1", "--buddy-bias", "1"],
    ["--method", "regress", "--fine", "1", "--fine-xyz", "1", "--vis-loss", "1",
     "--vis-feedback", "1", "--param", "velocity", "--buddies", "1"],
    # A finer ruler for the finder's window offsets, and more room per level.
    ["--method", "regress", "--fine", "1", "--fine-xyz", "1", "--fine-waves", "10",
     "--fine-width", "128"],
]


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<8} {detail}")
    return ok


def batch_from(cache: Path, points: int) -> dict:
    """Two real clips, with random RGB frames and depth for the finder."""
    clips = sorted(cache.glob("*.pt"))[:2]
    b = next(iter(DataLoader(ClipDataset(clips, points, resample=False), batch_size=2)))
    torch.manual_seed(1)
    B, T = b["traj"].shape[:2]
    b["frames"] = torch.randint(0, 256, (B, T, RES, RES, 3), dtype=torch.uint8)
    b["depth"] = torch.rand(B, T, RES // 4, RES // 4) * 10 + 1
    return b


def changed(b: dict, frames: slice) -> dict:
    """The batch with every input of `frames` replaced by noise."""
    torch.manual_seed(2)
    c = {k: v.clone() for k, v in b.items()}
    ctx = c["context"][:, frames]
    c["context"][:, frames] = (torch.randn(ctx.shape) * 50).to(ctx.dtype)
    c["patch_xyz"][:, frames] = torch.randn_like(c["patch_xyz"][:, frames]) * 5
    c["frames"][:, frames] = torch.randint_like(c["frames"][:, frames], 0, 256)
    c["depth"][:, frames] = torch.rand_like(c["depth"][:, frames]) * 30 + 0.5
    c["intrinsics"][:, frames, :2] *= 1.5
    c["extrinsics"][:, frames, :3, 3] += torch.randn_like(c["extrinsics"][:, frames, :3, 3])
    return c


def run(model, b: dict, judging: ScoreSettings):
    """The tracks, the loss, the match loss of the training step and the
    visibility logits."""
    # The flow loss draws its noise level and noise from the global generator.
    torch.manual_seed(3)
    prediction = Predictor(model, judging)(b)
    pb = prediction.batch
    with torch.no_grad():
        scores, target, valid = match_targets(model, pb, prediction.visual_mask)
        match = match_ce(scores, model.match_head.tau(), target, valid).sum() / valid.sum()
    return prediction.tracks, prediction.loss, match, prediction.visibility


def check(flags: list[str], cache: Path, b: dict) -> bool:
    settings = Settings.from_command_line(
        ["--cache", str(cache), "--dim", "32", "--depth", "2", "--heads", "2",
         "--match-learn", "1", "--match-dim", "16", "--fine-levels", "2",
         "--refine-iters", "2", "--target", "displacement", "--vis-loss", "1",
         "--no-amp"] + flags)
    torch.manual_seed(0)
    model = build_model(settings, b["context"].shape[-1], True)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    judging = ScoreSettings(device=torch.device("cpu"), method=settings.method, steps=4,
                            refine_iters=settings.refine_iters, param=settings.param,
                            anchor_frame0=True,
                            loss_type=settings.loss_type, cutoff=CUT)
    print(" ".join(flags))
    base = run(model, b, judging)
    blind = run(model, changed(b, slice(CUT, None)), judging)
    sees = run(model, changed(b, slice(CUT - 1, CUT)), judging)
    same = [torch.equal(base[0], blind[0]), base[1] == blind[1], torch.equal(base[2], blind[2]),
            torch.equal(base[3], blind[3])]
    moved = (sees[0] - base[0]).abs().max().item()
    return all([
        report("blind", all(same),
               f"tracks / loss / match loss / visibility {'/'.join('same' if s else 'DIFFER' for s in same)}"
               f" with every input of frames >= {CUT} changed (want all same)"),
        report("sees", moved > 0,
               f"changing frame {CUT - 1}'s inputs moved the tracks {moved:.1e} (want >0)"),
    ])


def check_mask() -> bool:
    """The mask itself: frame 0 always kept, the cutoff where it says."""
    b = {"traj": torch.zeros(3, 6, 1, 3), "context": torch.zeros(1), "frames": None}
    vm = visual_mask(b, torch.tensor([1, 3, 6]))
    want = torch.tensor([[1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]]).bool()
    return report("mask", torch.equal(vm, want), "cutoffs 1, 3, 6 of 6 frames")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="local/cache/stub2",
                   help="a feature cache WITH patch positions, two clips are enough")
    cache = Path(p.parse_args().cache)
    b = batch_from(cache, points=16)
    print(f"cutoff {CUT} of {b['traj'].shape[1]} frames\n")
    ok = check_mask()
    for flags in CONFIGS:
        ok &= check(flags, cache, b)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
