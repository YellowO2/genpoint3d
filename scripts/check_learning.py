"""
Can the model LEARN to read the image? Two small training runs, on CPU.

`check_model.py` asks whether the wiring is there. This asks the question that
went unasked for fourteen runs: with the wiring there, does training actually
pick it up, in a few hundred steps, on a task whose answer is in the image and
nowhere else? Each task is trained three times -- the current model, the model
as it was (`legacy`), and the model with the mechanism removed (`off`) -- and
passes only if the current one ends far below the one without.

  1. read the patch    The answer is written in the patch under the point: the
                       target displacement at frame t is three feature channels
                       of the patch nearest the point. Needs the locality prior
                       and nothing else (correlation is off).
     proves            a prior one patch spacing wide lets cross-attention read
                       the right patch, at a realistic spacing of 0.05, and
                       that the old prior was no better than none.
     does not prove    anything about a point whose current estimate is WRONG.
                       The target scale here is 0.005, so the noisy sample
                       stays within a tenth of a patch of the truth. In a run it
                       is 0.0992 -- two patches of noise at k = 0 -- and the
                       prior is centred on that noisy estimate.

  2. follow the match  The point's own feature reappears, in each later frame,
                       in a patch up to three cells from where it started, among
                       patches of noise. The target is the displacement to that
                       patch, at the run's real scale (0.0992, spacing 0.05).
                       Needs the correlation; locality is on for all three.
     proves            the match offset reaches the output: the model can turn
                       "the patch that looks like me is over there" into a
                       displacement, which the old matching never did.
     does not prove    that DINOv3 features match this cleanly. Here the right
                       patch has cosine 1 and the rest about 0; real features
                       of neighbouring patches are far more alike, and nothing
                       here has seen a real feature.

Reported for each: the flow-matching loss (mean of the last 100 steps, on fresh
data every step, so it is not memorised) and the error of sampled trajectories.

Run:  .venv/bin/python scripts/check_learning.py      (about five minutes)
"""

import time

import torch

from genpoint3d.data.transform import TRAJ_SCALE_DISP
from genpoint3d.models.flow import flow_matching_loss, sample
from genpoint3d.models.model import PointDiT

B, T, N, G, FEAT = 8, 4, 16, 12, 32
P = G * G
SPACING = 0.05
STEPS = 1000

_i = (torch.arange(G) - (G - 1) / 2) * SPACING
_ys, _xs = torch.meshgrid(_i, _i, indexing="ij")
GRID = torch.stack([_xs, _ys, torch.ones_like(_xs)], dim=-1).reshape(P, 3)   # row-major


def read_the_patch(batch: int = B):
    """x1[t, n] = the first three feature channels of the patch nearest point n."""
    home = torch.randint(P, (batch, N))
    anchor = GRID[home] + (torch.rand(batch, N, 3) - 0.5) * 0.8 * SPACING
    ctx = torch.randn(batch, T, P, FEAT)
    x1 = ctx[..., :3].gather(2, home[:, None, :, None].expand(batch, T, N, 3)).clone()
    x1[:, 0] = 0
    return x1, anchor, ctx, torch.randn(batch, N, FEAT)


def follow_the_match(batch: int = B):
    """x1[t, n] = the displacement to the one patch in frame t that holds point
    n's feature. Each frame's patch is drawn independently, so no frame's
    answer can be guessed from another's."""
    row, col = torch.randint(3, G - 3, (2, batch, 1, N))
    step = torch.randint(-3, 4, (2, batch, T, N))
    step[:, :, 0] = 0
    target = (row + step[0]) * G + col + step[1]                    # (batch, T, N)
    anchor = GRID[(row * G + col)[:, 0]] + (torch.rand(batch, N, 3) - 0.5) * 0.8 * SPACING
    idc = torch.randn(batch, N, FEAT)
    ctx = torch.randn(batch, T, P, FEAT)
    ctx.scatter_(2, target[..., None].expand(-1, -1, -1, FEAT), idc[:, None].expand(-1, T, -1, -1))
    x1 = (GRID[target] - anchor[:, None]) / TRAJ_SCALE_DISP
    x1[:, 0] = 0
    return x1, anchor, ctx, idc


def run(data, **kw) -> tuple[float, float]:
    """Train on `data` for STEPS; return (loss, sampled error in target units)."""
    torch.manual_seed(0)
    m = PointDiT(dim=64, depth=2, num_heads=4, cross_attn=True, feat_dim=FEAT,
                 costvol=False, displacement=True, **kw)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    pxyz = GRID.expand(B, T, P, 3)
    vm = torch.ones(B, T, dtype=torch.bool)
    pin = torch.zeros(B, N, 3)              # frame 0's displacement is known: zero
    tail = []
    for step in range(STEPS):
        x1, anchor, ctx, idc = data()
        loss, _ = flow_matching_loss(m, x1, anchor, known_x0=pin, loss_type="l21",
                                     context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        tail.append(loss.item())
    m.eval()
    x1, anchor, ctx, idc = data()
    out = sample(m, anchor, T, steps=20, known_x0=pin,
                 context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
    err = (out - x1)[:, 1:].norm(dim=-1).median().item()
    return sum(tail[-100:]) / 100, err


def compare(name: str, data, unit: str, per_unit: float, variants: dict) -> bool:
    print(f"{name}  ({STEPS} steps each)")
    got = {}
    for label, kw in variants.items():
        t0 = time.time()
        loss, err = run(data, **kw)
        got[label] = loss
        print(f"    {label:<7} loss {loss:.3f}   sampled error {err * per_unit:.2f} {unit}"
              f"   ({time.time() - t0:.0f}s)", flush=True)
    ok = got["new"] < 0.5 * got["off"]
    print(f"  {'PASS' if ok else 'FAIL'}  new {got['new']:.3f} vs off {got['off']:.3f}"
          f" (want under half), legacy {got['legacy']:.3f}\n", flush=True)
    return ok


def main() -> int:
    torch.set_num_threads(1)                # the same numbers on any machine
    ok = compare(
        "1. read the patch", read_the_patch, "target std", 1.0,
        {"new": dict(traj_scale=0.005, correlate=False),
         "legacy": dict(traj_scale=0.005, correlate=False, locality_mode="legacy"),
         "off": dict(traj_scale=0.005, correlate=False, locality=False)})
    ok &= compare(
        "2. follow the match", follow_the_match, "patch spacings", TRAJ_SCALE_DISP / SPACING,
        {"new": dict(traj_scale=TRAJ_SCALE_DISP),
         "legacy": dict(traj_scale=TRAJ_SCALE_DISP, corr_mode="legacy"),
         "off": dict(traj_scale=TRAJ_SCALE_DISP, correlate=False)})
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
