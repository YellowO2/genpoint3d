"""The model and its loops at toy size, on the CPU in seconds: one training
step reaches every weight, and the tracks are unchanged by anything in the
frames or quarters of frames they have no image of: after a cutoff, in a
gap, or in a hidden quarter. Run after any change to the model or the loops:

  .venv/bin/python scripts/check_model.py
"""

import types

import torch

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.learn import learn
from genpoint3d.train.model.PointDiT import PointDiT
from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.score.track_scores import track_scores
from genpoint3d.train.score.visibility_scores import visibility_scores
from genpoint3d.train.tools.batch import black_out, fine_inputs, to_device
from genpoint3d.train.tools.regress import refine

torch.manual_seed(0)
B, T, N, P, F, R = 1, 16, 6, 16, 24, 64       # R: frame side, CNN map R/4; windows of 8 sharing 4: three


def batch():
    traj = torch.randn(B, T, N, 3) * 0.1
    traj[:, 0] = 0
    E = torch.eye(4).expand(B, T, 4, 4).clone()
    K = torch.tensor([[R, 0, R / 2], [0, R, R / 2], [0, 0, 1.0]]).expand(B, T, 3, 3).clone()
    anchor = torch.randn(B, N, 3) * 0.2 + torch.tensor([0, 0, 3.0])
    return {
        "traj": traj, "anchor": anchor, "visibility": torch.rand(B, T, N) > 0.3,
        "query_t": torch.tensor([[0, 0, 0, 2, 5, 9]]),
        "context": torch.randn(B, T, P, F).half(), "id_card": torch.randn(B, N, F).half(),
        "patch_xyz": torch.randn(B, T, P, 3) * 0.3 + torch.tensor([0, 0, 3.0]),
        "frames": torch.randint(0, 255, (B, T, R, R, 3), dtype=torch.uint8),
        "depth": torch.full((B, T, R // 4, R // 4), 3.0),
        "intrinsics": K, "intrinsics_256": K, "extrinsics": E, "hw": torch.tensor([[R, R]]),
        "norm_mean": torch.zeros(B, 3), "norm_scale": torch.ones(B), "norm_traj_scale": torch.full((B,), 0.1),
        "norm_offset": anchor[:, None].clone(),
    }


def changed(b: dict, frames: slice) -> dict:
    """`b` with everything the images give in `frames` replaced."""
    b2 = {k: v.clone() for k, v in b.items()}
    b2["context"][:, frames] = torch.randn_like(b2["context"][:, frames])
    b2["frames"][:, frames] = torch.randint(0, 255, b2["frames"][:, frames].shape, dtype=torch.uint8)
    b2["depth"][:, frames] = 5.0
    b2["patch_xyz"][:, frames] += 1.0
    return b2


def masked_tracks(m, b: dict, sched: str, view: torch.Tensor) -> torch.Tensor:
    """The tracks with `view` (B, T, 2, 2) the quarters seen, the hidden ones blacked out."""
    b = to_device(b, torch.device("cpu"))
    b["frames"] = black_out(b["frames"], view)
    cond = dict(context=b["context"], visual_mask=view, id_card=b["id_card"], patch_xyz=b["patch_xyz"],
                **fine_inputs(m, b))
    return refine(m, b["anchor"], T, iters=2, window=8, overlap=4, schedule=sched, **cond)[0]


def model(**kw):
    return PointDiT(dim=32, depth=1, num_heads=2, mlp_mult=2, feat_dim=F, cross_window=4, fine_neigh=4, **kw)


full = dict(visibility=True, group_feedback=4, rigid_proposal=True, feedback_scale=10.0, bidirectional=True,
            rope_current=True)
for kw, sched in ((dict(), ""), (full, "fbfb"), (full, "ffbb")):
    m = model(**kw)
    with torch.no_grad():             # off the zero inits, so every reader is live
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    s = types.SimpleNamespace(min_context=2, max_gap=4, partial_masks=1, hidden_loss=1.0, vel_loss=1.0, rigid_loss=1.0,
                              group_loss=1.0 if kw else 0.0, vis_loss=1.0 if kw else 0.0, refine_iters=2,
                              accum=1, window=8, overlap=4, schedule=sched)
    m.train()
    losses = learn(m, batch(), s, Hardware(torch.device("cpu"), False))
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    print(f"{sched or 'f x2'}: loss {losses.method:.4f} vis {losses.visibility:.4f}, "
          f"params without gradient: {dead}")
    j = ScoreSettings(device=torch.device("cpu"), refine_iters=2, window=8, overlap=4, schedule=sched, cutoff=6)
    b = batch()
    pred = Predictor(m, j)(b)
    b2 = changed(b, slice(6, None))
    pred2 = Predictor(m, j)(b2)
    leak = (pred.tracks - pred2.tracks).abs().max().item()
    print(f"  tracks {tuple(pred.tracks.shape)}, vis {None if pred.visibility is None else tuple(pred.visibility.shape)}, "
          f"forecast change when frames >= 6 change: {leak:.2e} (want 0)")
    scores = {**track_scores(pred, j), **visibility_scores(pred)}
    print(f"  {len(scores)} scores: {' '.join(sorted(scores))}")
    view = torch.ones(B, T, 2, 2, dtype=torch.bool)
    view[:, 5:8] = False
    gap = (masked_tracks(m, b, sched, view) - masked_tracks(m, changed(b, slice(5, 8)), sched, view)).abs().max().item()
    print(f"  change anywhere when the frames of a gap (5-7) change: {gap:.2e} (want 0)")
    view = torch.ones(B, T, 2, 2, dtype=torch.bool)
    view[:, 5:8, 0, 0] = False
    b2 = {k: v.clone() for k, v in b.items()}
    b2["frames"][:, 5:8, :R // 2, :R // 2] = torch.randint(0, 255, b2["frames"][:, 5:8, :R // 2, :R // 2].shape,
                                                           dtype=torch.uint8)
    b2["depth"][:, 5:8, :R // 8, :R // 8] = 5.0
    b2["context"][:, 5:8] = torch.randn_like(b2["context"][:, 5:8])      # no DINOv3 in a partial frame
    b2["id_card"][:, 4] = torch.randn_like(b2["id_card"][:, 4])          # its point, first seen at 5, is never handed over
    part = (masked_tracks(m, b, sched, view) - masked_tracks(m, b2, sched, view)).abs().max().item()
    seen = (masked_tracks(m, b, sched, view) - masked_tracks(m, changed(b, slice(5, 8)), sched, view)).abs().max().item()
    print(f"  top-left quarter of 5-7 hidden: change when it changes {part:.2e} (want 0), "
          f"when the rest of those frames does {seen:.2e} (want > 0)")
