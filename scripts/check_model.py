"""The model and its loops at toy size, on the CPU in seconds: one training
step reaches every weight, and a forecast is unchanged by anything in the
frames it has no image of. Run after any change to the model or the loops:

  .venv/bin/python scripts/check_model.py
"""

import types

import torch

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.learn import learn
from genpoint3d.train.model.PointDiT import PointDiT
from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.ScoreSettings import ScoreSettings

torch.manual_seed(0)
B, T, N, P, F, R = 1, 12, 6, 16, 24, 64       # R: frame side, CNN map R/4


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


def model(**kw):
    return PointDiT(dim=32, depth=1, num_heads=2, mlp_mult=2, feat_dim=F, cross_window=4, fine_neigh=4, **kw)


for kw, sched in ((dict(), ""), (dict(visibility=True, hidden_blank=True, group_feedback=4,
                                      feedback_scale=10.0, bidirectional=True, rope_current=True), "fbfb")):
    m = model(**kw)
    for p in m.parameters():          # off the zero inits, so every reader is live
        torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else None
    s = types.SimpleNamespace(min_context=2, hidden_loss=1.0, vel_loss=1.0, rigid_loss=1.0,
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
    b2 = {k: v.clone() for k, v in b.items()}
    b2["context"][:, 6:] = torch.randn_like(b2["context"][:, 6:])
    b2["frames"][:, 6:] = torch.randint(0, 255, b2["frames"][:, 6:].shape, dtype=torch.uint8)
    b2["depth"][:, 6:] = 5.0
    b2["patch_xyz"][:, 6:] += 1.0
    pred2 = Predictor(m, j)(b2)
    leak = (pred.tracks - pred2.tracks).abs().max().item()
    print(f"  tracks {tuple(pred.tracks.shape)}, vis {None if pred.visibility is None else tuple(pred.visibility.shape)}, "
          f"forecast change when frames >= 6 change: {leak:.2e} (want 0)")
