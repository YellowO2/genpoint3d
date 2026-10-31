"""A textured ball that spins, rolls or slides over a textured floor: a clean
test of tracking points on a turning surface.

Ray-traced in plain PyTorch, so every pixel's surface point, and every tracked
point's true position, comes from the motion itself: no physics engine, no
renderer to trust. The camera is fixed (OpenCV axes: x right, y down, z
forward) and looks down at the floor at --pitch degrees. The ball's pattern is
smooth colour waves in its own frame, so they turn with it; the floor's is a
checker under colour noise.

Each clip, written in TAPVid-3D's format so `scripts/tapvid3d_cache.py` reads
it: images (T, H, W, 3) uint8, fx_fy_cx_cy, tracks_XYZ (T, N, 3) metres in the
camera, visibility (T, N), queries_xyt (N, 3) all at frame 0; and beside it
`<clip>.depth.npy`, the true z-depth (T, H, W). The tracked points are points
of the ball visible in frame 0 and a few of the floor, so `points` are mostly
on the turning surface.

  python scripts/render_ball.py --out DIR [--seeds 3]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

H = W = 512
T = 24
F = 560.0
R_BALL = 0.6

# name: (spin axis in the ball's start frame, degrees per frame, velocity m/frame (x, z))
MOTIONS = {
    "still": ((0, 1, 0), 0.0, (0.0, 0.0)),
    "slide": ((0, 1, 0), 0.0, (0.06, 0.0)),
    "spin_up_5": ((0, 1, 0), 5.0, (0.0, 0.0)),
    "spin_up_10": ((0, 1, 0), 10.0, (0.0, 0.0)),
    "spin_up_20": ((0, 1, 0), 20.0, (0.0, 0.0)),
    "spin_side_10": ((1, 0, 0), 10.0, (0.0, 0.0)),
    "spin_side_20": ((1, 0, 0), 20.0, (0.0, 0.0)),
    "roll": ((0, 0, 1), None, (0.06, 0.0)),     # turns as it travels: no slipping
}


def rotation(axis, angle):
    a = torch.tensor(axis, dtype=torch.float64)
    a = a / a.norm()
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def ball_colour(local, phases):
    """(..., 3) points in the ball's own frame -> (..., 3) colours in [0, 1]."""
    out = []
    for c in range(3):
        v = sum(torch.sin(local @ phases[c, k, :3] + phases[c, k, 3]) for k in range(phases.shape[1]))
        out.append(0.5 + 0.5 * torch.tanh(v))
    return torch.stack(out, -1)


def floor_colour(pts, up, noise):
    """Points on the floor -> a checker of 0.5 m squares under colour noise."""
    e1 = torch.tensor([1.0, 0, 0], dtype=torch.float64)
    e2 = torch.linalg.cross(up, e1)
    a, b = pts @ e1, pts @ e2
    check = ((torch.floor(a / 0.5) + torch.floor(b / 0.5)) % 2)[..., None]
    wave = torch.stack([torch.sin(a * noise[c, 0] + b * noise[c, 1] + noise[c, 2]) for c in range(3)], -1)
    return (0.35 + 0.3 * check + 0.15 * wave).clamp(0, 1)


def render(name, axis, deg, vel, seed, out):
    g = torch.Generator().manual_seed(seed)
    pitch = math.radians(40)
    up = torch.tensor([0.0, -math.cos(pitch), -math.sin(pitch)], dtype=torch.float64)   # floor normal, camera axes
    fwd = torch.linalg.cross(up, torch.tensor([1.0, 0, 0], dtype=torch.float64))        # along the floor, away
    floor_point = torch.tensor([0.0, 0, 3.6], dtype=torch.float64)
    floor_point = floor_point - (floor_point @ up) * up - 2.2 * up    # the floor 2.2 m below the camera
    start = floor_point + R_BALL * up - 1.2 * torch.tensor([1.0, 0, 0], dtype=torch.float64) * (vel[0] > 0)
    # Waves of several sizes, down to a few centimetres, so a patch of the
    # ball has detail a matcher can lock onto, as Kubric's scanned textures do.
    scale = torch.tensor([3.0, 8.0, 20.0, 45.0], dtype=torch.float64)[None, :, None]
    phases = torch.rand(3, 4, 4, generator=g, dtype=torch.float64) * 2 - 1
    phases = torch.cat([phases[..., :3] * scale, phases[..., 3:] * 3.14], -1)
    noise = torch.rand(3, 3, generator=g, dtype=torch.float64) * 4
    v = vel[0] * torch.tensor([1.0, 0, 0], dtype=torch.float64) + vel[1] * fwd
    if deg is None:      # rolling: the turn the travel makes
        spin_axis = torch.linalg.cross(up, v / v.norm())
        step = v.norm().item() / R_BALL
    else:
        spin_axis = torch.tensor(axis, dtype=torch.float64)
        step = math.radians(deg)
    centres = [start + v * t for t in range(T)]
    rots = [rotation(spin_axis.tolist(), step * t) for t in range(T)]

    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64) + 0.5, torch.arange(W, dtype=torch.float64) + 0.5, indexing="ij")
    rays = torch.stack([(xs - W / 2) / F, (ys - H / 2) / F, torch.ones_like(xs)], -1)
    rays = rays / rays.norm(dim=-1, keepdim=True)
    frames, depths = [], []
    for c, R in zip(centres, rots):
        b = rays @ c
        disc = b ** 2 - (c @ c - R_BALL ** 2)
        t_ball = torch.where(disc > 0, b - disc.clamp_min(0).sqrt(), torch.full_like(b, float("inf")))
        denom = rays @ up
        t_floor = ((floor_point @ up) / denom).where(denom.abs() > 1e-9, torch.full_like(denom, float("inf")))
        t_floor = t_floor.where(t_floor > 0, torch.full_like(t_floor, float("inf")))
        hit_ball = t_ball < t_floor
        t_hit = torch.minimum(t_ball, t_floor)
        pts = rays * t_hit[..., None]
        local = (pts - c) @ R                         # into the ball's own frame
        shade = (0.55 + 0.45 * ((pts - c) / R_BALL @ torch.tensor([0.3, -0.8, -0.5], dtype=torch.float64)).clamp_min(0))[..., None]
        col = torch.where(hit_ball[..., None], ball_colour(local, phases) * shade, floor_colour(pts, up, noise))
        col = torch.where(torch.isfinite(t_hit)[..., None], col, torch.full_like(col, 0.7))
        frames.append((col.clamp(0, 1) * 255).round().byte())
        depths.append(torch.where(torch.isfinite(t_hit), pts[..., 2], torch.full_like(t_hit, 50.0)).float())

    # Points of the ball facing the camera in frame 0, and some of the floor.
    n = torch.randn(4000, 3, generator=g, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)
    p0 = centres[0] + R_BALL * n
    facing = ((-p0) * n).sum(-1) > 0.15 * p0.norm(dim=-1)
    local_pts = (R_BALL * n[facing])[:300] @ rots[0]
    floor_uv = torch.rand(60, 2, generator=g, dtype=torch.float64) * torch.tensor([6.0, 6.0], dtype=torch.float64) - torch.tensor([3.0, 0.0], dtype=torch.float64)
    floor_pts = floor_point + floor_uv[:, :1] * torch.tensor([1.0, 0, 0], dtype=torch.float64) + floor_uv[:, 1:] * fwd
    tracks, visible = [], []
    for c, R in zip(centres, rots):
        on_ball = c + local_pts @ R.T
        normal = (on_ball - c) / R_BALL
        pts = torch.cat([on_ball, floor_pts])
        seen_ball = (normal * (-on_ball)).sum(-1) > 0
        # A floor point is hidden if the ray to it meets the ball first.
        d = floor_pts / floor_pts.norm(dim=-1, keepdim=True)
        b = d @ c
        disc = b ** 2 - (c @ c - R_BALL ** 2)
        blocked = (disc > 0) & ((b - disc.clamp_min(0).sqrt()) < floor_pts.norm(dim=-1))
        uv = pts[:, :2] / pts[:, 2:] * F + W / 2
        inside = (uv >= 0).all(-1) & (uv < W).all(-1) & (pts[:, 2] > 0)
        tracks.append(pts)
        visible.append(torch.cat([seen_ball, ~blocked]) & inside)
    tracks, visible = torch.stack(tracks).float(), torch.stack(visible)
    keep = visible[0]
    tracks, visible = tracks[:, keep], visible[:, keep]
    uv0 = tracks[0, :, :2] / tracks[0, :, 2:] * F + W / 2
    queries = torch.cat([uv0, torch.zeros(len(uv0), 1)], -1)
    stem = f"{name}_{seed}"
    # The ball's centre per frame, and how many of the points are the ball's
    # (the first ones), for checks that need to know which surface is which.
    np.savez(out / f"{stem}.npz", images=torch.stack(frames).numpy(), fx_fy_cx_cy=np.array([F, F, W / 2, H / 2]),
             tracks_XYZ=tracks.numpy(), visibility=visible.numpy(), queries_xyt=queries.numpy(),
             centres=torch.stack(centres).float().numpy(), ball_points=int(keep[:len(local_pts)].sum()))
    np.save(out / f"{stem}.depth.npy", torch.stack(depths).numpy().astype(np.float16))
    turned = math.degrees(step * (T - 1))
    print(f"  {stem}: {int(keep[:len(local_pts)].sum())} ball + {int(keep[len(local_pts):].sum())} floor points,"
          f" turns {turned:.0f} deg in all", flush=True)


p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--seeds", type=int, default=3)
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for name, (axis, deg, vel) in MOTIONS.items():
    for seed in range(args.seeds):
        render(name, axis, deg, vel, seed, out)
