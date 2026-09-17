"""
Verification for step 1 (`genpoint3d/data/transform.py`).

Four checks, each of which fails loudly if the geometry is wrong:

  1. round-trip     -- denormalise(normalise(x)) recovers the input
  2. static scene   -- unprojected geometry agrees across frames despite the
                       camera moving. This is the real test of the reframing.
  3. reprojection   -- projecting the 3D trajectory back into each frame
                       reproduces the dataset's own 2D pixel coordinates
  4. scale sanity   -- normalised values land in a trainable range

Run:  .venv/bin/python scripts/check_transform.py [data_root]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from genpoint3d.data.kubric import KubricSequenceDataset
from genpoint3d.data.transform import project_to_frames, scene_pointmap, transform

PASS, FAIL = "  PASS", "  FAIL"


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"{PASS if ok else FAIL}  {name:<22} {detail}")
    return ok


def check_roundtrip(inputs) -> bool:
    recovered = inputs.norm.invert(inputs.norm.apply(inputs.traj_metric))
    err = (recovered - inputs.traj_metric).abs().max().item()
    return report("round-trip", err < 1e-3, f"max abs error {err:.2e} m")


def check_static_scene(inputs) -> bool:
    """A static world seen from a moving camera must unproject to the same points.

    Compares each frame's pointmap against frame 0 at pixels that are static
    across the clip. Uses the median so the moving objects in the scene (which
    legitimately differ) do not dominate.
    """
    pts = scene_pointmap(inputs, normalise=False)  # (T, 3, H, W)
    drift = (pts - pts[:1]).norm(dim=1)            # (T, H, W) metres
    per_frame = drift.flatten(1).median(dim=1).values
    worst = per_frame.max().item()
    extent = (pts[0].flatten(1).max(dim=1).values - pts[0].flatten(1).min(dim=1).values).norm().item()
    rel = worst / max(extent, 1e-6)
    return report(
        "static scene",
        rel < 0.02,
        f"median drift {worst:.4f} m = {100 * rel:.2f}% of scene extent {extent:.2f} m",
    )


def check_reprojection(inputs, sample) -> bool:
    """3D -> 2D must reproduce the dataset's own pixel coordinates."""
    uv = project_to_frames(inputs, inputs.traj_metric)          # (T, N, 2)
    gt = torch.from_numpy(sample.coords_2d).float()             # (T, N, 2)
    vis = inputs.visibility                                     # ignore off-screen points
    err = (uv - gt).norm(dim=-1)[vis]
    med = err.median().item()
    return report("reprojection", med < 1.0, f"median error {med:.3f} px over {vis.sum()} visible samples")


def check_scale(inputs) -> bool:
    """Geometry should be O(1); the *denoising target* must be near unit variance.

    Flow matching mixes the target with `x0 ~ N(0, I)`. A target with std 0.26
    (what scene normalisation alone gives) makes `x_k` noise-dominated, so the
    model drifts toward just outputting `-x0`. This check is why `TRAJ_SCALE`
    exists.
    """
    geom = inputs.norm.apply(inputs.traj_metric)
    target_std = inputs.traj.std().item()
    ok = geom.abs().max().item() < 50.0 and 0.3 < target_std < 3.0
    return report(
        "scale sanity",
        ok,
        f"geometry |max| {geom.abs().max():.2f}; TARGET std {target_std:.3f} (want ~1)",
    )


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "local/data/kubric_test"
    ds = KubricSequenceDataset(root, num_query_points=256)
    print(f"{len(ds)} sequence(s) in {root}\n")

    all_ok = True
    for i in range(len(ds)):
        sample = ds[i]
        T = sample.frames.shape[0]
        inputs = transform(sample, num_context_frames=max(1, T // 2))
        print(f"[{sample.seq_id}]  T={T}  N={sample.traj_3d.shape[1]}  T_C={inputs.num_context_frames}")
        all_ok &= check_roundtrip(inputs)
        all_ok &= check_static_scene(inputs)
        all_ok &= check_reprojection(inputs, sample)
        all_ok &= check_scale(inputs)
        print()

    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
