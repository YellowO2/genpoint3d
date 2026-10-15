"""What every step does to a batch: move it, unscale it, weigh it, read it."""

import math

import torch

from genpoint3d.train.tools.camera import image_projection
from genpoint3d.train.tools.matching import true_patch
from genpoint3d.train.tools.metrics import _to_camera_t

# TAPIP3D clamps depth the same way before dividing, so a point behind the
# camera or at zero depth cannot produce an enormous weight.
DEPTH_MIN = 0.1


def to_device(batch: dict, device, amp: bool = False) -> dict:
    """Move a batch, and turn the empty placeholder tensors back into None.

    Cached features are fp16 and are widened here. Leaving them narrow looks
    like free memory under autocast, but autocast does not reach every module:
    AdaRMSNorm returns `.to(x.dtype)`, so a half tensor survives into a Linear
    holding fp32 weights and raises "mat1 and mat2 have different dtype".
    Revisit only with a measurement, not by reasoning about autocast.
    """
    b = {k: v.to(device) for k, v in batch.items()}
    for k in ("context", "id_card", "patch_xyz"):
        b[k] = None if b[k].numel() == 0 else b[k].float()
    if b["frames"].numel() == 0:
        b["frames"] = None
    return b


def known_frame0(b: dict) -> torch.Tensor:
    """(B, N, 3) frame 0 in model units.

    `anchor` is the same point in scene-normalised units, so this is available
    at inference exactly as it is in training -- no ground truth is used that a
    deployed model would not have.
    """
    # With a per-point displacement target frame 0 is exactly zero, which this
    # expression produces without special-casing: anchor IS each point's origin.
    return (b["anchor"] - b["norm_offset"][:, 0]) / b["norm_traj_scale"][:, None, None]


def apd_weight(b: dict, traj: torch.Tensor) -> torch.Tensor:
    """(B, T, N) weight that puts the loss in units of the APD threshold.

    Times the clip's own scale, because the error it multiplies is in model
    units -- metres divided by that scale -- and without it a clip twice as
    large had its points weighted half as much for the same pixel error.

    The metric allows a point an error of `thresh * depth / focal`, so a point
    20 m away is given ten times the slack of one 2 m away. An unweighted loss
    does not know that: it spends the same effort on a distant point that was
    going to pass anyway as on a near point that was always going to fail.

    Dividing the error by `depth / focal` measures it in threshold units
    instead, which is what TAPIP3D's `scale_loss_by_depth` does
    (`training/criterion.py:165`, "Normalize to match with the APD metric").

    Normalised to mean 1 over the batch so the loss keeps its magnitude and the
    learning rate stays comparable across runs -- only the RELATIVE weighting
    of points is the point here.
    """
    gt_cam = _to_camera_t(to_metres(traj, b), b["extrinsics"])
    depth = gt_cam[..., 2].abs().clamp(min=DEPTH_MIN)                  # (B, T, N)
    focal = (b["intrinsics"][..., 0, 0] * b["intrinsics"][..., 1, 1]).sqrt()
    w = focal[..., None] / depth * b["norm_scale"][:, None, None]
    return w / w.mean().clamp(min=1e-8)


def to_metres(pts: torch.Tensor, b: dict) -> torch.Tensor:
    """(B, T, N, 3) model units -> metres, using each clip's own normalisation.

    The inverse of `NormStats.invert_traj`, batched: every clip in the batch has
    its own scale, which is exactly why a threshold in model units meant a
    different physical distance per clip.
    """
    ts = b["norm_traj_scale"][:, None, None, None]
    scale = b["norm_scale"][:, None, None, None]
    scene = pts * ts + b["norm_offset"]                    # undo the target offset
    return scene * scale + b["norm_mean"][:, None, None, :]


def fine_inputs(model, b: dict, amp: bool = False, grad: bool = False) -> dict:
    """What the relative finder adds to a forward pass: the CNN's feature map
    for the batch's frames, and each frame's camera. Empty for a model without
    one, or a batch without frames.

    The map is computed once and shared by every forward pass of the step. It
    comes back with no graph; with `grad` it is a leaf that collects the
    gradient of each backward pass, for `FineFinder.replay` to carry into the
    CNN once they are all done.
    """
    if model.fine is None or b["frames"] is None:
        return {}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        fmap = model.fine.encode(b["frames"])
    return {"fine_map": fmap.requires_grad_(grad),
            "fine_proj": image_projection(b["intrinsics"], b["extrinsics"], b["hw"],
                                          b["norm_scale"], b["norm_mean"])}


def match_targets(model, b: dict, visual_mask: torch.Tensor):
    """The match as a classification: scores, the right answer, and where
    there is one. None for a model or a batch with no match to score.

    scores  (B, T, N, P) from `PointDiT.match_scores`, with their graph
    target  (B, T, N)    the patch the ground truth is drawn in
    valid   (B, T, N)    the point is visible and inside the frame, the frame
                         has its image, and it is not frame 0

    Frame 0 is left out. The template is read from frame 0 at the point's own
    pixel, so the match there is right by construction: it would add nothing to
    the loss and would lift the accuracy by a share that says nothing about
    finding the point again.

    The target comes from the batch's metres and cameras, so it does not
    depend on `--target` or the normalisation.
    """
    if b["context"] is None or b["id_card"] is None or b["patch_xyz"] is None:
        return None
    scores = model.match_scores(b["context"], b["id_card"], b["patch_xyz"])
    if scores is None:
        return None
    target, inside = true_patch(to_metres(b["traj"], b), b["intrinsics"], b["extrinsics"],
                                b["hw"], math.isqrt(scores.shape[-1]))
    valid = b["visibility"] & inside & visual_mask[..., None]
    valid[:, 0] = False
    return scores, target, valid
