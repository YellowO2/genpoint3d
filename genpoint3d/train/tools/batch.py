"""What every step does to a batch: move it, unscale it, weigh it, read it."""

import torch

from genpoint3d.train.tools.camera import image_projection, unproject
from genpoint3d.train.tools.metrics import _to_camera_t



def to_device(batch: dict, device) -> dict:
    """Move a batch, the fp16-cached features widened to fp32: autocast does
    not reach every module, and a half tensor meeting fp32 weights raises."""
    b = {k: v.to(device) for k, v in batch.items()}
    for k in ("context", "id_card", "patch_xyz"):
        b[k] = b[k].float()
    return b


def random_cutoff(b: dict, min_context: int) -> torch.Tensor:
    """(B,) a cutoff per clip, uniform in [min_context, T], as Gen-points
    trains: T is pure tracking, anything less forecasts the rest.

    Drawn on the CPU from torch's global generator, so a seeded run draws the
    same cutoffs whatever the device."""
    B, T = b["traj"].shape[:2]
    if not 1 <= min_context <= T:
        raise ValueError(f"--min-context {min_context} must be in [1, {T}]: "
                         "frame 0 is the query and always keeps its image")
    return torch.randint(min_context, T + 1, (B,)).to(b["traj"].device)


def visual_mask(b: dict, cutoff: torch.Tensor | None) -> torch.Tensor:
    """(B, T) True where a frame keeps its image: the frames before each
    clip's cutoff, every frame without one."""
    B, T = b["traj"].shape[:2]
    device = b["traj"].device
    if cutoff is None:
        return torch.ones(B, T, dtype=torch.bool, device=device)
    if not (cutoff >= 1).all():
        raise ValueError("a cutoff below 1 would hide frame 0, the query frame")
    return torch.arange(T, device=device)[None] < cutoff[:, None]


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

    One weight per point, from its query frame, kept for all its frames: how
    clearly the point was seen when it was picked. Every point is visible and
    in front of the camera there. Another frame's depth is not: off screen a
    point can sit beside or behind the camera, where `depth / focal` goes to
    zero and one hidden point would outweigh the rest of the batch. Held to at
    least 10 cm, as TAPIP3D holds it.

    Normalised to mean 1 over the batch so the loss keeps its magnitude and the
    learning rate stays comparable across runs -- only the RELATIVE weighting
    of points is the point here.
    """
    q = b.get("query_t")
    if q is None:
        q = torch.zeros(traj.shape[0], traj.shape[2], dtype=torch.long, device=traj.device)
    gt_cam = _to_camera_t(to_metres(traj, b), b["extrinsics"])         # (B, T, N, 3)
    depth = gt_cam[..., 2].gather(1, q[:, None]).clamp_min(0.1)        # (B, 1, N) at each query
    fx, fy = b["intrinsics"][..., 0, 0], b["intrinsics"][..., 1, 1]   # (B, T)
    focal = (fx * fy).sqrt().gather(1, q)[:, None]                     # (B, 1, N)
    w = focal / depth * b["norm_scale"][:, None, None]
    return (w / w.mean().clamp(min=1e-8)).expand_as(traj[..., 0])


def hidden_weight(vis: torch.Tensor, weight: torch.Tensor | None,
                  hidden: float) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The loss's `mask` and `weight` with hidden frames counted at `hidden`
    times a visible one. 0 is visible frames only: `vis` and `weight` as given.

    The loss divides by the sum of the weights it counts, so at 1 every frame
    counts the same and the loss is the plain mean over all of them.
    """
    if not hidden:
        return vis, weight
    w = torch.where(vis, 1.0, hidden)
    return torch.ones_like(vis), w if weight is None else w * weight


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


def rigid_pairs(b: dict) -> torch.Tensor:
    """(B, N, N) bool: the pairs of points the truth shows on one moving rigid
    object, for the rigidity loss. See `rigid_buddies`."""
    return rigid_buddies(to_metres(b["traj"], b))


def rigid_groups(b: dict, tol: float = 0.02) -> torch.Tensor:
    """(B, N, N) bool: the pairs of points whose distance never strays more
    than `tol` metres from its frame-0 value, moving or not -- the pairs that
    move as one, which a group-feedback model's vectors should put together. Two
    still points count: they move alike. A point pairs with itself."""
    p = to_metres(b["traj"], b)
    d = (p[:, :, :, None] - p[:, :, None]).norm(dim=-1)
    return (d - d[:, :1]).abs().amax(dim=1) < tol


def rigid_buddies(p: torch.Tensor, tol: float = 0.02, moved: float = 0.05) -> torch.Tensor:
    """(B, N, N) bool from true positions `p` (B, T, N, 3) in metres: the pairs
    of points on one moving rigid object.

    The cache keeps no object ids, but every Kubric object is rigid: two
    points on one keep their distance in every frame, the hidden ones too.
    A pair counts when that distance never strays more than `tol` metres from
    its frame-0 value and both points move more than `moved` metres at some
    frame -- two points of the still background keep their distance as well,
    and say nothing about motion. A point is never paired with itself.
    """
    d = (p[:, :, :, None] - p[:, :, None]).norm(dim=-1)          # (B, T, N, N)
    rigid = (d - d[:, :1]).abs().amax(dim=1) < tol
    move = (p - p[:, :1]).norm(dim=-1).amax(dim=1) > moved       # (B, N)
    eye = torch.eye(p.shape[2], dtype=torch.bool, device=p.device)
    return rigid & move[:, :, None] & move[:, None, :] & ~eye


def fine_inputs(model, b: dict, amp: bool = False, grad: bool = False) -> dict:
    """The matcher's inputs: the CNN's map of the batch's frames, each frame's
    camera, each cell's 3D position and the query frames. The map has no
    graph; with `grad` it is a leaf that gathers every backward pass's
    gradient for `FineFinder.replay`."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        fmap = model.fine.encode(b["frames"])
    proj = image_projection(b["intrinsics"], b["extrinsics"], b["hw"], b["norm_scale"], b["norm_mean"])
    return {"fine_map": fmap.requires_grad_(grad), "fine_proj": proj,
            "fine_xyz": _cell_xyz(b["depth"].float(), proj), "query_t": b["query_t"]}


def _cell_xyz(depth: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """depth (B, T, h, w) metres at each CNN cell, proj (B, T, 3, 4) ->
    (B, T, 3, h, w): each cell's 3D position, scene-normalised like `anchor`.

    Cell j of a w-wide map is centred on pixel INDEX 4j of the frame, which is
    `(4j + 0.5) / W = (j + 1/8) / w` across (`FineFinder.py`'s docstring), and
    `scripts/cache_frames.py` reads the depth at the same spot.
    """
    B, T, h, w = depth.shape
    v = (torch.arange(h, device=depth.device, dtype=torch.float32) + 0.125) / h
    u = (torch.arange(w, device=depth.device, dtype=torch.float32) + 0.125) / w
    uv = torch.stack(torch.meshgrid(u, v, indexing="xy"), dim=-1).reshape(1, 1, -1, 2)
    xyz = unproject(uv.expand(B, T, -1, -1), depth.flatten(2), proj.float())
    return xyz.transpose(2, 3).reshape(B, T, 3, h, w)
