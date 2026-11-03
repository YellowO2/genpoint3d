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
    """(B,) a cutoff per clip, uniform in [min_context, T] (T: pure tracking),
    drawn on the CPU so a seed gives the same cutoffs on any device."""
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


def apd_weight(b: dict, traj: torch.Tensor) -> torch.Tensor:
    """(B, T, N) weights that measure the error in APD threshold units, as
    TAPIP3D's `scale_loss_by_depth`: focal / depth, from each point's query
    frame (held to at least 10 cm), times the clip's scale, mean 1 over the batch."""
    q = b["query_t"]
    gt_cam = _to_camera_t(to_metres(traj, b), b["extrinsics"])         # (B, T, N, 3)
    depth = gt_cam[..., 2].gather(1, q[:, None]).clamp_min(0.1)        # (B, 1, N) at each query
    fx, fy = b["intrinsics"][..., 0, 0], b["intrinsics"][..., 1, 1]   # (B, T)
    focal = (fx * fy).sqrt().gather(1, q)[:, None]                     # (B, 1, N)
    w = focal / depth * b["norm_scale"][:, None, None]
    return (w / w.mean().clamp(min=1e-8)).expand_as(traj[..., 0])


def hidden_weight(vis: torch.Tensor, weight: torch.Tensor | None,
                  hidden: float) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The loss's `mask` and `weight` with a hidden frame counted `hidden`
    times a visible one; 0 counts visible frames only."""
    if not hidden:
        return vis, weight
    w = torch.where(vis, 1.0, hidden)
    return torch.ones_like(vis), w if weight is None else w * weight


def to_metres(pts: torch.Tensor, b: dict) -> torch.Tensor:
    """(B, T, N, 3) model units -> metres, each clip by its own normalisation."""
    ts = b["norm_traj_scale"][:, None, None, None]
    scale = b["norm_scale"][:, None, None, None]
    scene = pts * ts + b["norm_offset"]                    # undo the target offset
    return scene * scale + b["norm_mean"][:, None, None, :]


def rigid_pairs(b: dict) -> torch.Tensor:
    """(B, N, N) the pairs on one moving rigid object (`rigid_buddies`), for the rigidity loss."""
    return rigid_buddies(to_metres(b["traj"], b))


def rigid_groups(b: dict, tol: float = 0.02) -> torch.Tensor:
    """(B, N, N) the pairs whose distance never strays more than `tol` metres
    from frame 0's, moving or not: what the group vectors should put together."""
    p = to_metres(b["traj"], b)
    d = (p[:, :, :, None] - p[:, :, None]).norm(dim=-1)
    return (d - d[:, :1]).abs().amax(dim=1) < tol


def rigid_buddies(p: torch.Tensor, tol: float = 0.02, moved: float = 0.05) -> torch.Tensor:
    """(B, N, N) from true positions (B, T, N, 3) in metres: two different
    points whose distance never strays more than `tol` from frame 0's, both
    moving more than `moved` at some frame. Kubric objects are rigid, so this
    finds them without object ids; the moving part leaves the background out."""
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
    (B, T, 3, h, w) each cell's 3D position in `anchor`'s space. Cell j is
    centred at `(j + 1/8) / w` across, where `cache_frames.py` reads its depth."""
    B, T, h, w = depth.shape
    v = (torch.arange(h, device=depth.device, dtype=torch.float32) + 0.125) / h
    u = (torch.arange(w, device=depth.device, dtype=torch.float32) + 0.125) / w
    uv = torch.stack(torch.meshgrid(u, v, indexing="xy"), dim=-1).reshape(1, 1, -1, 2)
    xyz = unproject(uv.expand(B, T, -1, -1), depth.flatten(2), proj.float())
    return xyz.transpose(2, 3).reshape(B, T, 3, h, w)
