"""What every step does to a batch: move it, unscale it, weigh it, read it."""

import torch

from genpoint3d.tools.traj_scale import RADIUS
from genpoint3d.train.tools.camera import image_projection, unproject
from genpoint3d.train.tools.metrics import _to_camera_t


def to_device(batch: dict, device) -> dict:
    """Move a batch, the fp16-cached features widened to fp32: autocast does
    not reach every module, and a half tensor meeting fp32 weights raises."""
    b = {k: v.to(device) for k, v in batch.items()}
    for k in ("context", "id_card", "patch_xyz"):
        b[k] = b[k].float()
    return b


def random_visual_mask(b: dict, min_context: int, max_gap: int) -> torch.Tensor:
    """(B, T, 2, 2) which quarters of each training frame are seen. Each clip
    loses, over a span of frames, a random quarter, half, three quarters or
    the whole frame, sizes mixed alike. The span is, for half the clips, every
    frame from a cutoff uniform in [min_context, T] (T: none), for the other
    half 2 to `max_gap` frames from min_context on, seen again after (0:
    cutoffs only). Drawn on the CPU, so a seed gives the same masks anywhere."""
    B, T = b["traj"].shape[:2]
    if not 1 <= min_context < T - 1:
        raise ValueError(f"--min-context {min_context} must be in [1, {T - 2}]: "
                         "frame 0 always keeps its image")
    t = torch.arange(T)[None]
    cut = torch.randint(min_context, T + 1, (B, 1))
    length = torch.randint(2, min(max(max_gap, 2), T - min_context - 1) + 1, (B, 1))
    # Ends before the last frame, so the clip is seen again after it.
    start = min_context + (torch.rand(B, 1) * (T - length - min_context)).long()
    gap = (torch.rand(B, 1) < 0.5) & (max_gap > 0)
    lost = torch.where(gap, (t >= start) & (t < start + length), t >= cut)             # (B, T)
    # The quarters lost: one, a half, all but one, or all, turned a random
    # multiple of 90 degrees so that any quarter or half can be the one.
    shapes = torch.tensor([[[1, 0], [0, 0]], [[1, 1], [0, 0]], [[1, 1], [1, 0]], [[1, 1], [1, 1]]], dtype=torch.bool)
    sizes = torch.randint(4, (B,))
    grid = torch.stack([torch.rot90(shapes[k], int(r)) for k, r in zip(sizes, torch.randint(4, (B,)))])
    view = ~(lost[:, :, None, None] & grid[:, None])
    return view.to(b["traj"].device)


def visual_mask(b: dict, cutoff: torch.Tensor | None) -> torch.Tensor:
    """(B, T, 2, 2) the whole frame seen before each clip's cutoff, none from it;
    every frame without one."""
    B, T = b["traj"].shape[:2]
    device = b["traj"].device
    if cutoff is None:
        return torch.ones(B, T, 2, 2, dtype=torch.bool, device=device)
    if not (cutoff >= 1).all():
        raise ValueError("a cutoff below 1 would hide frame 0, the query frame")
    seen = torch.arange(T, device=device)[None] < cutoff[:, None]
    return seen[:, :, None, None].expand(B, T, 2, 2).clone()


def black_out(frames: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 frames with their hidden quarters black, so the
    CNN cannot see them either (H, W even)."""
    H, W = frames.shape[2:4]
    seen = view.repeat_interleave(H // 2, 2).repeat_interleave(W // 2, 3)
    return frames * seen[..., None].to(frames.dtype)


def seen_at(view: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """(B, T, 2, 2) seen quarters and (B, T, N, 2) pixels in [0, 1] -> (B, T, N):
    whether each pixel is in a seen quarter (a pixel off the frame reads the
    nearest quarter)."""
    r, c = (uv[..., 1] >= 0.5).long(), (uv[..., 0] >= 0.5).long()
    return view.flatten(2).gather(2, r * 2 + c)


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


def group_pairs(b: dict, by_object: bool = True, tol: float = 0.02, frames: tuple[int, int] | None = None,
                k: int = 20, far: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """The pairs the group vectors are trained on, as `rigid_groups` labels
    them but only for each point's `k` nearest points in the first of
    `frames` (the truth's: where pieces meet, the pairs that are hard) and
    `far` points drawn at random (mostly other pieces): (B, N, k + far) their
    indices and (B, N, k + far) bool which are on its piece."""
    p = to_metres(b["traj"], b)
    if frames is not None:
        p = p[:, frames[0]:frames[1]]
    B, T, N, _ = p.shape
    self_ = torch.arange(N, device=p.device)[None, :, None]
    k = min(k, N - 1)
    near = torch.cdist(p[:, 0], p[:, 0]).masked_fill(torch.eye(N, dtype=torch.bool, device=p.device), float("inf"))
    idx = near.topk(k, dim=-1, largest=False).indices
    if far:
        rand = torch.randint(0, N - 1, (B, N, far), device=p.device)
        idx = torch.cat([idx, rand + (rand >= self_).long()], -1)        # never the point itself
    m = idx.shape[-1]
    mate = p.gather(2, idx.reshape(B, 1, N * m, 1).expand(B, T, N * m, 3)).reshape(B, T, N, m, 3)
    d = (mate - p[:, :, :, None]).norm(dim=-1)
    same = (d - d[:, :1]).abs().amax(dim=1) < tol
    if by_object and "segment" in b:
        seg = b["segment"]
        labelled = (seg >= 0).all(-1)
        same = torch.where(labelled[:, None, None], seg[:, :, None] == seg.gather(1, idx.flatten(1)).view(B, N, m), same)
    return idx, same


def rigid_pairs(b: dict, k: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Each point's partners for the rigidity loss: up to `k` of its nearest
    points in frame 0 (the truth's) on the same moving rigid piece
    (`rigid_buddies`): (B, N, k) their indices and (B, N, k) bool which are
    partners. As Dynamic 3D Gaussians (Luiten et al. 2024) keep each point's
    nearest neighbours rigid, found once in the first frame: on a rigid piece
    a pair's distance is the same in every frame."""
    return rigid_buddies(to_metres(b["traj"], b), k)


def rigid_groups(b: dict, by_object: bool = True, tol: float = 0.02,
                 frames: tuple[int, int] | None = None) -> torch.Tensor:
    """(B, N, N) what the group vectors should put together: with `by_object`,
    in a clip that says which object each point is on (`segment`), the pairs on
    one object (Kubric's objects are rigid, the background one piece);
    otherwise the pairs whose distance never strays more than `tol` metres from
    the first frame's, moving or not, over `frames` (start, end), the frames a
    window sees and its rigid fit works over (the whole clip by default)."""
    p = to_metres(b["traj"], b)
    if frames is not None:
        p = p[:, frames[0]:frames[1]]
    B, T, N, _ = p.shape
    d = torch.cdist(p.flatten(0, 1), p.flatten(0, 1)).view(B, T, N, N)
    groups = (d - d[:, :1]).abs().amax(dim=1) < tol
    if by_object and "segment" in b:
        seg = b["segment"]
        labelled = (seg >= 0).all(-1)                                     # (B,)
        groups = torch.where(labelled[:, None, None], seg[:, :, None] == seg[:, None], groups)
    return groups


def rigid_buddies(p: torch.Tensor, k: int, tol: float = 0.02, moved: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """From true positions (B, T, N, 3) in metres, each point's up to `k`
    nearest points in frame 0 whose distance to it never strays more than
    `tol` from frame 0's, both moving more than `moved` at some frame: (B, N,
    k) indices and (B, N, k) bool which count. Kubric objects are rigid, so
    this finds them without object ids; the moving part leaves the
    background out. Only the 2k nearest are checked, never every pair."""
    B, T, N, _ = p.shape
    c = min(2 * k, N - 1)
    near = torch.cdist(p[:, 0], p[:, 0])
    near = near.masked_fill(torch.eye(N, dtype=torch.bool, device=p.device), float("inf"))
    idx = near.topk(c, dim=-1, largest=False).indices                    # (B, N, c) nearest first
    mate = p.gather(2, idx.reshape(B, 1, N * c, 1).expand(B, T, N * c, 3)).reshape(B, T, N, c, 3)
    d = (mate - p[:, :, :, None]).norm(dim=-1)                          # (B, T, N, c)
    rigid = (d - d[:, :1]).abs().amax(dim=1) < tol
    move = (p - p[:, :1]).norm(dim=-1).amax(dim=1) > moved             # (B, N)
    ok = rigid & move[:, :, None] & move.gather(1, idx.flatten(1)).view(B, N, c)
    # The k nearest that count: a stable sort keeps the nearest first among them.
    order = (~ok).float().argsort(dim=-1, stable=True)[..., :k]
    return idx.gather(2, order), ok.gather(2, order)


def fine_inputs(model, b: dict, amp: bool = False, grad: bool = False) -> dict:
    """The matcher's inputs: the CNN's map of the batch's frames, each frame's
    camera, each cell's 3D position and the query frames. The map has no
    graph; with `grad` it is a leaf that gathers every backward pass's
    gradient for `FineFinder.replay`."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        fmap = model.fine.encode(b["frames"])
    proj = image_projection(b["intrinsics"], b["extrinsics"], b["hw"], b["norm_scale"], b["norm_mean"])
    return {"fine_map": fmap.requires_grad_(grad), "fine_proj": proj,
            "fine_xyz": _cell_xyz(b["depth"].float(), proj), "query_t": b["query_t"],
            "gravity": b["gravity"], "down": b["down"], "down_known": b["down_known"]}


def _cell_xyz(depth: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """depth (B, T, h, w) metres at each CNN cell, proj (B, T, 3, 4) ->
    (B, T, 3, h, w) each cell's 3D position in `anchor`'s space. Cell j is
    centred at `(j + 1/8) / w` across, where `cache_frames.py` reads its depth;
    depth beyond RADIUS held at it."""
    B, T, h, w = depth.shape
    v = (torch.arange(h, device=depth.device, dtype=torch.float32) + 0.125) / h
    u = (torch.arange(w, device=depth.device, dtype=torch.float32) + 0.125) / w
    uv = torch.stack(torch.meshgrid(u, v, indexing="xy"), dim=-1).reshape(1, 1, -1, 2)
    xyz = unproject(uv.expand(B, T, -1, -1), depth.flatten(2).clamp(max=RADIUS), proj.float())
    return xyz.transpose(2, 3).reshape(B, T, 3, h, w)
