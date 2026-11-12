"""Turn one cached clip into a normalised, subsampled training sample."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from genpoint3d.CachedClip import CachedClip
from genpoint3d.tools.feature_lookup import sample_at
from genpoint3d.NormStats import NormStats
from genpoint3d.tools.traj_scale import (GRAVITY, GRAVITY_REF, KUBRIC_FPS, MATCH_SPREAD, RADIUS, REACH, SCENE_SCALE,
                                         TRAJ_SCALE_DISP)
from genpoint3d.train.model.FineFinder import frame_spread
from genpoint3d.train.tools.batch import _cell_xyz
from genpoint3d.train.tools.camera import image_projection
from genpoint3d.train.tools.metrics import resize_intrinsics

# Each clip's depth beside its frames, as `scripts/cache_frames.py` writes it.
DEPTH = ".depth.npy"


class ClipDataset(Dataset):
    """Serves cached clips, `num_points` of their points per draw (drawn
    afresh each time with `resample`), loaded lazily so workers stay small.

    The target is each point's displacement from its anchor, over
    TRAJ_SCALE_DISP. `frames` is a directory from `scripts/cache_frames.py`
    (RGB and depth per clip), or one per cache directory; None serves neither. `level` turns each clip so
    up is -y (`_levelled`). `query_before` C keeps only points first seen
    before frame C: a forecast from C is scored on points it was handed. Only
    points within REACH of the camera in their query frame are served (`_near`).
    """

    def __init__(self, clips: list[Path | dict], num_points: int, resample: bool = True,
                 frames: Path | dict[Path, Path] | None = None, level: bool = False, query_before: int = 0):
        self.clips, self.num_points, self.resample = clips, num_points, resample
        self.frames, self.level, self.query_before = frames, level, query_before
        self.traj_scale = TRAJ_SCALE_DISP

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        c = self.clips[i]
        frames_dir = self.frames.get(c.parent) if isinstance(self.frames, dict) else self.frames
        if isinstance(c, Path):
            c = torch.load(c, weights_only=False)
        clip = _near(c if isinstance(c, CachedClip) else CachedClip.from_dict(c))

        if self.level:
            clip = _levelled(clip)
        # Empty tensors rather than None, for the default collate.
        frames = depth = torch.zeros(0)
        if frames_dir is not None:
            frames = torch.from_numpy(np.load(frames_dir / f"{clip.seq_id}.npy"))
            if frames.shape[0] != clip.traj.shape[0]:
                raise ValueError(f"clip {clip.seq_id}: {frames.shape[0]} cached frames "
                                 f"for a {clip.traj.shape[0]}-frame track")
            depth = torch.from_numpy(np.load(frames_dir / f"{clip.seq_id}{DEPTH}"))
        # Positions, anchors and patches in the clip's own unit; only the target
        # is then divided by traj_scale. Patches beyond RADIUS held at it.
        # Without the depth (scoring others' tracks), a typical clip's unit.
        unit = _clip_unit(depth, clip) if depth.numel() else torch.tensor(SCENE_SCALE)
        n = NormStats(torch.zeros(3), unit, torch.tensor(1.0))
        # Down, as strong as gravity is in this clip's unit; a typical clip's 1.
        gravity = (torch.zeros(3) if clip.up is None
                   else -clip.up.float() / clip.up.float().norm() * (GRAVITY / KUBRIC_FPS ** 2 / unit / GRAVITY_REF))
        traj, anchor = n.apply(clip.traj), n.apply(clip.anchor)
        pxyz = None
        if clip.patch_xyz is not None:
            p = clip.patch_xyz.float()
            pxyz = n.apply(p * (RADIUS / p.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1))
        # Each point's own start, as the model is given it.
        offset = anchor[None].clone()      # (1, N, 3)
        traj = traj - offset
        traj = traj / self.traj_scale
        vis = clip.visibility
        id_card = clip.id_card
        # Each point's object; -1 where the cache does not say.
        segment = (clip.segment.long() if clip.segment is not None
                   else torch.full((traj.shape[1],), -1, dtype=torch.long))
        # Each point's query frame; a cache from before it had them all at 0.
        query_t = (torch.zeros(traj.shape[1], dtype=torch.long) if clip.query_t is None
                   else clip.query_t.long())
        if id_card is not None and clip.context is not None:
            # Read again in each point's query frame with the exact lookup
            # (`tools/feature_lookup.py`), before the subsampling.
            fixed = id_card.clone()
            for t in query_t.unique().tolist():
                m = query_t == t
                ft = clip.context[t].float()
                g = math.isqrt(ft.shape[0])
                fixed[m] = sample_at(ft.T.reshape(1, -1, g, g), clip.query_uv[m][None].float(),
                                     clip.hw)[0].to(id_card.dtype)
            id_card = fixed
        if self.query_before:
            keep = query_t < self.query_before
            traj, anchor, vis, query_t, offset = traj[:, keep], anchor[keep], vis[:, keep], query_t[keep], offset[:, keep]
            segment = segment[keep]
            if id_card is not None:
                id_card = id_card[keep]
        n_pts = traj.shape[1]
        if self.num_points < n_pts:
            idx = (torch.randperm(n_pts)[: self.num_points] if self.resample
                   else torch.arange(self.num_points))
            traj, anchor, vis, query_t = traj[:, idx], anchor[idx], vis[:, idx], query_t[idx]
            offset = offset[:, idx]
            segment = segment[idx]
            if id_card is not None:
                id_card = id_card[idx]

        # Features stay fp16 until `to_device`.
        ctx = clip.context
        norm = NormStats(n.mean, n.scale, torch.tensor(self.traj_scale))
        return {
            "traj": traj, "anchor": anchor, "visibility": vis, "query_t": query_t, "segment": segment,
            # (3,) down, in the clip's unit per frame squared over a typical clip's.
            "gravity": gravity,
            "context": ctx if ctx is not None else torch.zeros(0),
            "id_card": id_card if id_card is not None else torch.zeros(0),
            "patch_xyz": pxyz if pxyz is not None else torch.zeros(0),
            # (T, R, R, 3) uint8 until the CNN takes it.
            "frames": frames,
            # (T, R / 4, R / 4) float32 z-depth in metres at each CNN cell.
            "depth": depth,
            "intrinsics": clip.intrinsics,
            # The benchmark's thresholds are for a 256 px short side.
            "intrinsics_256": resize_intrinsics(clip.intrinsics, clip.hw),
            "extrinsics": clip.extrinsics,
            # (H, W) of the native frame.
            "hw": torch.tensor(clip.hw),
            "norm_mean": norm.mean,
            "norm_scale": norm.scale,
            "norm_traj_scale": norm.traj_scale,
            # (1, N, 3) scene-normalised per-point origin, for `to_metres`.
            "norm_offset": offset,
        }

    @staticmethod
    def frames_dir(path: str | None, clips: list, res: int) -> Path:
        """The frames cache for `clips`, checked before training: every clip
        has its frames at `res` and its depth at one cell per 4 px."""
        if not path:
            raise SystemExit("pass --frames-cache, a directory from scripts/cache_frames.py")
        d = Path(path)
        for kind, suffix, size in (("frames", ".npy", res), ("depth", DEPTH, res // 4)):
            missing = [c.stem for c in clips if not (d / f"{c.stem}{suffix}").exists()]
            if missing:
                raise SystemExit(f"{len(missing)} of {len(clips)} clips have no {kind} in {d} "
                                 f"({' '.join(missing[:5])}{' ...' if len(missing) > 5 else ''})"
                                 " -- run scripts/cache_frames.py")
            got = np.load(d / f"{clips[0].stem}{suffix}", mmap_mode="r").shape[1:3]
            if got != (size, size):
                raise SystemExit(f"{kind} in {d} are {got[0]}x{got[1]}, --fine-res {res} needs {size}x{size}")
        return d


def _clip_unit(depth: torch.Tensor, clip: CachedClip) -> torch.Tensor:
    """() metres per model unit for `clip`: its frame-0 cells' spread, as the
    matcher measures it (`FineFinder.frame_spread`), over MATCH_SPREAD, so
    that spread is MATCH_SPREAD in every clip's unit and a typical clip's
    unit is about SCENE_SCALE."""
    proj = image_projection(clip.intrinsics[None, :1], clip.extrinsics[None, :1], torch.tensor(clip.hw)[None],
                            torch.ones(1), torch.zeros(1, 3))                  # metres
    first = _cell_xyz(depth[None, :1].float(), proj)[:, 0]                   # (1, 3, h, w)
    return frame_spread(first)[1][0] / MATCH_SPREAD


def _near(clip: CachedClip) -> CachedClip:
    """The clip with only the points within REACH of the camera in their query
    frame: not Kubric's sky, a dome 40 m out."""
    q = (torch.zeros(clip.traj.shape[1], dtype=torch.long) if clip.query_t is None else clip.query_t.long())
    n = torch.arange(len(q))
    p = torch.cat([clip.traj[q, n].float(), torch.ones(len(q), 1)], -1)                 # (N, 4) frame-0 camera
    keep = (clip.extrinsics[q].float() @ p[..., None])[:, :3, 0].norm(dim=-1) <= REACH
    if keep.all():
        return clip
    per_point = dict(traj=clip.traj[:, keep], anchor=clip.anchor[keep], visibility=clip.visibility[:, keep],
                     query_uv=clip.query_uv[keep])
    for k in ("query_t", "id_card", "segment"):
        if getattr(clip, k) is not None:
            per_point[k] = getattr(clip, k)[keep]
    return dataclasses.replace(clip, **per_point)


def _levelled(clip: CachedClip) -> CachedClip:
    """The clip turned about the frame-0 camera's centre, by the smallest
    rotation that takes its `up` to -y: tilt and roll gone, heading kept."""
    if clip.up is None:
        raise ValueError(f"clip {clip.seq_id} has no up direction to level by: re-cache it, "
                         "or add it with scripts/add_up.py")
    u = clip.up.float() / clip.up.float().norm()
    t = torch.tensor([0.0, -1.0, 0.0])
    v, c = torch.linalg.cross(u, t), float(u @ t)
    K = torch.tensor([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    R = torch.eye(3) + K + K @ K / (1.0 + c)                        # takes u to t
    turn = lambda p: None if p is None else p.float() @ R.T
    E = None
    if clip.extrinsics is not None:
        Rh = torch.eye(4)
        Rh[:3, :3] = R
        E = clip.extrinsics.float() @ Rh.T                          # levelled -> cam_t
    return dataclasses.replace(clip, traj=turn(clip.traj), anchor=turn(clip.anchor),
                               patch_xyz=turn(clip.patch_xyz), extrinsics=E, up=t)
