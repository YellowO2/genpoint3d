"""Turn one cached clip into a normalised, subsampled training sample."""

import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from genpoint3d.CachedClip import CachedClip
from genpoint3d.tools.feature_lookup import id_cards
from genpoint3d.NormStats import NormStats
from genpoint3d.tools.traj_scale import (GRAVITY, GRAVITY_REF, KUBRIC_FPS, MATCH_SPREAD, RADIUS, REACH, SCENE_SCALE,
                                         TRAJ_SCALE_DISP)
from genpoint3d.train.tools.batch import _cell_xyz, random_view, whole_frames
from genpoint3d.train.tools.camera import image_projection
from genpoint3d.train.tools.farthest_first import SPAWN_GAP, fill_frames
from genpoint3d.train.tools.frame_spread import frame_spread
from genpoint3d.train.tools.metrics import resize_intrinsics

# Each clip's depth beside its frames, as `scripts/cache_frames.py` writes it.
DEPTH = ".depth.npy"


class ClipDataset(Dataset):
    """Serves cached clips, `num_points` of their points per draw (drawn
    afresh each time with `resample`), loaded lazily so workers stay small.

    The target is each point's displacement from its anchor, over
    TRAJ_SCALE_DISP. `frames` is a directory from `scripts/cache_frames.py`
    (RGB and depth per clip), or one per cache directory; None serves neither. `query_before` C keeps only points first seen
    before frame C: a forecast from C is scored on points it was handed. Only
    points within REACH of the camera in their query frame are served (`_near`).
    With `view` (min_context, max_gap), each draw first loses part of its
    frames as training does (`random_view`, served as "view"), and only points
    it can hand over are served: none starts in a frame not seen whole. With
    `spawn`, that many of the points are helpers added by the spawning rule,
    the rest asked (`_spawned`); "asked" says which.
    """

    def __init__(self, clips: list[Path | dict], num_points: int, resample: bool = True,
                 frames: Path | dict[Path, Path] | None = None, query_before: int = 0,
                 spawn: int = 0, spawn_gap: float = SPAWN_GAP, view: tuple[int, int] | None = None):
        self.clips, self.num_points, self.resample = clips, num_points, resample
        self.frames, self.query_before = frames, query_before
        self.spawn, self.spawn_gap, self.view = spawn, spawn_gap, view
        self.traj_scale = TRAJ_SCALE_DISP

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        c = self.clips[i]
        frames_dir = self.frames.get(c.parent) if isinstance(self.frames, dict) else self.frames
        if isinstance(c, Path):
            c = torch.load(c, weights_only=False)
        clip = _near(c if isinstance(c, CachedClip) else CachedClip.from_dict(c))
        # Empty tensors rather than None, for the default collate.
        view = torch.zeros(0)
        whole = torch.ones(clip.traj.shape[0], dtype=torch.bool)
        if self.view is not None:
            view = random_view(clip.traj.shape[0], *self.view)
            whole = whole_frames(view)
        if self.query_before:
            # Forecast from C: frames from C on are not seen, so nothing is asked or added there.
            whole = whole & (torch.arange(clip.traj.shape[0]) < self.query_before)
        n_all = clip.traj.shape[1]
        asked = torch.ones(n_all, dtype=torch.bool)
        if self.spawn:
            order = torch.randperm(n_all) if self.resample else torch.arange(n_all)
            clip, asked = _spawned(clip, self.num_points, self.spawn, self.spawn_gap, whole, order)
        elif self.view is not None:
            clip = _points(clip, torch.nonzero(whole[_query_frames(clip)])[:, 0])
            asked = asked[: clip.traj.shape[1]]

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
        # Gravity's strength in this clip's unit, a typical clip's 1: always
        # known. Which way is down only when the clip says which way is up.
        gravity = GRAVITY / KUBRIC_FPS ** 2 / unit / GRAVITY_REF
        known = clip.up is not None
        down = -clip.up.float() / clip.up.float().norm() if known else torch.zeros(3)
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
        # Its object's centre of mass, a target like `traj`, on the points whose cache says (`centre_known`).
        centre_known = (torch.isfinite(clip.centre).all(-1).all(0) if clip.centre is not None
                        else torch.zeros(traj.shape[1], dtype=torch.bool))
        centre = ((n.apply(clip.centre.float().nan_to_num()) - offset) / self.traj_scale * centre_known[:, None]
                  if clip.centre is not None else torch.zeros_like(traj))
        query_t = _query_frames(clip)
        if id_card is not None and clip.context is not None:
            # Read again in each point's query frame with the exact lookup
            # (`tools/feature_lookup.py`), before the subsampling.
            id_card = id_cards(clip.context, query_t, clip.query_uv, clip.hw).to(id_card.dtype)
        if self.query_before:
            keep = query_t < self.query_before
            traj, anchor, vis, query_t, offset = traj[:, keep], anchor[keep], vis[:, keep], query_t[keep], offset[:, keep]
            segment, centre, centre_known, asked = segment[keep], centre[:, keep], centre_known[keep], asked[keep]
            if id_card is not None:
                id_card = id_card[keep]
        n_pts = traj.shape[1]
        if self.num_points < n_pts:
            idx = (torch.randperm(n_pts)[: self.num_points] if self.resample
                   else torch.arange(self.num_points))
            traj, anchor, vis, query_t = traj[:, idx], anchor[idx], vis[:, idx], query_t[idx]
            offset = offset[:, idx]
            segment, centre, centre_known, asked = segment[idx], centre[:, idx], centre_known[idx], asked[idx]
            if id_card is not None:
                id_card = id_card[idx]

        # Features stay fp16 until `to_device`.
        ctx = clip.context
        norm = NormStats(n.mean, n.scale, torch.tensor(self.traj_scale))
        return {
            "traj": traj, "anchor": anchor, "visibility": vis, "query_t": query_t, "segment": segment,
            # (N,) the points asked about; the others are helpers the clip was filled with, never scored.
            "asked": asked,
            # (T, 2, 2) the quarters of each frame seen, with `view`.
            "view": view,
            # (T, N, 3) its object's centre of mass, as `traj`; (N,) where it is known.
            "centre": centre, "centre_known": centre_known,
            # () gravity in the clip's unit per frame squared over a typical clip's;
            # (3,) which way is down in the frame-0 camera, where `down_known`.
            "gravity": gravity, "down": down, "down_known": torch.tensor(known),
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
        has its frames at `res` and its depth at one cell per 4 px (a cache
        with no clips in the run, as with --overfit, has nothing to check)."""
        if not path:
            raise SystemExit("pass --frames-cache, a directory from scripts/cache_frames.py")
        d = Path(path)
        for kind, suffix, size in (("frames", ".npy", res), ("depth", DEPTH, res // 4)):
            missing = [c.stem for c in clips if not (d / f"{c.stem}{suffix}").exists()]
            if missing:
                raise SystemExit(f"{len(missing)} of {len(clips)} clips have no {kind} in {d} "
                                 f"({' '.join(missing[:5])}{' ...' if len(missing) > 5 else ''})"
                                 " -- run scripts/cache_frames.py")
            got = np.load(d / f"{clips[0].stem}{suffix}", mmap_mode="r").shape[1:3] if clips else (size, size)
            if got != (size, size):
                raise SystemExit(f"{kind} in {d} are {got[0]}x{got[1]}, --fine-res {res} needs {size}x{size}")
        return d


def _clip_unit(depth: torch.Tensor, clip: CachedClip) -> torch.Tensor:
    """() metres per model unit for `clip`: its frame-0 cells' spread, as the
    matcher measures it (`tools/frame_spread.py`), over MATCH_SPREAD, so
    that spread is MATCH_SPREAD in every clip's unit and a typical clip's
    unit is about SCENE_SCALE."""
    proj = image_projection(clip.intrinsics[None, :1], clip.extrinsics[None, :1], torch.tensor(clip.hw)[None],
                            torch.ones(1), torch.zeros(1, 3))                  # metres
    first = _cell_xyz(depth[None, :1].float(), proj)[:, 0]                   # (1, 3, h, w)
    return frame_spread(first)[1][0] / MATCH_SPREAD


def _query_frames(clip: CachedClip) -> torch.Tensor:
    """(N,) each point's query frame; a cache from before them has all at 0."""
    return torch.zeros(clip.traj.shape[1], dtype=torch.long) if clip.query_t is None else clip.query_t.long()


def _spawned(clip: CachedClip, num_points: int, cap: int, gap: float, whole: torch.Tensor,
             order: torch.Tensor) -> tuple[CachedClip, torch.Tensor]:
    """The clip with `num_points - cap` of its points asked, the first in
    `order` whose own query frame is seen whole (`whole` (T,)), and the rest
    up to `num_points` added by the spawning rule (`fill_frames`), as a
    tracker on real video adds them: in each frame seen whole, a point of the
    clip seen there, farther than `gap` (a share of the frame's width) from
    every point already there and seen, starts in that frame, the farthest
    first, its track's place there its query. The clip's own tracks stand in
    for the new points, so their truth is known. Also (N,) which points are
    asked: the others are helpers, trained on but never scored."""
    T, N = clip.visibility.shape
    q = _query_frames(clip)
    order = order[whole[q][order]]
    asked = order[: num_points - cap]
    vis = clip.visibility.bool()
    P = torch.cat([clip.traj.float(), torch.ones(T, N, 1)], -1)
    pix = torch.einsum("tij,tnj->tni", clip.intrinsics.float(),
                       torch.einsum("tij,tnj->tni", clip.extrinsics.float()[:, :3, :], P))
    uv = pix[..., :2] / pix[..., 2:].clamp_min(1e-6)                                   # (T, N, 2) pixels
    there = torch.zeros(T, N, dtype=torch.bool)                                         # (T, N) given by frame t
    there[:, asked] = q[asked][None] <= torch.arange(T)[:, None]

    def taken(born):
        t = torch.zeros(N, dtype=torch.bool)
        t[asked] = True
        t[[k for _, k in born]] = True
        return t

    def candidates(t, born):
        c = torch.nonzero(vis[t] & ~taken(born))[:, 0]
        return uv[t, c], c

    def covering(t, born):
        on = there[t] & vis[t]
        on[[k for t0, k in born if t0 <= t and vis[t, k]]] = True
        return uv[t, on]

    born = fill_frames(torch.nonzero(whole)[:, 0].tolist(), num_points - len(asked), gap * clip.hw[1],
                       candidates, covering)
    keep, _ = torch.sort(torch.cat([asked, torch.tensor([k for _, k in born], dtype=torch.long)]))
    is_asked = torch.zeros(N, dtype=torch.bool)
    is_asked[asked] = True
    if born:
        b = torch.tensor([k for _, k in born])
        start = q.clone()
        start[b] = torch.tensor([t for t, _ in born])
        n = torch.arange(N)
        anchor, query_uv = clip.anchor.clone(), clip.query_uv.clone()
        anchor[b] = clip.traj[start[b], n[b]].to(anchor.dtype)
        query_uv[b] = uv[start[b], n[b]].to(query_uv.dtype)
        clip = dataclasses.replace(clip, anchor=anchor, query_uv=query_uv, query_t=start)
    return _points(clip, keep), is_asked[keep]


def _points(clip: CachedClip, keep: torch.Tensor) -> CachedClip:
    """The clip with only the points `keep`."""
    per_point = dict(traj=clip.traj[:, keep], anchor=clip.anchor[keep], visibility=clip.visibility[:, keep],
                     query_uv=clip.query_uv[keep])
    for k in ("query_t", "id_card", "segment"):
        if getattr(clip, k) is not None:
            per_point[k] = getattr(clip, k)[keep]
    if clip.centre is not None:
        per_point["centre"] = clip.centre[:, keep]
    return dataclasses.replace(clip, **per_point)


def _near(clip: CachedClip) -> CachedClip:
    """The clip with only the points within REACH of the camera in their query
    frame: not Kubric's sky, a dome 40 m out."""
    q = _query_frames(clip)
    n = torch.arange(len(q))
    p = torch.cat([clip.traj[q, n].float(), torch.ones(len(q), 1)], -1)                 # (N, 4) frame-0 camera
    keep = (clip.extrinsics[q].float() @ p[..., None])[:, :3, 0].norm(dim=-1) <= REACH
    if keep.all():
        return clip
    return _points(clip, keep)
