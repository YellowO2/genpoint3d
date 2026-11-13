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
    (RGB and depth per clip), or one per cache directory; None serves neither. `query_before` C keeps only points first seen
    before frame C: a forecast from C is scored on points it was handed. Only
    points within REACH of the camera in their query frame are served (`_near`).
    With `spawn`, up to that many of the points start where new surface comes
    into view (`_spawned`).
    """

    def __init__(self, clips: list[Path | dict], num_points: int, resample: bool = True,
                 frames: Path | dict[Path, Path] | None = None, query_before: int = 0,
                 spawn: int = 0, spawn_gap: float = 0.05):
        self.clips, self.num_points, self.resample = clips, num_points, resample
        self.frames, self.query_before = frames, query_before
        self.spawn, self.spawn_gap = spawn, spawn_gap
        self.traj_scale = TRAJ_SCALE_DISP

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        c = self.clips[i]
        frames_dir = self.frames.get(c.parent) if isinstance(self.frames, dict) else self.frames
        if isinstance(c, Path):
            c = torch.load(c, weights_only=False)
        clip = _near(c if isinstance(c, CachedClip) else CachedClip.from_dict(c))
        if self.spawn:
            clip = _spawned(clip, self.num_points, self.spawn, self.spawn_gap)

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
        # Each point's query frame; a cache from before it had them all at 0.
        query_t = (torch.zeros(traj.shape[1], dtype=torch.long) if clip.query_t is None
                   else clip.query_t.long())
        if id_card is not None and clip.context is not None:
            # Read again in each point's query frame with the exact lookup
            # (`tools/feature_lookup.py`), before the subsampling.
            id_card = id_cards(clip.context, query_t, clip.query_uv, clip.hw).to(id_card.dtype)
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


def _spawned(clip: CachedClip, num_points: int, cap: int, gap: float) -> CachedClip:
    """`num_points` of the clip's points, up to `cap` of them started where new
    surface comes into view, as a tracker on real video would add them: frame
    by frame, a point of the clip that comes into view there (hidden in the
    frame before: surface turning towards the camera, out from behind
    something, into the frame) farther than `gap` (a share of the frame's
    width) from every point already tracked and seen there starts in that
    frame, the farthest first, its track's place there its query. The
    rest are drawn as given, at random. The clip's own tracks stand in for the
    new points, so their truth is known."""
    T, N = clip.visibility.shape
    if N <= num_points - cap:
        return clip
    vis = clip.visibility.bool()
    q = torch.zeros(N, dtype=torch.long) if clip.query_t is None else clip.query_t.long()
    P = torch.cat([clip.traj.float(), torch.ones(T, N, 1)], -1)
    pix = torch.einsum("tij,tnj->tni", clip.intrinsics.float(),
                       torch.einsum("tij,tnj->tni", clip.extrinsics.float()[:, :3, :], P))
    uv = pix[..., :2] / pix[..., 2:].clamp_min(1e-6)                                   # (T, N, 2) pixels
    perm = torch.randperm(N)
    tracked = torch.zeros(N, dtype=torch.bool)
    tracked[perm[: num_points - cap]] = True
    start = q.clone()
    born = []
    far = gap * clip.hw[1]
    for t in range(1, T):
        if len(born) >= cap:
            break
        live = tracked & vis[t] & (start <= t)
        cand = torch.nonzero(~tracked & vis[t] & ~vis[t - 1])[:, 0]
        if not len(cand):
            continue
        d = (torch.cdist(uv[t, cand], uv[t, live]).min(1).values if live.any()
             else torch.full((len(cand),), float("inf")))
        for j in farthest_first(uv[t, cand], d, far, cap - len(born)):
            k = int(cand[j])
            tracked[k], start[k] = True, t
            born.append(k)
    # The budget the new points left, filled with more points as given.
    fill = perm[~tracked[perm]][: num_points - int(tracked.sum())]
    tracked[fill] = True
    keep = torch.nonzero(tracked)[:, 0]
    if not born:
        return _points(clip, keep)
    b = torch.tensor(born)
    n = torch.arange(N)
    anchor, query_uv = clip.anchor.clone(), clip.query_uv.clone()
    anchor[b] = clip.traj[start[b], n[b]].to(anchor.dtype)
    query_uv[b] = uv[start[b], n[b]].to(query_uv.dtype)
    return _points(dataclasses.replace(clip, anchor=anchor, query_uv=query_uv, query_t=start), keep)


def farthest_first(px: torch.Tensor, d: torch.Tensor, far: float, most: int) -> list[int]:
    """Up to `most` of the candidates at pixels `px` (C, 2), the farthest
    first: each farther than `far` both by `d` (C,), its distance to what is
    there already, and from the ones picked before it."""
    picked = []
    while len(picked) < most:
        j = int(d.argmax())
        if d[j] <= far:
            break
        picked.append(j)
        d = torch.minimum(d, (px - px[j]).norm(dim=-1))
    return picked


def _points(clip: CachedClip, keep: torch.Tensor) -> CachedClip:
    """The clip with only the points `keep`."""
    per_point = dict(traj=clip.traj[:, keep], anchor=clip.anchor[keep], visibility=clip.visibility[:, keep],
                     query_uv=clip.query_uv[keep])
    for k in ("query_t", "id_card", "segment"):
        if getattr(clip, k) is not None:
            per_point[k] = getattr(clip, k)[keep]
    return dataclasses.replace(clip, **per_point)


def _near(clip: CachedClip) -> CachedClip:
    """The clip with only the points within REACH of the camera in their query
    frame: not Kubric's sky, a dome 40 m out."""
    q = (torch.zeros(clip.traj.shape[1], dtype=torch.long) if clip.query_t is None else clip.query_t.long())
    n = torch.arange(len(q))
    p = torch.cat([clip.traj[q, n].float(), torch.ones(len(q), 1)], -1)                 # (N, 4) frame-0 camera
    keep = (clip.extrinsics[q].float() @ p[..., None])[:, :3, 0].norm(dim=-1) <= REACH
    if keep.all():
        return clip
    return _points(clip, keep)
