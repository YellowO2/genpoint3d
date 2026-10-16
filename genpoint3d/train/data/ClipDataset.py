"""Turn one cached clip into a normalised, subsampled training sample."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from genpoint3d.CachedClip import CachedClip
from genpoint3d.tools.feature_lookup import sample_at
from genpoint3d.tools.traj_scale import TRAJ_SCALE, TRAJ_SCALE_DISP
from genpoint3d.train.tools.metrics import resize_intrinsics

# The name each clip's depth has beside its frames: `<clip>.depth.npy`, as
# `scripts/cache_frames.py` writes it.
DEPTH = ".depth.npy"


class ClipDataset(Dataset):
    """Serves cached clips, subsampling `num_points` of them per draw.

    Reads from `genpoint3d.prepare`'s output rather than decoding PNGs.
    Resampling which points are used each epoch is free augmentation and makes
    the model robust to the point count, which the paper does deliberately.

    Clips are held as PATHS and loaded in `__getitem__`, not up front. At 7.3 MB
    per cached clip an eager load is 25 GB for 3500 clips -- and every dataloader
    worker gets its own copy, so it OOMs the moment `--workers` is non-zero.
    Lazily, only the clips in flight are resident. This is what makes workers
    affordable, which is what keeps the GPU fed.
    """

    def __init__(self, clips: list[Path | dict], num_points: int, resample: bool = True,
                 norm_mode: str = "median", target: str = "absolute",
                 fix_idcard: bool = True, frames: Path | None = None,
                 features: bool = True, depth: bool = False):
        self.clips, self.num_points, self.resample = clips, num_points, resample
        # The RGB frames, for the relative finder: a directory from
        # `scripts/cache_frames.py` (checked by `frames_dir`) holding one uint8 array per clip, named
        # like the clip. Every frame is served, in order and untouched --
        # nothing here crops, flips or drops frames, and which POINTS are
        # drawn does not concern an image.
        self.frames = frames
        # And beside each clip's frames its depth at every CNN cell, for a
        # finder that reads where each cell is in 3D (`--fine-xyz`).
        self.depth = depth
        # Off serves no DINOv3 feature, ID card or patch position at all, for
        # a model that is to track from the CNN alone.
        self.features = features
        # The cached ID card was read with a lookup that was up to half a patch
        # off (see `tools/feature_lookup.py`). The cache also holds frame 0's
        # whole feature grid and the query pixels, so it is read again here
        # rather than re-encoding 3500 clips. Off serves the cached one, only so
        # checkpoints trained on it still evaluate as they were trained.
        self.fix_idcard = fix_idcard
        # The cache holds metres; normalisation happens here, so switching
        # scheme is a flag rather than hours of reprocessing.
        self.norm_mode = norm_mode
        # "absolute" denoises where each point IS. "displacement" denoises how
        # far each point has moved from ITS OWN frame-0 position.
        #
        # Measured per clip: absolute std 0.59, one-shared-anchor 0.49,
        # per-point 0.08. Only per-point removes the static layout, which is
        # 86% of an absolute target's magnitude and which the model was already
        # told via `anchor`. MolmoMotion subtracts one shared anchor instead,
        # but it emits coordinates as text in a single frame and needs them
        # mutually comparable; we do not. TAPIP3D and Tesfaldet's model both
        # use per-point displacement from the query.
        #
        # The layout is not lost: `anchor` stays absolute and is what spatial
        # RoPE reads, so where points sit relative to each other is unaffected.
        self.target = target
        self.traj_scale = TRAJ_SCALE_DISP if target == "displacement" else TRAJ_SCALE

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        c = self.clips[i]
        if isinstance(c, Path):
            c = torch.load(c, weights_only=False)
        clip = c if isinstance(c, CachedClip) else CachedClip.from_dict(c)

        # traj_scale=1.0 leaves both in scene-normalised units, so the offset
        # below is subtracted before the target is scaled, not after.
        n = clip.norm(self.norm_mode, traj_scale=1.0)
        traj, anchor = n.apply(clip.traj), n.apply(clip.anchor)
        # Patch positions share the ANCHOR's normalisation, not the target's:
        # `traj_scale` divides only what gets denoised. A patch position has to be
        # comparable with `anchor` -- that is the whole point of a feature cloud --
        # so it goes through the same `n.apply`. Done here, not further down,
        # because `n` is rebound below.
        pxyz = None if clip.patch_xyz is None else n.apply(clip.patch_xyz)
        if self.target == "displacement":
            offset = traj[:1].clone()          # (1, N, 3), each point's own start
            traj = traj - offset
        else:
            offset = torch.zeros_like(traj[:1])
        traj = traj / self.traj_scale
        vis = clip.visibility
        id_card = clip.id_card
        if self.fix_idcard and id_card is not None:
            # Before the subsampling below, so it stays aligned with traj.
            # `context` is (T, P, F) with P row-major, which is what `tokens()`
            # flattens a (F, g, g) grid into.
            f0 = clip.context[0].float()
            g = math.isqrt(f0.shape[0])
            id_card = sample_at(
                f0.T.reshape(1, -1, g, g), clip.query_uv[None].float(), clip.hw,
            )[0].to(id_card.dtype)
        n_pts = traj.shape[1]
        if self.num_points < n_pts:
            idx = (torch.randperm(n_pts)[: self.num_points] if self.resample
                   else torch.arange(self.num_points))
            traj, anchor, vis = traj[:, idx], anchor[idx], vis[:, idx]
            offset = offset[:, idx]
            if id_card is not None:
                id_card = id_card[idx]

        # A dict rather than a tuple: evaluation needs the geometry that scoring
        # in metres depends on, and a positional tuple of eight was already
        # hard to read. Empty tensors rather than None so default collate works.
        #
        # Features stay in the fp16 they were cached in. Upcasting here cost
        # 452 MB per batch of 16 -- the single largest tensor in the step --
        # only for autocast to cast it straight back down. `to_device` restores
        # fp32 when autocast is off.
        ctx = clip.context
        if not self.features:
            ctx = id_card = pxyz = None
        frames = torch.zeros(0)
        if self.frames is not None:
            frames = torch.from_numpy(np.load(self.frames / f"{clip.seq_id}.npy"))
            if frames.shape[0] != traj.shape[0]:
                raise ValueError(f"clip {clip.seq_id}: {frames.shape[0]} cached frames "
                                 f"for a {traj.shape[0]}-frame track")
        depth = torch.zeros(0)
        if self.depth:
            depth = torch.from_numpy(np.load(self.frames / f"{clip.seq_id}{DEPTH}"))
        norm = clip.norm(self.norm_mode, traj_scale=self.traj_scale)
        return {
            "traj": traj, "anchor": anchor, "visibility": vis,
            "context": ctx if ctx is not None else torch.zeros(0),
            "id_card": id_card if id_card is not None else torch.zeros(0),
            "patch_xyz": pxyz if pxyz is not None else torch.zeros(0),
            # (T, R, R, 3) uint8, and it stays uint8 until the CNN takes a
            # clip of it: as floats a batch of 16 is 2 GB.
            "frames": frames,
            # (T, R / 4, R / 4) float32 z-depth in metres at each CNN cell.
            "depth": depth,
            # metrics only -- the model never sees these
            "intrinsics": clip.intrinsics,
            # The benchmark's pixel thresholds are defined on a frame whose
            # short side is 256 px. Kubric frames are larger, so scoring with
            # the native intrinsics makes every threshold too tight.
            "intrinsics_256": resize_intrinsics(clip.intrinsics, clip.hw),
            "extrinsics": clip.extrinsics,
            # (H, W) of the native frame: with the intrinsics, what says which
            # patch a point is drawn in (`tools/batch.py`'s `match_targets`).
            "hw": torch.tensor(clip.hw),
            "norm_mean": norm.mean,
            "norm_scale": norm.scale,
            "norm_traj_scale": norm.traj_scale,
            # (1, N, 3) scene-normalised per-point origin; zeros when the
            # target is absolute, so `to_metres` inverts both with one
            # expression.
            "norm_offset": offset,
        }

    @staticmethod
    def frames_dir(path: str | None, clips: list, res: int, depth: bool = False) -> Path:
        """The frames cache for `clips`, checked before anything is trained on it:
        a clip with no frames (or no depth, with `depth`), or frames at another
        size than the model was told, should stop the run now and not at the
        step that first draws it."""
        if not path:
            raise SystemExit("the relative finder reads RGB frames: pass --frames-cache, "
                             "a directory from scripts/cache_frames.py")
        d = Path(path)
        missing = [c.stem for c in clips if not (d / f"{c.stem}.npy").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} of {len(clips)} clips have no frames in {d} "
                             f"({' '.join(missing[:5])}{' ...' if len(missing) > 5 else ''})"
                             " -- run scripts/cache_frames.py")
        got = np.load(d / f"{clips[0].stem}.npy", mmap_mode="r").shape[1:3]
        if got != (res, res):
            raise SystemExit(f"frames in {d} are {got[0]}x{got[1]}, --fine-res is {res}")
        if depth:
            missing = [c.stem for c in clips if not (d / f"{c.stem}{DEPTH}").exists()]
            if missing:
                raise SystemExit(f"{len(missing)} of {len(clips)} clips have no depth in {d}"
                                 f" ({' '.join(missing[:5])}{' ...' if len(missing) > 5 else ''})"
                                 " -- run scripts/cache_frames.py again, it adds it")
            # One depth per cell of the CNN's map, which is 4 px of the frame.
            got = np.load(d / f"{clips[0].stem}{DEPTH}", mmap_mode="r").shape[1:3]
            if got != (res // 4, res // 4):
                raise SystemExit(f"depth in {d} is {got[0]}x{got[1]} cells, --fine-res {res}"
                                 f" needs {res // 4}x{res // 4}")
        return d
