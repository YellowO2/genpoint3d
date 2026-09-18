"""
The on-disk cache format -- one schema, written by `preprocess.py` and read by
`train.py`.

Two rules, both learned the hard way.

**Cache facts, not choices.** The trajectory is stored in METRES, and the
normalisation is applied when a clip is loaded. Storing normalised coordinates
baked a training decision into the data, so every change to the scheme -- max
to median, centred to camera-origin -- invalidated hours of preprocessing.
Metres are a fact about the clip; how to normalise them is a knob.

**One schema, both sides.** The cache used to be a bare dict built in one
script and indexed by string in another, with nothing tying the halves
together. A field could be dropped on the write side and simply never noticed
on the read side, which is how `norm` went missing.

Stored as a plain dict rather than a pickled dataclass so the file format does
not depend on this module's import path.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from genpoint3d.data.transform import TRAJ_SCALE, NormStats


@dataclass
class ScaleStats:
    """Every candidate normalisation statistic, measured once at preprocess time.

    All of them are kept because they cost four floats and computing any one of
    them needs the depth maps -- the single expensive part of preprocessing.
    Storing only the one currently in favour is what forced a re-run last time
    the choice changed.

    `centroid` and `max_from_centroid` describe the MotionForesight scheme
    (centre the scene, divide by its bounding radius). `mean_dist` and
    `median_dist` are distances to the camera for the DUSt3R scheme, which
    divides but never centres -- see `compute_norm_stats`.
    """

    centroid: torch.Tensor          # (3,) metric centre of the observed scene
    max_from_centroid: torch.Tensor  # () furthest point from that centre
    mean_dist: torch.Tensor          # () mean distance to the camera
    median_dist: torch.Tensor        # () median distance to the camera

    def norm(self, mode: str = "median", traj_scale: float = TRAJ_SCALE) -> NormStats:
        """Build the `NormStats` for one scheme. `mode` matches `compute_norm_stats`."""
        ts = torch.as_tensor(traj_scale, dtype=torch.float32)
        zero = torch.zeros(3)
        if mode == "median":
            return NormStats(zero, self.median_dist, ts)
        if mode == "mean":
            return NormStats(zero, self.mean_dist, ts)
        if mode == "centroid_max":   # the pre-2026-09-18 behaviour
            return NormStats(self.centroid, self.max_from_centroid, ts)
        raise ValueError(f"bad norm mode {mode!r}")

    def to_dict(self) -> dict:
        return {f"stats_{k}": getattr(self, k) for k in
                ("centroid", "max_from_centroid", "mean_dist", "median_dist")}

    @classmethod
    def from_dict(cls, d: dict) -> "Optional[ScaleStats]":
        if "stats_median_dist" not in d:
            return None
        return cls(**{k: d[f"stats_{k}"] for k in
                      ("centroid", "max_from_centroid", "mean_dist", "median_dist")})


def scale_stats(pointmap: torch.Tensor) -> ScaleStats:
    """Measure every candidate statistic from an observed pointmap.

    `pointmap`: (T_obs, 3, H, W) unprojected scene points, frame-0 frame.
    Computed once because it needs the depth maps, which are the expensive
    part of preprocessing; picking a scheme afterwards is then free.
    """
    pts = pointmap.permute(0, 2, 3, 1).reshape(-1, 3)
    pts = pts[torch.isfinite(pts).all(dim=-1)]
    if pts.numel() == 0:
        one = torch.ones(())
        return ScaleStats(torch.zeros(3), one, one, one)

    centroid = pts.mean(dim=0)
    dist = pts.norm(dim=-1)                 # to the camera, which is the origin
    return ScaleStats(
        centroid=centroid,
        max_from_centroid=(pts - centroid).norm(dim=-1).max(),
        mean_dist=dist.mean(),
        median_dist=dist.median(),
    )


@dataclass
class CachedClip:
    """One preprocessed clip. Frozen frame-0 camera frame throughout.

    `traj` and `anchor` are in METRES. Call `normalised()` to get the tensors
    the model actually eats.
    """

    seq_id: str

    # --- ground truth, in metres ---
    traj: torch.Tensor            # (T, N, 3) absolute position
    anchor: torch.Tensor          # (N, 3) frame-0 position, for conditioning
    visibility: torch.Tensor      # (T, N) bool, True = visible

    # --- query pointers ---
    query_uv: torch.Tensor        # (N, 2) pixel location in frame 0
    hw: tuple[int, int]           # frame size, to map uv into the feature grid

    # --- geometry the metrics need ---
    # TAP-Vid-3D back-projects its pixel thresholds through the focal length,
    # so intrinsics are not optional for scoring even though training ignores them.
    intrinsics: Optional[torch.Tensor] = None   # (T, 3, 3) pixel units
    extrinsics: Optional[torch.Tensor] = None   # (T, 4, 4) cam_0 -> cam_t
    stats: Optional[ScaleStats] = None

    # --- visual conditioning, absent for the step-2 (no-image) model ---
    context: Optional[torch.Tensor] = None      # (T, P, D) fp16 DINOv3 tokens + depth features
    id_card: Optional[torch.Tensor] = None      # (N, D) fp16, sampled at frame 0

    points: Optional[int] = None
    feat_dim: Optional[int] = None
    image_size: Optional[int] = None

    def norm(self, mode: str = "median", traj_scale: float = TRAJ_SCALE) -> NormStats:
        if self.stats is None:
            raise ValueError(
                f"clip {self.seq_id} was cached without scale statistics -- "
                "run scripts/patch_cache.py"
            )
        return self.stats.norm(mode, traj_scale)

    def normalised(self, mode: str = "median", traj_scale: float = TRAJ_SCALE):
        """(traj, anchor) in model units, under the chosen scheme."""
        n = self.norm(mode, traj_scale)
        return n.apply_traj(self.traj), n.apply(self.anchor)

    def to_dict(self) -> dict:
        d = {
            "seq_id": self.seq_id,
            "traj_metric": self.traj,
            "anchor_metric": self.anchor,
            "visibility": self.visibility,
            "query_uv": self.query_uv,
            "hw": self.hw,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "context": self.context,
            "id_card": self.id_card,
            "points": self.points,
            "feat_dim": self.feat_dim,
            "image_size": self.image_size,
        }
        if self.stats is not None:
            d.update(self.stats.to_dict())
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CachedClip":
        if "traj_metric" not in d:
            raise ValueError(
                "this cache stores normalised coordinates, not metres -- it "
                "predates the metres-in-cache format. Run scripts/patch_cache.py"
            )
        return cls(
            seq_id=d["seq_id"],
            traj=d["traj_metric"],
            anchor=d["anchor_metric"],
            visibility=d["visibility"],
            query_uv=d["query_uv"],
            hw=d["hw"],
            intrinsics=d.get("intrinsics"),
            extrinsics=d.get("extrinsics"),
            stats=ScaleStats.from_dict(d),
            context=d.get("context"),
            id_card=d.get("id_card"),
            points=d.get("points"),
            feat_dim=d.get("feat_dim"),
            image_size=d.get("image_size"),
        )
