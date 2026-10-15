"""One preprocessed clip: the format prepare writes and train reads.

The on-disk cache format -- one schema, written by `genpoint3d/prepare` and
read by `genpoint3d/train`.

Two rules, both learned the hard way.

**Cache facts, not choices.** The trajectory is stored in METRES, and the
normalisation is applied when a clip is loaded. Storing normalised coordinates
baked a training decision into the data, so every change to the scheme -- max
to median, centred to camera-origin -- invalidated hours of preprocessing.
Metres are a fact about the clip; how to normalise them is a knob. The same rule
governs `patch_xyz`, and the reason features are now cached at the backbone's raw
width: a projection is a choice, so it belongs to training, not to the data.

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

from genpoint3d.NormStats import NormStats
from genpoint3d.ScaleStats import ScaleStats
from genpoint3d.tools.traj_scale import TRAJ_SCALE


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
    # RAW backbone features, at DINOv3's own width -- no projection, no position
    # folded in. Both of those are learnable, and anything cached cannot learn;
    # they live in `PointDiT` now. See `prepare/VisualEncoder.py`.
    context: Optional[torch.Tensor] = None      # (T, P, D) fp16 raw DINOv3 patch tokens
    id_card: Optional[torch.Tensor] = None      # (N, D) fp16, sampled at frame 0
    # Where each patch is, in METRES, matching `context`'s patch order. The model
    # embeds and adds this itself, so it can learn how much to weigh "where"
    # against "what" -- a sum baked in before caching fixes that ratio forever.
    patch_xyz: Optional[torch.Tensor] = None    # (T, P, 3) fp32 metres

    points: Optional[int] = None
    feat_dim: Optional[int] = None
    image_size: Optional[int] = None

    def norm(self, mode: str = "median", traj_scale: float = TRAJ_SCALE) -> NormStats:
        if self.stats is None:
            raise ValueError(
                f"clip {self.seq_id} was cached without scale statistics -- "
                "re-run python -m genpoint3d.prepare.main"
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
            "patch_xyz": self.patch_xyz,
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
                "predates the metres-in-cache format. Re-run python -m genpoint3d.prepare.main"
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
            patch_xyz=d.get("patch_xyz"),
            points=d.get("points"),
            feat_dim=d.get("feat_dim"),
            image_size=d.get("image_size"),
        )
