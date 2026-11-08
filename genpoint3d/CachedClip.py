"""One preprocessed clip: the cache schema prepare writes and train reads.

Facts are cached, choices are not: positions in metres (normalised on load),
DINOv3 features at the backbone's own width. Stored as a plain dict, so the
files do not depend on this module's import path.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from genpoint3d.ScaleStats import ScaleStats


@dataclass
class CachedClip:
    """One preprocessed clip, in the frame-0 camera, positions in metres."""

    seq_id: str

    # --- ground truth, in metres ---
    traj: torch.Tensor            # (T, N, 3) absolute position
    anchor: torch.Tensor          # (N, 3) each point at its query frame
    visibility: torch.Tensor      # (T, N) bool, True = visible

    # --- query pointers ---
    query_uv: torch.Tensor        # (N, 2) pixel location in each point's query frame
    hw: tuple[int, int]           # frame size, to map uv into the feature grid
    query_t: Optional[torch.Tensor] = None   # (N,) each point's query frame; None is frame 0
    up: Optional[torch.Tensor] = None        # (3,) which way is up, in the frame-0 camera
    segment: Optional[torch.Tensor] = None   # (N,) each point's object, 0 the background; None if unknown

    # --- cameras ---
    intrinsics: Optional[torch.Tensor] = None   # (T, 3, 3) pixel units
    extrinsics: Optional[torch.Tensor] = None   # (T, 4, 4) cam_0 -> cam_t
    stats: Optional[ScaleStats] = None

    # --- DINOv3 ---
    context: Optional[torch.Tensor] = None      # (T, P, D) fp16 raw DINOv3 patch tokens
    id_card: Optional[torch.Tensor] = None      # (N, D) fp16, sampled at each query
    patch_xyz: Optional[torch.Tensor] = None    # (T, P, 3) fp32 metres, `context`'s order

    points: Optional[int] = None
    feat_dim: Optional[int] = None
    image_size: Optional[int] = None

    def to_dict(self) -> dict:
        d = {
            "seq_id": self.seq_id,
            "traj_metric": self.traj,
            "anchor_metric": self.anchor,
            "visibility": self.visibility,
            "query_uv": self.query_uv,
            "query_t": self.query_t,
            "up": self.up,
            "segment": self.segment,
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
            query_t=d.get("query_t"),
            up=d.get("up"),
            segment=d.get("segment"),
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
