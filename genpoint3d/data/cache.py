"""
The on-disk cache format -- one schema, written by `preprocess.py` and read by
`train.py`.

Why this file exists: the cache used to be a bare dict built in one script and
indexed by string in another. Nothing tied the two halves together, so a field
could be dropped on the write side and simply never noticed on the read side.
That is exactly what happened to `norm` -- without it, model output cannot be
turned back into metres, which made every distance-based metric meaningless.

Stored as a plain dict rather than a pickled dataclass so the file format does
not depend on this module's import path. `from_dict` tolerates missing optional
fields, so caches written before a field existed still load.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from genpoint3d.data.transform import NormStats


@dataclass
class CachedClip:
    """One preprocessed clip.

    Everything is in the frozen frame-0 camera frame. `traj` is in *model
    units*; `norm` is what converts it back to metres -- see
    `NormStats.invert_traj`.
    """

    seq_id: str

    # --- what the model denoises ---
    traj: torch.Tensor            # (T, N, 3) model units <- TARGET
    anchor: torch.Tensor          # (N, 3) scene-normalised frame-0 position
    visibility: torch.Tensor      # (T, N) bool, True = visible

    # --- query pointers ---
    query_uv: torch.Tensor        # (N, 2) pixel location in frame 0
    hw: tuple[int, int]           # frame size, to map uv into the feature grid

    # --- needed to score in metres, NOT used for training ---
    # `norm` inverts the per-clip normalisation; `intrinsics` gives the focal
    # length the TAP-Vid-3D thresholds are back-projected through.
    norm: Optional[NormStats] = None
    intrinsics: Optional[torch.Tensor] = None   # (T, 3, 3) pixel units

    # --- visual conditioning, absent for the step-2 (no-image) model ---
    context: Optional[torch.Tensor] = None      # (T, P, D) fp16 DINOv3 tokens
    id_card: Optional[torch.Tensor] = None      # (N, D) fp16, sampled at frame 0

    points: Optional[int] = None
    feat_dim: Optional[int] = None
    image_size: Optional[int] = None

    @property
    def traj_metric(self) -> torch.Tensor:
        """(T, N, 3) positions in metres. Raises if the cache predates `norm`."""
        if self.norm is None:
            raise ValueError(
                f"clip {self.seq_id} was cached without `norm`, so model units "
                "cannot be converted to metres -- re-run scripts/preprocess.py"
            )
        return self.norm.invert_traj(self.traj)

    def to_dict(self) -> dict:
        d = {
            "seq_id": self.seq_id,
            "traj": self.traj,
            "anchor": self.anchor,
            "visibility": self.visibility,
            "query_uv": self.query_uv,
            "hw": self.hw,
            "intrinsics": self.intrinsics,
            "context": self.context,
            "id_card": self.id_card,
            "points": self.points,
            "feat_dim": self.feat_dim,
            "image_size": self.image_size,
        }
        if self.norm is not None:
            # Flattened rather than nested so the .pt holds only tensors and
            # plain values -- no dataclass to import back.
            d["norm_mean"] = self.norm.mean
            d["norm_scale"] = self.norm.scale
            d["norm_traj_scale"] = self.norm.traj_scale
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CachedClip":
        norm = None
        if "norm_mean" in d:
            norm = NormStats(mean=d["norm_mean"], scale=d["norm_scale"],
                             traj_scale=d["norm_traj_scale"])
        return cls(
            seq_id=d["seq_id"],
            traj=d["traj"],
            anchor=d["anchor"],
            visibility=d["visibility"],
            query_uv=d["query_uv"],
            hw=d["hw"],
            norm=norm,
            intrinsics=d.get("intrinsics"),
            context=d.get("context"),
            id_card=d.get("id_card"),
            points=d.get("points"),
            feat_dim=d.get("feat_dim"),
            image_size=d.get("image_size"),
        )
