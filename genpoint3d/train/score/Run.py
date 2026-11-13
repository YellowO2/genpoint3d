"""A trained run loaded for scoring: its model, and its data and refinement
as it was trained."""

from dataclasses import dataclass
from pathlib import Path

import torch

from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.model.PointDiT import PointDiT
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.score.ScoreSettings import ScoreSettings


@dataclass
class Run:
    model: PointDiT
    settings: Settings
    step: int

    @classmethod
    def load(cls, path, device: torch.device) -> "Run":
        """The checkpoint at `path`, in eval mode on `device`."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        settings = Settings.saved(ckpt["args"])
        model = build_model(settings, ckpt["model"]["frame_proj.weight"].shape[1]).to(device)
        model.load_state_dict(ckpt["model"])
        return cls(model.eval(), settings, ckpt["step"])

    def dataset(self, clips, frames, points: int | None = None, query_before: int = 0, spawn: int = 0) -> ClipDataset:
        """`clips`, a cache directory (every clip in it) or a list of clip
        files, their first `points` points (the run's own count by default),
        with the frames in `frames`; with `spawn`, up to that many of them
        started where new surface comes into view, as training draws them."""
        if isinstance(clips, (str, Path)):
            clips = sorted(Path(clips).expanduser().glob("*.pt"))
        if not clips:
            raise SystemExit("no cached clips to score")
        return ClipDataset(clips, points or self.settings.points, resample=False, frames=Path(frames),
                           query_before=query_before, spawn=spawn)

    def judging(self, device: torch.device, cutoff: int = 0, **kw) -> ScoreSettings:
        """Scoring with the run's own refinement, from `cutoff` (0: every frame has its image)."""
        s = self.settings
        return ScoreSettings(device=device, amp=device.type == "cuda", window=s.window, overlap=s.overlap, schedule=s.schedule, cutoff=cutoff, **kw)
