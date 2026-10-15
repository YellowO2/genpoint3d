"""Write a run's log, settings and checkpoints, keeping the best apart."""

import json
import logging
import subprocess
from pathlib import Path

import torch

from genpoint3d.train.Report import Report
from genpoint3d.train.data.Clips import Clips

log = logging.getLogger(__name__)


class Saver:
    """Everything a run leaves in `--out`:

        log.json   one entry per validation, a plain list `plot_log.py` reads
        run.json   the settings, the commit and the clip counts
        ckpt.pt    the latest weights, with their settings and scores
        best.pt    the weights with the best val APD so far
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self.out = Path(settings.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.commit = _git_commit()
        self.log = []
        self.best = -float('inf')

    def save(self, report: Report, model, clips: Clips) -> None:
        out, m = self.out, report.scores
        self.log.append({"step": report.step, "train_loss": report.train_loss, **m})
        (out / "log.json").write_text(json.dumps(self.log, indent=2))
        # Config and commit alongside the curve, in their own file so
        # log.json stays a plain list that plot_log.py can read.
        (out / "run.json").write_text(json.dumps(
            {"commit": self.commit, "args": vars(self.settings),
             "clips": {"train": len(clips.train), "val": len(clips.val)}},
            indent=2))
        ckpt = {"model": model.state_dict(), "step": report.step,
                "args": vars(self.settings), "val": m, "commit": self.commit}
        torch.save(ckpt, out / "ckpt.pt")
        # Keep the best separately: val typically bottoms out and then
        # drifts up as the model overfits, so the LAST checkpoint is
        # not the one you want.
        # Maximise APD, as genpt's model_checkpoint does
        # (monitor: val/d_all_avg/avg, mode: max).
        if m["average_pts_within_thresh"] > self.best:
            self.best = m["average_pts_within_thresh"]
            torch.save(ckpt, out / "best.pt")
            log.info(f"       new best APD {self.best:.3f}")


def _git_commit() -> str:
    """The commit this run is from, so a number can always be traced to code.

    train.pbs prints this to the live log, but that log lives on scratch and is
    purged. A result whose code cannot be identified is not reproducible, so it
    goes in the log and the checkpoint too.
    """
    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5, cwd=repo)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5, cwd=repo)
        if out.returncode:
            return "unknown"
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        # Compute nodes have no git on PATH; an unknown commit is not worth
        # killing a run over.
        return "unknown"
