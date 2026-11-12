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
        ckpt.pt    the latest weights, with their settings and scores (none
                   between validations), so a run cut short keeps them
        best.pt    the weights with the best val APD so far
        resume.pt  what `--resume` goes on from: the step, the optimiser's
                   state (`Optimiser.state`), the log and the best so far
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
        (out / "run.json").write_text(json.dumps(
            {"commit": self.commit, "args": vars(self.settings),
             "clips": {"train": len(clips.train), "val": len(clips.val)}},
            indent=2))
        ckpt = {"model": model.state_dict(), "step": report.step,
                "args": vars(self.settings), "val": m, "commit": self.commit}
        torch.save(ckpt, out / "ckpt.pt")
        # The best val APD so far, kept apart from the latest.
        if m["average_pts_within_thresh"] > self.best:
            self.best = m["average_pts_within_thresh"]
            torch.save(ckpt, out / "best.pt")
            log.info(f"       new best APD {self.best:.3f}")


    def keep_resume(self, step: int, optimiser_state: dict) -> None:
        """resume.pt, written beside and moved over so a job stopped mid-write
        leaves the last one whole."""
        tmp = self.out / "resume.pt.tmp"
        torch.save({"step": step, "optimiser": optimiser_state, "log": self.log, "best": self.best,
                    "args": vars(self.settings)}, tmp)
        tmp.replace(self.out / "resume.pt")

    def resume(self, state: dict) -> None:
        self.log, self.best = state["log"], state["best"]

    def keep(self, model, step: int) -> None:
        """The latest weights to ckpt.pt between validations, without scores."""
        torch.save({"model": model.state_dict(), "step": step, "args": vars(self.settings), "val": None,
                    "commit": self.commit}, self.out / "ckpt.pt")


def _git_commit() -> str:
    """The commit this run is from, "-dirty" with uncommitted changes."""
    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5, cwd=repo)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5, cwd=repo)
        if out.returncode:
            return "unknown"
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:      # compute nodes have no git
        return "unknown"
