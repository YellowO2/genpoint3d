"""Run a scoring script with the rigid proposal switched off in the model it
loads: the same weights, no proposal given. A score that does not move says
the model ignores the proposal.

Run:  python scripts/without_rigid.py scripts/evaluate.py --ckpt ... (the script's own arguments)
"""

import runpy
import sys

from genpoint3d.train.score.Run import Run

_load = Run.load.__func__


def _without(cls, path, device):
    run = _load(cls, path, device)
    run.model.rigid_proposal = False
    return run


Run.load = classmethod(_without)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
