# Runs

One `log.json` per run, copied off scratch before it is purged. The code is not
copied -- `run.json` in the run's output directory records the commit, and
`train.py` stamps it into the checkpoint too, so `git show <commit>` is the
source of truth for what produced a number.

APD is `average_pts_within_thresh` from TAP-Vid-3D, scored in metres. Higher is
better.

| run | commit | steps | best val APD | test APD | what changed |
| --- | --- | --- | --- | --- | --- |
| `run3493` | `cabc639` | 30000 | 0.0516 | 0.0105 | first full-dataset run, 3144 train clips |
| `run3493_apdloss` | `0796ab2` | 30000 | *running* | — | depth-weighted loss, frame 0 pinned, l21 instead of l2 |

Notes that are not obvious from the numbers:

- `run3493`'s `best.pt` is step 20000, not its best-APD step 26000 -- it
  predates selecting checkpoints on APD and was still minimising the homemade
  `ratio`.
- Loss values are not comparable between these two runs. `l21` and the depth
  weighting change the magnitude; only APD means the same thing in both.
- `run3493_apdloss` bundles three changes, so none of them is attributed. Each
  is a flag: `--depth-scaled-loss`, `--anchor-frame0`, `--loss-type`.
