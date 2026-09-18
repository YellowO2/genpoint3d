# Runs

One `log.json` per run.

APD is `average_pts_within_thresh` from TAP-Vid-3D, scored in metres. Higher is
better.

| run | commit | steps | best val APD | test APD | what changed |
| --- | --- | --- | --- | --- | --- |
| `run3493` | `cabc639` | 30000 | 0.0516 | 0.0105 | first full-dataset run, 3144 train clips |
| `run3493_apdloss` | `0796ab2` | 30000 | 0.2100 | — | depth weighted loss + frame 0 starting pinned + l21 instead of l2 |

`compare.png` overlays every run: `python scripts/plot_runs.py runs/*.json`
