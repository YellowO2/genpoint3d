# Runs

One `log.json` per run.

APD is `average_pts_within_thresh` from TAP-Vid-3D, scored in metres. Higher is
better.

| # | run | commit | steps | best val APD | test APD | what changed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `run3493` | `cabc639` | 30000 | 0.0516 | 0.0105 | first full-dataset run, 3144 train clips |
| 2 | `run3493_apdloss` | `0796ab2` | 30000 | 0.2100 | — | depth weighted loss + frame 0 starting pinned + l21 instead of l2 |
| 3 | `run3493_disp` | `ef50476` | 30000 | 0.2487 | — | per-point displacement target instead of absolute |

`compare.png` overlays every run: `python runs/plot.py runs/*.json`
Legend labels are positional -- run 1 is the first file on the command line.
