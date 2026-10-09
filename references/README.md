# References

The short list. Each one has a folder here; `repo/` inside it is a git clone
and is gitignored. Older, longer notes are still in `docs/references.md`.

## 3D point tracking

| | paper | code | why it is on the list |
| --- | --- | --- | --- |
| **TAPIP3D** | arXiv 2504.14717 | `zbw001/TAPIP3D` | Our Kubric data and our metric both come from it. |
| **DELTA** | arXiv 2410.24211, ICLR 2025 | `snap-research/DELTA_densetrack3d` | The baseline the newest papers compare against most. Source of the Kubric3D numbers. |
| **SpatialTrackerV2** | arXiv 2507.12462 | `henry123-boy/SpaTrackerV2` | Current best on TAPVid-3D. Tracks and estimates depth in one model. |
| **4RC** | arXiv 2602.10094 | `Luo-Yihang/4RC` | Yihang's. 4D reconstruction from RGB only, which is where this project wants to end up. |

**How these were chosen (2026-10-09).** The two newest 3D tracking papers we
could find (arXiv 2609.34035 and 2605.12587) both compare against DELTA,
TAPIP3D and SpatialTrackerV2, and mention neither D4RT nor 4RC. The 4D
reconstruction papers (4RC, SM4RT) compare against each other instead. So these
are two families: trackers are given depth and camera, reconstruction models
estimate them. We are given both, so the trackers match our setup.

## What we have read out of them so far

**TAPIP3D** (`tapip3d/repo`)
- Picks which points to track the way CoTracker does: any point visible in the
  first or the middle frame (`configs/dataset/train/kubric_base.yaml`,
  `traj_mode`). No preference for moving points.
- Scores with the global median rescale, and with intrinsics resized so the
  short side is 256 px (`evaluation/metrics.py`). The pixel thresholds are
  defined at that size.
- Trains at 384x512 on 384 tracks per clip, with flips, crops, blur and
  colour jitter.

**DELTA** (`delta/repo`)
- Kubric3D, 143 test videos of 24 frames, a query on every pixel: AJ 81.4,
  APD3D 88.6. Dense queries, so not like-for-like with sparse ones.
- Predicts depth as `log(d_t / d_1)`.

**SpatialTrackerV2**, **4RC**: cloned, not read yet.
