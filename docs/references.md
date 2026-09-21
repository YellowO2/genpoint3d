# Reference papers & code

## Lookup table — the facts we keep re-deriving

Checked against the actual source, not the papers. Clone paths are the session
scratchpad. Add a row rather than re-scraping.

| | coord frame | task-causal | attn-causal | token INPUT | TARGET | generative |
| --- | --- | --- | --- | --- | --- | --- |
| **Gen-points** (ours to extend) | — (2D pixels) | yes | yes | — | absolute | flow matching |
| **genpt** | — (2D raster) | no | no | displacement fwd+bwd | absolute | flow matching |
| **TAPIP3D** | world (depth+pose lifted) | no | no | displacement fwd+bwd | absolute (iterative) | no, regression |
| **MotionForesight** | frozen **last-observed** cam | **yes** | **no** | absolute pointmaps | **residual from frame-0** | video diffusion, 1-step |
| **MolmoMotion** | frozen **frame-0** cam (`t0`) | yes | yes | text tokens | absolute | autoregressive LLM |
| **ours** | frozen **frame-0** cam | yes | yes | absolute (← odd one out) | **displacement from frame-0** | flow matching |

**Two meanings of "causal", do not confuse them.**
*Task-causal* = the future is not in the input at all. *Attention-causal* = a
mask inside the transformer. MotionForesight is task-causal but NOT
attention-causal: its future slots hold learnable mask latents, so bidirectional
attention has nothing to leak. **You only need an attention mask if the future
slots contain information worth leaking.** Ours exists for autoregressive
rollout and diffusion forcing, not for leak prevention.

**Consequence for us:** TAPIP3D and genpt feed forward *and* backward frame
differences. The forward one (`coords[t] - coords[t+1]`) looks at the future.
They can afford it — they are trackers with the whole video. We cannot, because
we chose an attention mask. If we adopt displacement input we take only the
past-looking difference, `coords[t] - coords[t-1]`.

### Where to look, by question

| question | file:line |
| --- | --- |
| genpt: what enters the transformer | `genpt/src/models/networks/genpt_fm.py:344-364` |
| genpt: iterative refinement update | `genpt_fm.py:368-375` |
| genpt: noise sigma per quantity | `genpt_fm.py:50-55` (`p_1_coords_sigma=0.25`, vis/conf `1.0`) |
| genpt: coords normalised to raster | `genpt/src/data/tapvid_kubric_subseq_dataset.py:32` |
| genpt: the transformer itself | `genpt/src/models/networks/transformer_genpt.py` |
| TAPIP3D: updater input construction | `TAPIP3D/models/point_tracker_3d.py:185-220` |
| MotionForesight: mask latents for future frames | `motionforesight/models_pretrained/future_scene_flow/model.py:123-129, 248-263` |
| MotionForesight: residual target | `model.py:238-245, 322-330` |
| MotionForesight: per-clip normalisation | `motionforesight/.../sparse_dataset.py` `_compute_pj_norm` |
| MotionForesight: reframe to a frozen camera | `sparse_dataset.py:1-15` |
| MolmoMotion: coordinate frame | `molmo-motion/README.md:37` |

### Scratchpad clones
`genpt`, `TAPIP3D`, `DiT`, `co-tracker`, `motionforesight`, `molmo-motion`, `4RC`
under the session scratchpad. Re-clone with `git clone --depth 1` if gone:
`tesfaldet/genpt`, `zbw001/TAPIP3D`, `facebookresearch/DiT`,
`facebookresearch/co-tracker`, `brains-bots-n-behavior/motionforesight`,
`allenai/molmo-motion`, `Luo-Yihang/4RC`.

---


## Published numbers, and why none is like-for-like

Nobody publishes a sparse 3D point-tracking number on Kubric, so these are the
nearest comparisons rather than targets we are directly behind.

| | task | metric | score |
| --- | --- | --- | --- |
| 4RC | Kubric, DENSE per-pixel, Sim(3)-aligned | APD | 85.44 |
| V-DPM | same | APD | 71.12 |
| TraceAnything | same | APD | 59.98 |
| St4RTrack | same | APD | 50.65 |
| TAPIP3D | LSFOdyssey, synthetic | AJ3D | 72.2 |
| TAPIP3D | TAPVid-3D, real | AJ3D | ~30 |
| TAPIR | TAP-Vid-Kubric, 2D | delta_avg | 93.99 |

4RC's Kubric protocol differs from ours in three ways that all favour it: dense
per-pixel rather than sparse query points, global Sim(3) alignment by RANSAC
rather than median rescaling, and a threshold their paper does not state. Use
it to judge the scale of the number, not the size of the gap.

AJ is stricter than APD -- it also requires the visibility prediction to be
right -- so a method's APD is always at or above its AJ.

sota2.com hosts a leaderboard for the 4RC Kubric benchmark but reports 4RC at
APD 55.38 where the paper says 85.44. Do not cite it.

---

## The FYP paper (target to extend to 3D)
- **Generative Point Tracking and Forecasting** — Lu, Cao, Feng, Owens. CVPR 2026.
  - Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Generative_Point_Tracking_and_Forecasting_CVPR_2026_paper.html
  - Project page: https://gen-points.github.io/
  - Official code: https://github.com/Charles-Lu/Generative-Point-Tracking-and-Forecasting
    (EMPTY as of 2026-09-10 — "code & weights coming soon"; re-check periodically)
  - Local copy of full text: `../Generative Point Tracking and Forecasting.txt`
  - Architecture spec extracted: `map.html`

## Closest concurrent work (has full code — read, don't clone)
- **Generative Point Tracking with Flow Matching** — Tesfaldet, Harley, Derpanis,
  Nowrouzezahrai, Pal. arXiv 2510.20951 (Oct 2025).
  - Code: https://github.com/tesfaldet/genpt  (src/, configs/, checkpoints, MIT)
  - Same core method (video-conditioned flow-matching point generation), different authors.

## 3D point tracking — coordinate-frame reference
- **D4RT: Efficiently Reconstructing Dynamic Scenes One D4RT at a Time** —
  Zhang et al. (Google DeepMind / UCL / Oxford). CVPR 2026 **Best Paper**.
  - Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Efficiently_Reconstructing_Dynamic_Scenes_One_D4RT_at_a_Time_CVPR_2026_paper.html
  - arXiv: https://arxiv.org/abs/2512.08924
  - Project page: https://d4rt-paper.github.io/
  - Coordinate frame: **flexible** — query carries a `t_cam` reference-frame index;
    the model can output points in the camera frame of *any* timestep or in a
    single consistent world frame. Evaluates BOTH "camera coordinate tracking"
    and "world coordinate tracking" as separate protocols (their Table 4).
- **TAPIP3D** (already used for the data pipeline) — 3D TAP; works internally in a
  world frame (lifts depth + camera pose to a stationary point cloud), evaluated
  in both camera and world coordinates.

## Motion forecasting predecessor (the forecasting benchmark Gen-points builds on)
- **What Happens Next? Anticipating Future Motion by Generating Point Trajectories**
  — Boduljak, Karazija, Laina, Rupprecht, Vedaldi (Oxford VGG). ICLR 2026.
    Ref [9] in the Gen-points paper.
  - arXiv: https://arxiv.org/abs/2509.21592
  - Coordinate space: **2D pixel space only** — "a point trajectory is a sequence
    of 2D pixel coordinates". Trajectory tensor `x in R^{H/s x W/s x T x 2}`.
    No world/camera-frame question because it is not a 3D method.

## 3D point forecasting — the most directly comparable work
- **MolmoMotion: Forecasting Point Trajectories in 3D with Language Instruction**
  — Zhang, Zheng, Yang et al. (AI2 / UW / UNC-Chapel Hill).
  - Project page: https://molmomotion.github.io/
  - Code: https://github.com/allenai/molmo-motion
  - Dataset: https://huggingface.co/datasets/allenai/molmo-motion-1m
  - Benchmark: https://huggingface.co/datasets/allenai/PointMotionBench
  - **Coordinate frame: metric WORLD frame.** "MolmoMotion predicts each point's
    future 3D trajectory in a metric world frame." Data pipeline "lifts dense 2D
    tracks into a shared metric 3D frame."
  - **Camera pose: NOT estimated.** Uses depth to lift 2D tracks to 3D; does not
    predict camera parameters.
  - Has both autoregressive and flow-matching decoder variants -> closest existing
    design to "Gen-points in 3D". This is the reference to copy for the
    coordinate-frame decision.

- **MotionForesight: Re-purposing Video Models for Future 3D Scene-Flow
  Prediction** — Bharadhwaj, Jangir (Johns Hopkins).
  - Project page: https://motionforesight.github.io/
  - Code: https://github.com/brains-bots-n-behavior/motionforesight
  - Data: Something-Something-V2 (40K videos)
  - **3D point forecasting** for robot manipulation. Predicts "future 3D
    trajectories for points on the manipulated object" as **metric 3D tracks**.
  - **Camera pose: recovered in the data-curation preprocessing** ("camera
    motion, aligned pointmaps" from monocular video), NOT predicted by the
    trajectory model. Same pattern as MolmoMotion.

## 2D motion forecasting (camera handled by preprocessing, not the model)
- **Forecasting Motion in the Wild** — Thakkar, Ginosar, Walker, Malik, Carreira,
  Doersch. ECCV 2026.
  - Project page: https://motion-forecasting.github.io/  (code "coming soon")
  - 2D pixel space, after **homography-based camera stabilization** to remove
    camera motion in preprocessing.

## 4D reconstruction family (flexible query frame, camera estimated — different task)
- **D4RT** — see "3D point tracking" section above. CVPR 2026 Best Paper.
- **4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere** —
  Luo, Zhou, Lan, Pan, Loy (NTU S-Lab). "ARC".
  - Project page: https://yihangluo.com/projects/4RC/
  - Code: https://github.com/Luo-Yihang/4RC
  - arXiv: https://arxiv.org/abs/2602.10094
  - 4D reconstruction from monocular video; conditional decoder queries "3D
    geometry and motion for any query frame at any target timestamp" — same
    query-any-frame design as D4RT. Camera pose handling not stated in abstract;
    treat as the D4RT-style full-reconstruction category, not a trajectory-
    forecasting reference.

## Component code sources (for adapting the architecture)
- DiT block / AdaLN-Zero / TimestepEmbedder / FinalLayer:
  https://github.com/facebookresearch/DiT  (ref [57], Peebles & Xie)
- Point-track spatial + temporal attention, sliding window:
  https://github.com/facebookresearch/co-tracker  (CoTracker3, refs [38,39])
- DINOv3 encoder (frozen):
  https://github.com/facebookresearch/dinov3  (ref [66])
- Feature upsampling for DINO-based tracking:
  https://github.com/gorkaydemir/track_on  (Track-On, Aydemir et al., ref [2])
- RoPE / axial RoPE:
  https://github.com/lucidrains/rotary-embedding-torch
- Flow matching loss / logit-normal timestep sampling / ODE sampler:
  https://github.com/facebookresearch/flow_matching
- Diffusion forcing (per-frame independent noise):
  Chen et al., "Diffusion Forcing" (ref [12]) — official repo

## Curated lists for finding newer work
- https://github.com/amusi/CVPR2026-Papers-with-Code
- https://github.com/SkalskiP/top-cvpr-2026-papers
- https://github.com/colorfulfuture/Awesome-Trajectory-Motion-Prediction-Papers
- https://github.com/topics/point-tracking

## Coordinate-frame survey — conclusion
| Paper | 2D/3D | Frame | Camera pose |
| --- | --- | --- | --- |
| Gen-points (FYP paper) | 2D | pixel | not used |
| What Happens Next? (Boduljak) | 2D | pixel | not used |
| Forecasting Motion in the Wild (Doersch) | 2D | pixel (post-stabilization) | removed in preprocessing |
| **MolmoMotion** | **3D** | **metric world** | **given (depth-lift), not predicted** |
| **MotionForesight** | **3D** | **metric world / 3D** | **recovered in preprocessing, not predicted** |
| D4RT | 3D/4D | flexible (query picks) | predicted as output |
| 4RC | 3D/4D | flexible (query any frame) | full-reconstruction task |
| TAPIP3D | 3D | world (internal), eval both | given |

**Decision: follow MolmoMotion + MotionForesight (the two directly comparable 3D
point-forecasting papers — both agree) — output in a metric world frame; camera
pose is an input (Kubric GT for us), the model does NOT predict it.** The 4D
reconstruction family (D4RT, 4RC) makes the frame a query parameter and estimates
the camera, but that is a bigger/different task (full 4D reconstruction from bare
monocular video) and out of scope.

---

## Benchmark numbers — what "good" looks like

**TAPVid-3D** (TAPIP3D paper, Table 1). Averaged over Aria / DriveTrack /
PStudio, monocular RGB with *estimated* depth:

| method | AJ3D | APD3D | OA |
| --- | --- | --- | --- |
| TAPIP3D | 18.8 | 27.4 | 86.4 |
| DELTA | 17.8 | 26.3 | 86.4 |
| CoTracker3 + M-SaM | 17.3 | 25.9 | 87.8 |
| SpatialTracker | 13.0 | 20.8 | 84.5 |

**State of the art is AJ3D ~19 / 100.** The task is far from solved; do not
expect high numbers.

**Not comparable to us**: those subsets are real video with ESTIMATED depth.
Much harder than our setting.

**Kubric3D is the right comparison** -- same simulator, 24 frames, RGB-D with
ground-truth depth (DELTA paper, Table 3):

| method | AJ | APD3D | OA |
| --- | --- | --- | --- |
| DELTA | 81.4 | 88.6 | 96.6 |
| DOT-3D | 72.3 | 77.5 | 88.7 |
| SpatialTracker | 42.7 | 51.6 | 96.5 |

With GT depth the numbers are **80-90**, not ~19. Their eval set is 143 videos
of 24 frames at 384x512 -- structurally near-identical to our val split.

**DELTA trains on 5,632 Kubric videos. We have 322.** A 17x data gap, which is
the concrete basis for "we are data-limited, not architecture-limited".

**Gen-points** (our paper) reports Kubric *2D pixel* tracking: delta_avg ~64,
AJ ~53, OA ~85-88. Also not comparable -- 2D, different split, 200k steps.

### The metrics, defined
From `TAPIP3D/evaluation/tapvid3d_metrics.py`:
- `pts_within_{1,2,4,8,16}` -- fraction of points within a threshold. The
  thresholds are the 2D TAP pixel thresholds **back-projected into 3D using
  depth and intrinsics**, so a distant point gets a larger tolerance. NOT
  fixed metric distances.
- `average_pts_within_thresh` = APD3D (a.k.a. delta_avg).
- `jaccard_{x}` -> `average_jaccard` = AJ3D. Counts a point only if it is both
  within threshold AND correctly predicted visible.
- `occlusion_accuracy` = OA.
- Predictions are rescaled to the GT scale first (`scaling="median"`) because
  monocular depth is scale-ambiguous. With Kubric GT depth we would use
  `scaling="none"`.

**TODO:** port this implementation so our results become comparable. Our
current `delta_avg` in `scripts/train.py` uses arbitrary fixed thresholds and
is NOT the same measure.
