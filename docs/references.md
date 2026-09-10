# Reference papers & code

## The FYP paper (target to extend to 3D)
- **Generative Point Tracking and Forecasting** — Lu, Cao, Feng, Owens. CVPR 2026.
  - Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Generative_Point_Tracking_and_Forecasting_CVPR_2026_paper.html
  - Project page: https://gen-points.github.io/
  - Official code: https://github.com/Charles-Lu/Generative-Point-Tracking-and-Forecasting
    (EMPTY as of 2026-09-10 — "code & weights coming soon"; re-check periodically)
  - Local copy of full text: `../Generative Point Tracking and Forecasting.txt`
  - Architecture spec extracted: `architecture_spec.md`

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
