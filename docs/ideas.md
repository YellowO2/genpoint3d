# Ideas, open questions, deliberate omissions

What the diagram cannot show: things not built, and things undecided. The
diagram in `map.html` covers what exists.

## Not built yet

Decided to skip for now, with the reason.

- Visibility head — `Linear(D -> 1)` + BCE, occluded or off-screen per frame.
  `model.py` has only `Linear(dim, 3)` for position. AJ and OA both score
  occlusion, so neither is reportable until this exists.
- Autoregressive sliding window — generate W frames, slide by stride S, feed
  the previous window back as a causal prefix. How the paper handles video
  longer than one window. We do all 24 frames at once, which only works because
  Kubric clips are exactly 24 frames.
- Diffusion forcing — independent noise level per frame. `flow.py` accepts
  `per_frame_k` already, it is just off. Needed for stable autoregressive
  rollout.
- Point-count factorization — reshape N points into random factor pairs each
  step so the model handles any query count.
- dt conditioning — ours, not in the paper. The model sees frame index, never
  elapsed time, so frame rate is baked into the motion prior. Feeding dt
  alongside the noise level would make it frame-rate agnostic.

## To do next

- Walk through what the model actually sees at each denoising step -- which
  tensors enter, through which path (tokens, AdaLN, cross-attention), and what
  is shared across the 50 steps versus recomputed. Requested 2026-09-18.
- Displacement target instead of absolute. TRAJ_SCALE would be 0.0992 rather
  than 0.8344, measured on 200 clips. Read how TAPIP3D and MotionForesight
  parameterise and scale it rather than deriving it again. Caveat: per-clip
  motion std spans 0.021 to 0.320, a 15x range, so a single global scale leaves
  the slowest clips below check_transform's 0.3 floor.
- Verify the claim that MolmoMotion's autoregressive variant beats its
  flow-matching one. If true it is a big deal; it may also be specific to
  having an LLM backbone and 1.16M videos.

## Open questions

Undecided. Each needs an experiment, not a discussion.

- Spatial attention per block (our paper's choice, what we do) versus once at
  the end (genpt, cheaper).
- Virtual points, CoTracker3's O(N^2) saver. Currently off.
- Config is unsettled: dim 256 / depth 6 / 4 heads is what trained, not what
  was chosen. TAPIP3D trains on ~28x our samples, so size may not be the lever.
- Normalisation statistic — `median` distance to camera is the default, `mean`
  and `centroid_max` are cached too. One flag apart, never compared.
- `TRAJ_SCALE` is 0.8344 from 200 clips. Cheap to redo on all 3493 now the
  cache stores metres.
- Per-clip units cost the gravity prior. In "fraction of this scene" units a
  falling ball accelerates at 4.9 in a 2 m room and 0.49 on a 20 m street, so
  the model cannot learn one universal "things fall like this" — it must infer
  scene scale first. Real loss for forecasting, where falling dominates.
  Kubric gives ground-truth metres, so both arms are trainable on identical
  data; on real video the metric arm is impossible, so this is the only chance
  to measure it. If metric wins big: predict scene scale as an auxiliary
  output, feed a scale token, or normalise by something physically anchored
  like estimated camera height.
- Masking only whole frames may be leaving capability on the table. A superset
  scheme — sometimes whole frames, sometimes parts of a frame, MAE-style —
  would cover tracking, forecasting, and tracking through a visual dropout in
  one model, and partial 3D occlusion conditioning is not well explored.
  Whole-frame masking stays a subset of the training distribution, so the
  standard benchmark still applies. Costs: steps spent on a regime benchmarks
  do not reward, no standard eval for the new capability, and spatial masks
  instead of a per-frame boolean.
