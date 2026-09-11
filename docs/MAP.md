# The map

**Open this first.** It is the architecture, the file layout, and the progress
tracker in one place.

Project: extend *Gen-points* (Lu, Cao, Feng, Owens, CVPR 2026 — generative
point tracking **and** forecasting) from 2D to 3D. Debugging on Kubric
synthetic data before real video.

---

## Where we are

```
   RGB video + depth + camera pose              query points
   ───────────────┬────────────────             (2D pixel + frame index)
                  │                                      │
   ┌──────────────▼───────────────────────────────────┐  │
   │ [0] INPUT PREP                            DONE   │  │
   │     genpoint3d/data/kubric.py    read files      │  │
   │     genpoint3d/data/transform.py reframe+scale   │  │
   │     genpoint3d/data/masking.py   hide frames     │  │
   └──────────────┬───────────────────────────────────┘  │
                  │                                      │
   ┌──────────────▼───────────────────────────────────┐  │
   │ [1] VISUAL ENCODER  DINOv3-S, frozen             │  │
   │     genpoint3d/models/encoder.py   NOT STARTED   │  │
   └──────┬──────────────────────┬────────────────────┘  │
          │                      │                       │
   feature map C_t         sample at Q                   │
   "what is where"         "who am I"                    │
          │                      │                       ▼
          │        ┌─────────────▼──────────┐  ┌──────────────────────┐
          │        │ [3] CONDITIONING c     │  │ [2] POINT TOKENISER  │
          │        │  ID card + Q pos-emb   │  │  3D positions        │
          │        │  + noise level k       │  │  -> tokens           │
          │        │  models/model.py       │  │  models/model.py     │
          │        │  DONE (no ID card yet) │  │  DONE                │
          │        └─────────────┬──────────┘  └──────────┬───────────┘
          │                      │                        │
          │                      └────────┬───────────────┘
          │                               ▼
          │        ┌──────────────────────────────────────────────┐
          └───────►│ [4] DiT BLOCKS  x depth        ◄── YOU ARE    │
    (or ∅ if the   │     a. temporal attention          HERE       │
     frame is      │     b. spatial attention                      │
     masked)       │     c. point-image cross-attn  NOT STARTED    │
                   │     models/model.py + models/layers.py        │
                   │     DONE (a and b), UNTESTED                  │
                   └──────────────────┬───────────────────────────┘
                                      ▼
                   ┌──────────────────────────────────────────────┐
                   │ [5] OUTPUT HEADS         models/model.py     │
                   │     position   -> 3 xyz        DONE, UNTESTED│
                   │     visibility -> 1 occluded?  NOT STARTED   │
                   └──────────────────────────────────────────────┘
```

Only **[2]–[5] are trained**. [1] is frozen off-the-shelf. Training objective:
`models/flow.py` (not a stage — it scores the output).

## Build order

| Step | What | Status |
| --- | --- | --- |
| 1 | Input prep — [0] | **DONE**, 4 checks pass, committed |
| 2 | Backbone + flow matching, **no images** — [2][4a][4b][5-position] | **code written, unproven** |
| 3 | DINOv3 + cross-attention — [1][3-ID card][4c] | not started |
| 4 | Visibility head — [5-visibility] | not started |

Step 2 is the de-risk: an *unconditional* trajectory generator. If it cannot
memorise 2 clips, the fault is the backbone or the flow-matching code — and
finding that out with an encoder attached would be miserable.

**Step 2 remaining:** run `scripts/overfit.py` and confirm loss → 0. Blocked on
compute (not running on the user's Mac). Already verified: 11.93M params,
causal masking exact, gradients flow.

---

## File layout

```
genpoint3d/              the package
  data/                    [0] input prep
    kubric.py                reads Kubric files -> raw numpy
    transform.py             raw -> model-ready (reframe, normalise, scale)
    masking.py               which frames get hidden (tracking vs forecasting)
  models/                  [1]-[5] the network
    model.py                 THE ARCHITECTURE — open this one
    layers.py                the machinery (RMSNorm, RoPE, Attention, ...)
    flow.py                  training objective + sampler
    encoder.py               [1] DINOv3 — step 3
  geometry.py              3D<->2D maths (project / unproject)
  viz/rerun_log.py         3D visualisation

scripts/                 things you run
  check_transform.py       proves [0] is correct (4 checks)
  calibrate_traj_scale.py  measures the TRAJ_SCALE constant
  overfit.py               step 2's test: can it memorise 2 clips?
  visualize.py             writes the .rrd you open in rerun

docs/MAP.md              this file
docs/references.md       every paper/repo and why
docs/ideas.md            deferred ideas
data/ outputs/ refs/     gitignored
```

---

## Stage detail

### [0] Input prep

Three operations, in order, then the inverse for turning predictions back into
metres. Verified by `scripts/check_transform.py`.

1. **Reframe** → the *frozen frame-0 camera* frame. `rel[t] = E[t] @ inv(E[0])`.
2. **Normalise** → per-clip, from observed frames only (percentile depth clip,
   inlier centroid, max centred norm). Makes a tabletop and a street land in
   the same box.
3. **Scale the target** → one global `TRAJ_SCALE`, so the denoising target has
   unit variance.

### [1] Visual encoder — *step 3*

DINOv3-S, frozen, whole frame at once — it never sees the query points. Take
multiple layers, concatenate, project to `D`, upsample 2×.

**3D addition, the "feature cloud":** each patch token also gets its 3D
position (downsample depth to the patch grid, unproject, Fourier-encode, add).
TAPIP3D's construction.

### [2] Point tokeniser

`nn.Linear(3, D)` on the noisy 3D position. Point-space diffusion — positions
in and out directly, no VAE, no grid. That is the whole stage.

### [3] Conditioning vector `c`

One vector per query point, constant across frames and blocks:

```
c = [DINOv3 feature at Q]  +  [pos-emb of Q]  +  [emb of noise level k]
         step 3                  ── have these now ──
```

Enters through **AdaLN** (`AdaRMSNorm`): it predicts a per-channel scale for
the norm inside every block. Differs from vanilla DiT, where one *global*
vector conditions all tokens — ours is **per point**.

> `c` is the point's fixed **ID card** — *what* we track. It cannot say *where
> the point is in frame t*; that is what we are predicting. Finding "where" is
> [4c]'s job.

### [4] DiT blocks

Tokens form a `(T frames) × (N points)` grid, so attention is factorised:

| | attention | who talks to whom | reshape |
| --- | --- | --- | --- |
| a | **temporal**, causal | one point across time; frame `t` sees only `t' ≤ t` | `(B·N, T, D)` |
| b | **spatial** | all `N` points within one frame | `(B·T, N, D)` |
| c | **cross-attn** — *step 3* | each point queries its frame's feature map | — |

RMSNorm, QK-norm, GEGLU FFN. RoPE on time; axial RoPE on query xyz.

**[4c] is the tracking↔forecasting switch.** Masked frames get the learned null
embedding `∅` instead of `C_t`. Same machinery, no branching.

> Why cross-attention and not correlation features: correlation needs a
> template matched against a real image. In forecasting there *is* no image.
> Cross-attention just swaps in `∅` and keeps running.

Paper sizes: 6 blocks / dim 384 (main), 12 / 768 (Kinetics). Ours: 11.93M
params at dim 256 / depth 6 / 4 heads.

### [5] Output heads

A "head" is a small output layer converting a hidden vector into the format we
want.

- **Position** `Linear(D → 3)` — predicts the velocity `x1 − x0`, L2 loss.
- **Visibility** `Linear(D → 1)` — predicts clean `V1`, BCE loss. *Step 4.*

Visibility means **geometric** occluded/off-screen, predicted for *every* frame
including forecast frames. A forecast point is not automatically invisible.

---

## Training

**Conditional flow matching.** `x_k = (1−k)·x0 + k·x1`, `x0 ~ N(0,I)`; predict
`x1 − x0`. `k` from a **logit-normal (loc −1, scale 1.5)** — *not* uniform; the
paper reports this as critical, and SD3 found the same.

**The switch that controls everything** is whether frame `t` has visual
conditioning. `t < T_C` → tracking (conditional). `t ≥ T_C` → features replaced
by `∅` → forecasting (unconditional). `T_C = T` is pure tracking, `T_C = 1`
pure forecasting. No architectural change, just a mask.

Deferred past step 4: diffusion forcing (independent `k` per frame),
autoregressive sliding window, point-count factorization.

---

## Locked decisions

**Frozen frame-0 camera frame.** Everything expressed in frame 0's camera, never
re-anchored. Static like a world frame (ego-motion removed), but constructible
from *relative* poses alone — which is all real video gives you. Costs the
gravity prior ("down" is no longer a fixed axis). MotionForesight freezes the
*last observed* frame instead; we use frame 0 because our `T_C` is random per
sample and a moving origin would make targets depend on the mask.

**Camera pose is an input, never predicted, never a token.** Used only as
geometry: unproject depth → 3D, project 3D → 2D for feature sampling.

**Absolute positions, not residuals.** Tried MotionForesight's residual
parameterisation, dropped it — they get away with it because they denoise in
VAE latent space. The anchor is kept as conditioning instead.

**Two scales.** Per-clip scene scale for geometry; one global `TRAJ_SCALE` for
the denoising target. Global, not per-clip, so a fast clip stays genuinely
faster than a slow one.

**Query = 2D pixel + source frame index.** The pixel is a *pointer* — used to
sample the DINOv3 ID card and look up depth for the 3D anchor. The tokens the
model denoises are 3D.

**Feature-cloud conditioning in from the first real test.** Free on Kubric (GT
depth + pose). "Remove pointmap conditioning" is a planned ablation.

---

## Where the code comes from

The paper's own repo is **empty** ("coming soon"), so:

| Piece | Source |
| --- | --- |
| DiT block, AdaRMSNorm, RoPE, QK-norm, GEGLU, zero-init, temporal/spatial factorization | **`tesfaldet/genpt`** `src/models/networks/transformer_genpt.py` |
| feature cloud / 3D lifting | **`TAPIP3D`** |
| cross-check only | `facebookresearch/DiT` (ICCV'23), `co-tracker` |

**On genpt:** arXiv 2510.20951 — authors include Adam Harley (PIPs /
PointOdyssey, who defined modern point tracking) and Chris Pal (Mila). Its
transformer descends from HDiT (Crowson et al., **ICML 2024**). We take
*engineering* from it, not claims — every mechanism was chosen from the CVPR
spec first.

**Not reused:** genpt injects image info as correlation-pyramid features through
AdaRMSNorm. We need cross-attention so `∅` works for forecasting.

---

## Still open

1. Exact config for the first real test (currently dim 256 / depth 6 / 4 heads).
2. Spatial attention **per block** (our paper, current choice) vs **once at the
   end** (genpt, cheaper). Ablate later.
3. Virtual points on/off (CoTracker3's `O(N²)` saver). Currently off.
4. Recalibrate `TRAJ_SCALE` — 0.2136 came from only 2 clips.
