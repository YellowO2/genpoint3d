# Gen-points 3D — architecture spec

Extending: Lu, Cao, Feng, Owens. *Generative Point Tracking and Forecasting*,
CVPR 2026. (Full text: `../Generative Point Tracking and Forecasting.txt`.)

Their official code is unreleased. We assemble from other repos — see
`## Where the code comes from`.

---

## The map — read this first

```
   RGB video (T frames)                      query points Q
   + depth + camera pose                     (2D pixel + frame index)
            │                                         │
            ▼                                         │
  ╔═════════════════════════╗                         │
  ║ 1. VISUAL ENCODER       ║                         │
  ║    DINOv3-S  (FROZEN)   ║                         │
  ║    → feature map per fr ║                         │
  ╚════════════╤════════════╝                         │
               │                                      │
        ┌──────┴───────┐                              │
        │              │                              │
        ▼              ▼                              ▼
   feature map    sample at Q              ╔═════════════════════════╗
   C_t per frame  = "ID card"              ║ 2. POINT TOKENISER      ║
   (what's where)  (what am I)             ║    noisy 3D traj →      ║
        │              │                   ║    T×N tokens           ║
        │              ▼                   ╚════════════╤════════════╝
        │      ╔═══════════════════╗                    │
        │      ║ 3. COND VECTOR c  ║                    │
        │      ║  ID card          ║                    │
        │      ║  + pos-emb of Q   ║───────┐            │
        │      ║  + timestep k     ║       │            │
        │      ╚═══════════════════╝       │            │
        │                                  ▼            ▼
        │                        ╔═══════════════════════════════╗
        └───────────────────────►║ 4. DiT BLOCKS  × L            ║
          (or ∅ if frame masked) ║    a. temporal attention      ║
                                 ║    b. spatial attention       ║
                                 ║    c. point–image cross-attn  ║
                                 ║    (all AdaLN-modulated by c) ║
                                 ╚═══════════════╤═══════════════╝
                                                 ▼
                                 ╔═══════════════════════════════╗
                                 ║ 5. OUTPUT HEADS               ║
                                 ║    position   → 3  (xyz)      ║
                                 ║    visibility → 1  (occluded?)║
                                 ╚═══════════════════════════════╝
```

Only **stages 2–5 are trained**. Stage 1 is frozen off-the-shelf.

---

## Stage 0 — what goes in and what comes out

Learn `p_theta(P, V | I_C, Q)`:

| symbol | shape | meaning |
| --- | --- | --- |
| `I_C` | `T_C × H × W × 3` | conditioning video — only the first `T_C` frames are shown |
| `Q` | `N × 2` (+ frame idx) | query points, as **pixels** in their source frame |
| `P` | `T × N × 3` | 3D trajectories to generate ← *was 2 in the paper* |
| `V` | `T × N` in `[0,1]` | per-point visibility to generate |

**The one switch that controls everything** is whether frame `t` has visual
conditioning:

- `t < T_C`: image features given → **tracking** (conditional generation)
- `t ≥ T_C`: features replaced by a learned null embedding `∅`
  → **forecasting** (unconditional — samples the motion prior)

`T_C = T` → pure tracking. `T_C = 1` → pure forecasting. In between → track
then forecast. No architectural change, just a mask.

---

## Stage 1 — Visual encoder  (frozen)

**What:** turn each RGB frame into a grid of feature vectors, one per image
patch. Each vector describes *what is at that spot* ("furry orange cat-ear").

**Why frozen:** DINOv3 already knows what things look like. Training it would
cost a lot and gain nothing at our scale. Zero trainable params here.

- **DINOv3-S** ViT, frozen.
- Resize shorter side to **768 px** (paper) — we start smaller.
- Take **multiple layers**, concatenate, project to model dim `D`.
- **Upsample 2×** (nearest + conv) → per-frame map `C_t ∈ R^{H'×W'×D}`.

**3D addition — the "feature cloud":** each patch token also gets its **3D
world position**. Downsample depth to the patch grid, unproject with the
camera pose, Fourier-encode the XYZ, add to the patch feature. Now the
feature map knows not just *what* is at each patch but *where it is in
3D*. (TAPIP3D's construction.)

---

## Stage 2 — Point tokeniser

**What:** turn the trajectory being denoised into `T × N` tokens — one token
per (frame, point).

- Point-space diffusion: positions go in and out **directly**, no VAE, no grid.
- Input per token = noisy 3D position (+ noisy visibility) at that frame.
- One `nn.Linear` up to dim `D`.

That's the whole stage. It is genuinely just an embedding layer.

---

## Stage 3 — Conditioning vector `c`  (feeds AdaLN)

**What:** build one vector per query point that says *who this point is and
how noisy we currently are*. It never changes across frames or blocks.

`c_n = [ DINOv3 feature sampled at Q_n ]  +  [ pos-emb of Q_n ]  +  [ emb of timestep k ]`

**Differs from vanilla DiT:** vanilla DiT has one *global* conditioning
vector (timestep/class) for all tokens. Ours is **per point**.

**How it enters the network:** AdaLN — the vector produces scale/shift values
that modulate the normalisation inside every block. Think of it as a set of
knobs turned on the token, not a conversation.

> **Why this isn't enough on its own** (common confusion): `c` is the point's
> fixed **ID card** — *what* we're tracking. It cannot say *where the point is
> in frame t*, because that is exactly what we're predicting. Finding "where"
> is stage 4c's job.

---

## Stage 4 — DiT blocks  × L

Tokens live on a `(T frames) × (N points)` grid, so attention is **factorised**
into cheap slices instead of one giant `T·N` attention.

Each block, in order:

| | attention | who talks to whom | reshape |
| --- | --- | --- | --- |
| **a** | **temporal** (causal) | one point, across time — frame `t` sees only `t' ≤ t` | `(B·N, T, D)` |
| **b** | **spatial** | all `N` points within one frame | `(B·T, N, D)` |
| **c** | **point–image cross-attn** | each point token queries its frame's feature map `C_t` | — |

Plus RMSNorm, QK-norm, and an FFN. Positional info via RoPE (temporal) and
axial RoPE indexed by query location (spatial).

**4c is the tracking↔forecasting switch.** For masked frames, `C_t` is replaced
by the learned null embedding `∅`. Same machinery, no branching.

> **Why cross-attention and not correlation features:** correlation needs a
> template matched against an actual image. In forecasting there *is* no image,
> so correlation is undefined. Cross-attention just swaps in `∅` and keeps
> running. (genpt uses correlation because it is tracking-only.)

**Sizes:** paper main = 6 blocks / dim 384. Kinetics variant = 12 / 768.
Our first test ≈ 10M params.

**Optional cost saver:** replace spatial attention's `O(N²)` with `V` learnable
*virtual points* the tokens read/write through — CoTracker3's trick, already
implemented in genpt.

---

## Stage 5 — Output heads

A "head" is just a small output layer converting a hidden vector into the
format we want.

- **Position:** `Linear(D → 3)`, predicts the flow vector `v = x1 − x0`, L2 loss.
- **Visibility:** `Linear(D → 1)`, predicts the clean `V1`, BCE loss. Take the
  last sampling step's prediction as final.

Visibility means **geometric** occluded/off-screen vs in-view. It is predicted
for *every* frame including forecast frames — a forecast point is not
automatically "not visible".

---

## Training

### Conditional flow matching (position)
- Path `x_k = (1−k)·x0 + k·x1`, `x0 ~ N(0,I)`, `x1` = ground truth.
- Loss `E‖ F_theta(x_k, k, c) − (x1 − x0) ‖²`.
- Normalise trajectories to zero mean / unit variance.
- **`k` from a logit-normal (loc = −1, scale = 1.5)** — *not* uniform. The
  paper reports this as critical.

### Diffusion forcing
Independent noise level `k` **per frame**. Breaks reliance on a perfect
history, which is what makes autoregressive rollout stable.

### Autoregressive sliding window
Generate `W` frames, slide by stride `S`. The previous window's last `W − S`
frames become a causal prefix. At inference the prefix is **re-noised slightly**
so the model doesn't over-trust its own past output.
Defaults: tracking stride 8 / context noise 0.15; forecasting stride 1 / 0.02.

### Point-count factorization
Factor `N_train = a·b`; each step pick a pair and reshape into `a` samples of
`b` points → robust to arbitrary `N`.

### Task conditioning
- tracking-only: visual condition always present
- forecasting-only: visual condition on the query frame only
- **unified:** pick a random frame, mask all visual input after it
  (`masking.py`)

### Optimizer
AdamW, grad accumulation, grad clipping, EMA. 200k steps (paper).

---

## 3D extension — decisions

### What changes from the 2D paper
1. Position head `2 → 3`; token position channels `2 → 3`.
2. A coordinate frame must be chosen (2D pixel space needed none).
3. Feature sampling needs a **3D → 2D projection** first (`batch_project`).
4. Loss/normalisation move from pixels to metres.
5. Metrics move to metric 3D thresholds (TAPIP3D's 3D-AJ). Eval only.

### LOCKED

**Coordinate frame — frozen frame-0 camera.**
Express everything in the camera frame of **frame 0**, then never re-anchor.
Static across time (ego-motion removed) so it behaves like a world frame, but
the origin is a camera, not the scene.
*Why not Kubric's world origin:* a scene origin doesn't exist on real video —
only relative poses do. Frame-0 anchoring is always constructible, and it
canonicalises viewpoint so the model needn't learn rotation invariance.
*Cost:* "down" is no longer a fixed axis (gravity prior lost).
*Note:* MotionForesight freezes the **last observed** frame; we use frame 0
because our cutoff `T_C` is random per sample, and a moving origin would make
targets depend on the mask. One matmul to switch — cheap ablation.

**Camera pose — an input, never predicted, never a token.**
Used purely as geometry: (1) unproject depth → 3D points (`batch_unproject`),
(2) project 3D points → 2D for feature sampling (`batch_project`).
Plücker/explicit camera tokens are novel-view-synthesis lineage — skip.

**Residual parameterisation.** Predict `delta = P(t) − anchor(t=0)`, then add
the anchor back. Never predict absolute position. (MotionForesight.)

**Normalisation from observed frames only.** Clip depth to a percentile band,
`mean` = inlier centroid, `scale` = max centred norm, then `(x − mean)/scale`.
(MotionForesight `_compute_pj_norm`.)

**Scene geometry conditioning — IN from the first real test.**
Feature-cloud form (stage 1). Free on Kubric (GT depth + pose). Swap in a depth
model for real data later. "Remove pointmap conditioning" = planned ablation.

**Query representation.** Query = 2D pixel + source frame index (TAP-Vid
convention). The pixel is a *pointer*: used to sample the DINOv3 ID card and to
look up depth for the initial 3D anchor. The tokens the model denoises are 3D.

**Visibility head — KEEP**, but add it *after* the position-only smoke test.
One `Linear(D→1)` + one BCE term, fully isolated.

---

## Where the code comes from

Our CVPR paper's repo is **empty** ("coming soon"). So:

| Piece | Source | Why that one |
| --- | --- | --- |
| DiT block, AdaRMSNorm, AxialRoPE, QK-norm, GEGLU, zero-init | **`tesfaldet/genpt`** `src/models/networks/transformer_genpt.py` | modern (2025) *and* already shaped for point tracks |
| temporal/spatial factorization, virtual points | same file, `Transformer` | CoTracker3's trick, already implemented |
| flow matching loss, noise schedules | `genpt` `helpers/{losses,noise_schedules}.py` | |
| DINOv3 encoder wrapper | `genpt` `networks/dinov3_convnext.py` | same encoder choice as us |
| feature cloud / 3D lifting | **`TAPIP3D`** | the 3D half |
| geometry utils | `genpoint3d/geometry.py` (ported from TAPIP3D) | done |
| Kubric loading + masking | `genpoint3d/data/{kubric,masking}.py` | done |
| **cross-check reference** | `facebookresearch/DiT` (ICCV'23), `facebookresearch/co-tracker` | peer-reviewed sanity check on AdaLN structure |

**On genpt's credibility:** arXiv 2510.20951, authors incl. **Adam Harley**
(PIPs / PointOdyssey — defined modern point tracking) and **Chris Pal** (Mila).
Its transformer strongly resembles **HDiT** (Crowson et al., **ICML 2024**);
they cite k-diffusion in `src/utils/ema_schedules.py:71`. We take *engineering*
from it, not scientific claims — and every mechanism was independently chosen
from the CVPR spec first.

**Not reused from genpt:** their conditioning path. They inject image info as
correlation-pyramid features through AdaRMSNorm (`input_cond`); we need
cross-attention so `∅` works for forecasting. Their `AttentionBlock` already
accepts `context_dim`, so cross-attention is available, just unused upstream.

---

## Build order

| Step | Build | Repos needed | Passes when |
| --- | --- | --- | --- |
| **1** | data transform → frozen-cam-0, residual, normalised tensors | none | round-trips + looks right in rerun |
| **2** | DiT backbone + flow matching, **no visual conditioning at all** | genpt | overfits 2 clips, loss → 0 |
| **3** | DINOv3 + feature cloud + point–image cross-attn | genpt, TAPIP3D | loss drops further |
| **4** | visibility head | none | BCE trains |

Step 2 is the key de-risk: an *unconditional* trajectory generator. If that
can't memorise 2 clips, the generative machinery is broken — and debugging that
with an encoder attached would be miserable.

---

## Still open

1. Exact config for the ~10M first test (blocks / dim / heads).
2. Whether to keep spatial attention **per block** (our paper) or **once at the
   end** (genpt — cheaper). Default: per block, ablate later.
3. Virtual points on or off for v1. Default: off (`N` is small).
4. Diffusion forcing / sliding window / point-count factorization — defer past
   step 4.
