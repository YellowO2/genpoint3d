# Overnight 2026-10-17

## Where we stood when the user went to sleep

All 24-grid tests: CNN on, 1,200 steps, moving points, val.

| Test | Result |
|---|---|
| Matcher off (run30) | 0.538 → 0.374: keep the matcher |
| 3D per 4 px cell in the CNN windows, `--fine-xyz 1` (run31) | 0.538 → 0.551 |
| Hidden frames in the loss, `--hidden-loss 1` (run32) | visible 0.551 → 0.529; hidden 0.148 → 0.170 |

Hidden points split (run31 / run32, val, moving):

| | Occluded | Out of frame |
|---|---|---|
| run31 | 0.171 | 0.089 |
| run32 | 0.190 | 0.113 |

About 28% of hidden entries are out of frame. Train ≈ val everywhere, so the
hidden-point problem is high bias.

Diagnosis (with the user): in a hidden frame every image input — DINOv3,
the matcher, the CNN windows, and the depth of the front-most surface — describes
the object in front of the point. Nothing tells the model it cannot see its
point. Plan: have the model predict "visible?" per (point, frame), where not
visible = occluded, out of frame, or a masked forecast frame.

48 grid, `--fine-xyz 1` (run33), at step 4,000 of 6,000: all points val 0.856 /
train 0.893, moving val 0.734, within 1 px 0.440. run26 (same, no `--fine-xyz`)
at the same step: 0.821 / 0.859, 0.686, 0.378. The 3D gain is much larger here
than in the 1,200-step 24-grid test, so short tests can underrate a change that
pays off late.

## How TAPIP3D does it (read from its code)

| Question | TAPIP3D |
|---|---|
| Starting guess | Sliding windows of 16 frames, stride 8. New frames start as a copy of the last estimate in the overlap; the first window starts at the query position. Same in training and inference. |
| Support points | Inference only: a 16×16 grid on frame 0, lifted to 3D from frame-0 depth, sliced off the output. Training uses none, but trains on 384 points per clip (we use 128). |
| Whole-scene view | No image-wide attention. Real tracks never attend to each other directly; they talk through 64 learned "virtual tracks". |
| Correlation reach | 3D kNN, 32 neighbours per level, 4 levels (4, 8, 16, 32 px cells). The coarsest level reaches roughly ±100 px around the estimate. |
| Visibility | Predicted every iteration as an update to a visibility logit, fed back as an input to the next iteration, BCE loss with gamma weighting. |
| Loss on occluded frames | Equal weight to visible frames. |

What this changed in the plan: "support points" in training means more points
per clip, so the support test is `--points 256` (TAPIP3D: 384), which needs no
new code. Feeding the visibility prediction back as an input is TAPIP3D's
design; our first version only predicts it.

## Runs queued tonight

| Run | What | Compared with |
|---|---|---|
| run34_pts256 | run32 + 256 points per clip (batch 4 × accum 4) | run32 |
| A (after the build) | run32 + "visible?" output | run32 |
| A48 (long queue, after run33) | run33 + "visible?" output + hidden loss, 6,000 steps | run33 |

Note (03:15): run34_pts256 runs at 6.8 s/step, so 1,200 steps need ~2.3 h and the
1:55 walltime will stop it around step 1,000. Its step-600 validation is still a
fair comparison with run32's step 600 (same total steps, so the same learning
rate there): run32 at 600 was moving 0.486, hidden 0.245 / moving hidden 0.149.

## TAPIP3D on hidden points (rescored from its saved predictions, 03:40)

Same 100 val clips, our scorer (`scripts/score_ref_preds.py`). Moving points, no rescale.

| | Visible APD | AJ | OA | Hidden | Occluded | Out of frame |
|---|---|---|---|---|---|---|
| TAPIP3D | 0.926 | 0.836 | 0.933 | **0.635** | 0.674 | 0.494 |
| ours, run31 (24 grid, val split) | 0.551 | — | — | 0.148 | 0.171 | 0.089 |
| ours, run32 (+ hidden loss) | 0.529 | — | — | 0.170 | 0.190 | 0.113 |

All points (protocol rescale): TAPIP3D APD 0.965, AJ 0.917, OA 0.966.

So hidden points are learnable: TAPIP3D places them about 4x better than we do.
Its hidden-point ingredients (from its code): equal-weight position loss on
occluded frames (we have that now), a visibility prediction fed back each
iteration, and a starting guess carried forward from the last estimate through
sliding windows. Ours starts every frame from the query position.
