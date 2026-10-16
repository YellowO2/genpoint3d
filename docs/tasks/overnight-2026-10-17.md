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

### Is TAPIP3D's hidden score just "keep the last place you saw it"? No. (04:00)

Moving points, same 100 clips. The baselines use the TRUE position (and speed)
at the last visible frames, so they are generous.

| | Hidden | Occluded | Out of frame |
|---|---|---|---|
| TAPIP3D | **0.635** | 0.674 | 0.494 |
| last seen, held still | 0.188 | 0.220 | 0.115 |
| last seen, kept at its last speed | 0.267 | 0.301 | 0.212 |
| ours (run31 / run32, 349 val clips) | 0.148 / 0.170 | 0.171 / 0.190 | 0.089 / 0.113 |

1. TAPIP3D is far above both carry-forward baselines, so it really works out where
   hidden points go. Carrying the last estimate forward is not what makes it good.
2. We are BELOW "hold the last place you saw it". The model does worse than not
   moving a hidden point at all: something actively pulls hidden points to the
   wrong place. That fits the diagnosis that every image input in a hidden frame
   describes the object in front.
3. Where TAPIP3D's extra skill can come from: other points on the same object
   that are still visible (it trains on 384 points per clip, tracks talk through
   virtual tracks), and its visibility estimate fed back into every iteration, so
   it knows when to ignore the image. Tonight's runs test exactly these two:
   more points (run34_pts256) and a visibility output (run35_vis, run36_vis_48).

Queued (03:40): run37_visfb = run35_vis + `--vis-feedback 1` (the visibility
guess fed back into the next refinement round, TAPIP3D's design; commit 2badd0c).
Compared with run35_vis and run32_hidden.

## Short-run results (04:35)

24 grid, CNN + 3D, hidden loss 1, moving points, val. Same total length (1,200), so
step 600 is comparable across runs.

| Run | Step | Visible | Hidden | Occluded | Out of frame | Train hidden | Visible? acc (hidden / visible) |
|---|---|---|---|---|---|---|---|
| run32_hidden | 600 | 0.486 | 0.149 | — | — | — | — |
| run34_pts256 (256 points) | 600 | 0.488 | 0.153 | 0.174 | 0.092 | 0.155 | — |
| run35_vis ("visible?" output) | 600 | 0.468 | 0.140 | 0.158 | 0.086 | 0.144 | 0.67 / 0.98 |
| run37_visfb (+ fed back) | 600 | 0.459 | 0.134 | 0.153 | 0.084 | 0.137 | 0.66 / 0.98 |
| run32_hidden | 1200 | 0.529 | 0.170 | 0.190 | 0.113 | 0.173 | — |
| run35_vis | 1200 | 0.511 | 0.161 | 0.182 | 0.101 | 0.164 | 0.71 / 0.98 |

None of the three helps hidden points; the visibility ones cost a little on
visible points. The model learns to tell hidden from visible fairly well (71% of
hidden cases at 1,200), but knowing it does not change where it puts the point.
Short runs underrated `--fine-xyz` before, so this is not final, but there is no
early sign.

Remaining suspect, by elimination: the image inputs on hidden frames — above all
the matcher, which hands the model an explicit "the point is here" that, on a
hidden frame, points at the object in front. TAPIP3D has no matcher. Next check
(evaluation only, no training): blank every image input on hidden frames, the way
forecast frames are blanked, and see whether hidden scores jump.

Re-plan (04:40): the "visible?" output showed no gain and a small cost, so I
cancelled the long 48-grid visibility run (run36_vis_48, it had not started) and
queued instead **run38_fc_xyz_48**: the best tracker (48 grid, CNN + 3D, as
run33) trained and scored in forecasting mode (random cutoff, eval cutoff 12).
That gives the report its forecasting numbers on our best tracker, whatever the
hidden-point work turns out to need.

## Blanking the image on hidden points does not help (05:00)

`--ablate hide-hidden` (evaluation only, uses the truth): every image input of a
(point, frame) that is truly hidden is blanked, exactly as a forecast frame is.
Moving points, val, 349 clips. Last seen (true position held): 0.204.

| Model | Hidden, normal | Hidden, image blanked |
|---|---|---|
| run32_hidden (trained on tracking only; never saw blanked frames) | 0.170 | 0.088 |
| run29_fc_rg24_fine (trained with forecast blanking, so it knows blanked frames) | 0.167 | 0.164 |

The forecasting-trained model, which understands blanked inputs, scores the same
with the image removed. So the misleading image is not what keeps hidden points
low; my earlier diagnosis was wrong, or at least not the main cause. With or
without eyes, the model places a hidden point about as well as holding its own
last estimate.

That is the same thing we saw in forecasting: the future part barely beats
freezing its own last guess. Hidden frames and future frames fail the same way:
**when it cannot see, the model does not reason about motion at all.** That
supports the user's view that occlusion and forecasting are one problem.

TAPIP3D's 0.635 is far above "keep the last speed" (0.267), so it does reason
when blind. The likely source is the other points of the same object that are
still visible (a hidden point on a visible ball can be placed from the ball), but
256 points per clip did not help us in 600 steps. The open question for the
morning: is that skill something we lack in the design, or does it only appear
with much longer training? run38_fc_xyz_48 (long, forecasting mode) will show
whether the future part starts to beat freezing with long training.

## The 48-grid run with 3D in the CNN windows finished: train passes 0.9 (05:15)

run33_fine_xyz_48 (48 grid, CNN + 3D per 4 px cell, 6,000 steps, no hidden loss),
val split 349 clips, in-training scores:

| Step | All points val / train | Moving val / train | <1 px | <2 px | Moving hidden |
|---|---|---|---|---|---|
| 1,000 | 0.724 / 0.762 | 0.562 / 0.556 | 0.246 | 0.392 | 0.143 |
| 2,000 | 0.788 / 0.810 | 0.638 / 0.649 | 0.312 | 0.487 | 0.182 |
| 3,000 | 0.832 / 0.872 | 0.699 / 0.731 | 0.387 | 0.572 | 0.223 |
| 4,000 | 0.856 / 0.893 | 0.734 / 0.782 | 0.440 | 0.624 | 0.253 |
| 5,000 | 0.865 / 0.909 | 0.749 / 0.810 | 0.469 | 0.649 | 0.267 |
| **6,000** | **0.866 / 0.916** | **0.754 / 0.820** | **0.478** | **0.656** | **0.271** |
| run26, same without 3D | 0.833 / 0.877 | 0.702 / 0.763 | 0.400 | 0.580 | — |

- Train passes 0.9 (0.916): the target from the Week 8 email.
- Hidden points improve steadily with long training even with no hidden loss:
  0.143 → 0.271, above "hold last seen" (0.204) and close to "keep last speed"
  (0.284). So the blind-reasoning skill does grow with training; the 1,200-step
  tests were far too short to show it. TAPIP3D is at 0.635.
- The train–val gap is now visible (moving 0.820 vs 0.754).
- A standard-scoring run on 100 clips (like run26's 0.832) is queued for the report.

Queued after this (05:10): run39_noncausal = run32 + `--causal 0` (time attention
both ways). TAPIP3D looks both ways inside its 16-frame windows, so a hidden
point between two visible stretches can be placed from where it reappears; ours
only looks back. For tracking this is legitimate (published trackers do it).

**Number for the report** (standard scoring, 100 clips, all 256 points, like
run26's 0.832): run33_fine_xyz_48 **val 0.878 / train 0.928**
(<1/2/4/8/16 px val 0.695 / 0.837 / 0.916 / 0.957 / 0.984). TAPIP3D 0.965.
Previous best run26: 0.832 / 0.894. Week-8 starting point: 0.46 / 0.58.
Moving hidden on those clips: val 0.283, train 0.327 (last seen 0.195, last speed 0.274).

## Last two short runs (06:40)

| Run | Step 1,200: visible / hidden / occluded / out of frame | Note |
|---|---|---|
| run32_hidden (reference) | 0.529 / 0.170 / 0.190 / 0.113 | |
| run37_visfb (visibility output fed back) | 0.508 / 0.158 / 0.179 / 0.101 | AJ 0.577, finds 71% of hidden |
| run39_noncausal (time attention both ways) | 0.527 / 0.171 / 0.193 / 0.108 | |

Both flat. Even seeing the frames after a point reappears does not help it in
1,200 steps.

## Morning summary

1. **Report number:** run33 (48 grid, CNN + 3D per 4 px cell) scores **0.878 val /
   0.928 train** in standard scoring (run26 was 0.832 / 0.894; TAPIP3D 0.965).
   Train passes the 0.9 target. Within 1 px on moving points: 0.478 (run26 0.400).
2. **Hidden points are learnable:** TAPIP3D places moving hidden points at 0.635,
   far above "keep the last speed" (0.267), so it really reasons when blind.
3. **None of tonight's short (1,200-step) ideas moved hidden points:** 256 points,
   "visible?" output, visibility fed back, two-way time attention. Blanking the
   image on hidden points does not help either, so the misleading image was not
   the main cause.
4. **But long training does:** run33's hidden score rose steadily from 0.143 to
   0.271 over 6,000 steps with no hidden loss at all (0.283 on the 100-clip
   standard check). Short tests cannot judge this skill.
5. **Hidden and future frames fail the same way:** when the model cannot see, it
   places points about as well as holding its own last estimate. That matches the
   user's view that occlusion and forecasting are one problem.

Still running: run38_fc_xyz_48 (best tracker in forecasting mode, ~10 h left).

Suggested next decision for the user: test the hidden-point ideas only in long
runs (48 grid, 6,000 steps). Since run33 already gives the baseline, the most
informative single long run is probably run33 + `--hidden-loss 1 --vis-loss 1
--vis-feedback 1 --causal 0` (everything TAPIP3D does that we now have), compared
with run33's 0.271 hidden. If that moves hidden clearly, split it up afterwards.
