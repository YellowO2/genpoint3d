# Results

Every training run, what was changed, and what it scored. The checkpoints live
on scratch and get purged; this file is the part that has to survive.

APD is `average_pts_within_thresh` from TAP-Vid-3D, scored in metres. Higher is
better. `apd_baseline` is the same metric on the mean trajectory, so it is the
score of having learnt nothing.

Runs are reproducible from the commit: `scripts/train.py` writes `run.json`
next to `log.json` with the commit and the full argument list.

## Summary

| run | commit | steps | val APD | test APD | change from previous |
| --- | --- | --- | --- | --- | --- |
| run3493 | `cabc639` | 30000 | 0.0512 | **0.0105** | first full-dataset run |
| run3493_apdloss | `0796ab2` | 30000 | *running* | — | metric-aligned loss, frame 0 pinned |

## run3493 — baseline

3144 train / 349 val Kubric clips, batch 16, 128 points, dim 256, depth 6,
4 heads, 30000 steps, cosine schedule after 500 warmup steps.

Loss: mean squared error on the flow-matching velocity, every visible point
weighted equally.

| step | train | val | p1 | p16 | APD |
| --- | --- | --- | --- | --- | --- |
| 2000 | 0.3131 | 0.0357 | 0.0000032 | 0.0198 | 0.0046 |
| 10000 | 0.0246 | 0.0184 | 0.000017 | 0.0571 | 0.0135 |
| 20000 | 0.0092 | 0.0165 | 0.000093 | 0.1551 | 0.0385 |
| 26000 | 0.0064 | 0.0178 | 0.00014 | 0.2039 | **0.0516** |
| 30000 | 0.0059 | 0.0174 | 0.00014 | 0.2012 | 0.0512 |

Test set (300 held-out clips, sequences 3500+), scoring `best.pt` from step
20000: APD **0.0105**, baseline 0.0001, val_loss 0.0360.

`best.pt` is from step 20000 rather than 26000 because this run predates the
switch to selecting on APD -- it was still minimising the homemade `ratio`.

Per-frame APD on the test set fell monotonically, 0.0276 at frame 0 to 0.0071
at frame 23. Frame 0 was the best frame and still scored 0.028, despite being
handed its own position as `anchor`. That measurement is what motivated pinning
it.

Read: learning is real -- APD rose monotonically and beat the baseline by 100x
-- but precision never arrived. `pts_within_1` stayed near 1e-4 for the whole
run, meaning almost nothing was tracked accurately; only the loosest threshold
moved. Overfitting from step 16000 (train 0.0151 against val 0.0174).

## run3493_apdloss — metric-aligned loss

Identical to run3493 apart from three changes, deliberately at the same 30000
steps so the schedule is unchanged and the comparison is clean.

1. Loss weighted by `focal / depth`, putting the error in units of the metric's
   threshold rather than metres. TAPIP3D's `scale_loss_by_depth`.
2. Frame 0 pinned to the anchor instead of denoised, with a zero target
   velocity, and pinned again at every sampling step.
3. Euclidean distance (`l21`) rather than mean squared error, so the loss
   follows the typical point rather than the worst one.

| step | train | val | p1 | p16 | APD | vs run3493 |
| --- | --- | --- | --- | --- | --- | --- |
| 2000 | 0.3109 | 0.1873 | 0.0015 | 0.1132 | 0.0295 | 6.5x |
| 4000 | 0.1383 | 0.1571 | 0.0029 | 0.1522 | 0.0419 | 3.6x |
| 6000 | 0.1120 | 0.1323 | 0.0028 | 0.3945 | 0.1268 | 12x |
| 8000 | 0.1008 | 0.1290 | 0.0027 | 0.3188 | 0.0962 | 10x |
| 10000 | 0.0924 | 0.1190 | 0.0052 | 0.4410 | **0.1506** | 11x |
| 12000 | 0.0868 | 0.1282 | 0.0038 | 0.4042 | 0.1378 | 5.5x |

Loss values are not comparable across the two runs -- `l21` and the depth
weighting change the magnitude. Only APD means the same thing in both.

Read: by step 10000 this run is already at three times run3493's final score,
and `pts_within_1` is 37x run3493's final. Precision, which never moved before,
is moving. Validation is noisy at plus or minus 30 percent between points, so
individual values should not be read closely.

Not yet attributed: three changes went in together. Each is a flag
(`--depth-scaled-loss`, `--anchor-frame0`, `--loss-type`), so bisecting is
cheap once there is a reason to.

## Reference points

Nobody publishes a sparse-3D-tracking number on Kubric, so these are the
nearest comparisons rather than like-for-like.

| | task | metric | score |
| --- | --- | --- | --- |
| 4RC | Kubric, DENSE per-pixel, Sim(3)-aligned | APD | 85.44 |
| V-DPM | same | APD | 71.12 |
| TAPIP3D | LSFOdyssey, synthetic | AJ3D | 72.2 |
| TAPIP3D | TAPVid-3D, real | AJ3D | ~30 |
| TAPIR | TAP-Vid-Kubric, 2D | delta_avg | 93.99 |

4RC's protocol differs from ours in three ways that all favour it: dense rather
than sparse points, global Sim(3) alignment rather than median rescaling, and
an unstated threshold. Treat it as an anchor for the scale of the number, not
as a target we are directly behind.
