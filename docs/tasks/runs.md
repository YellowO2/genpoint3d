# Run tracker

Fast setup unless noted: 24-frame Kubric clips, 128 points, 2,500 steps,
batch 8. Score: APD on visible moving points (val), higher is better.
TAPIP3D's released model scores 0.926 on the same points.

## Current thread: what does our tracker lack against TAPIP3D?

| Run | What it is | Step 500 | Best / end | Verdict |
|---|---|---|---|---|
| fx1 | DINOv3 + CNN, Fourier ruler on window offsets | 0.505 | 0.643 | ruler helps |
| fx7 | CNN only, gradient clip 10 (was 1) | 0.315 | 0.566 | clip was a handbrake |
| fx8 | fx1 + clip 10 | 0.528 | 0.653 | small gain |
| fx9 | fx7 + Fourier motion and pixel input | 0.346 | 0.592 | motion input helps |
| fyC | CNN only, TAPIP3D's 3D-neighbourhood matching with its trained weights, our transformer | 0.693 | 0.733 (1500) | current best |
| fyD | fyC + 64 virtual tracks | 0.694 | | no gain |
| fyE | fyC + 256 points per clip | 0.697 | | no gain |
| fyF | fyC with TAPIP3D's updater (transformer) | 0.698 | | no gain |
| fyG | fyC + visibility loss and feedback | 0.686 | | no gain |
| fyH | fyC + colour and blur augmentation | 0.695 | | no gain |
| tap_short | TAPIP3D's own code, unchanged, on our train clips at our budget (stopped at step ~2400) | 0.137 | 0.546 | worse than ours at equal budget: its 0.926 is long training |
| fyI | fyC + DINOv3 (cross-attention and whole-frame match) | 0.699 | 0.723 (1000) | no gain, cancelled |
| fyJ | fyC + velocity output, velocity loss, rigidity loss, hidden frames in the loss | 0.666 (hidden moving 0.231 vs 0.156) | 0.756 (1500; hidden moving 0.317 vs 0.186) | better on both, new best |
| fyK | fx8 (DINOv3 + our own CNN) with each point's cross-attention reading only the 49 patches nearest its guess, not all 576 | 0.524 | 0.599 (1000, same as fx8), stopped | whole-frame search not needed: same score, about 12x less cross-attention |
| fyL | fyJ tracked in 16-frame windows sharing 4 frames, each reading frame 0 as its template, instead of carry | stopped at start | | skipped: windows assumed fine, used in fyM and fyN |
| fyM | fyJ in 16-frame windows sharing 4 (not carry), trained on tracking and forecasting together: each clip keeps the images of a random number of frames; validated with frames 12-23 blank | forecast moving 0.286 (own-freeze 0.266) | 1500: forecast moving 0.316 (own-freeze 0.286, true-freeze 0.321, true last speed 0.317); seen moving 0.856; hidden moving 0.462 | forecast vs stop / constant-speed baselines; seen frames vs fyJ |
| fyN | fyM + DINOv3 (whole-frame match, cross-attention over the 49 nearest patches) | forecast moving 0.302 vs 0.286; seen moving 0.781 vs 0.793 | 1500: forecast moving 0.333 vs 0.316 (true-freeze 0.321, true last speed 0.317); seen moving 0.856 vs 0.856; hidden moving 0.463 vs 0.462 | DINOv3 helps the forecast, tracking unchanged |

| fyO | fyN at 6,000 steps (the joint model, long; for the weekly report) | | stopped by its 12 h limit after step 5,000; best.pt = step 5,000 | standard scoring: val 0.911 / train 0.919 (last week 0.878 / 0.928), moving hidden val 0.401 (0.283); forecast (frames 12-23 blank): moving future 0.380 (freeze own 0.306, freeze true 0.322, true last speed 0.310; last week 0.293), seen moving 0.892 |
Standard scoring (100 clips, all 256 points, like run33's 0.878 / 0.928): fyJ at step 1,500
val 0.871 / train 0.876; moving hidden val 0.316 (run33 0.283; last seen 0.188, last speed 0.267).

## Diagnosis on fyC (scoring only)

- Precision on moving points falls every frame (0.97 at frame 1 to 0.55 at
  frame 23, APD); points that do not move stay near perfect.
- No undershoot: the predicted motion is 98-100% of the true motion.
- Plain and corner-like points drift about the same.
- 6 refinement rounds instead of 4: +0.008. A 16x16 support grid at test: no change.
- Error against distance moved: ours grows in proportion (about 3-4% of the
  distance past 16 px); TAPIP3D released barely grows (0.6% at 128+ px);
  TAPIP3D at our budget is proportional too (7-8%). Snapping onto the exact
  point comes with long training.

## Reader probe (scoring only, `scripts/probe_reader.py`)

CoTracker3's CNN frozen; only a small reader trained (3,000 steps) to read the
four levels of 49 x 49 tables and say how far the true point is from a guess
pushed off by up to 16 px. One read, no refinement rounds. Val, 96k visible
moving point-frames:

| | Median | Mean | <1 px | <2 px | <4 px | <8 px |
|---|---|---|---|---|---|---|
| Stay at the guess | 12.7 px | 12.2 px | 0.003 | 0.012 | 0.050 | 0.198 |
| Reader | 2.4 px | 4.2 px | 0.239 | 0.449 | 0.656 | 0.833 |

The tables hold the motion, and the reader learns most of it within 500 steps
(train error 4.6 px at step 500, 3.7 px at 3,000).

Four reads, each re-centred on the last one's guess (same reader, same val
points; median px / share within 1 px):

| CNN (frozen) | Start | Look 1 | Look 2 | Look 3 | Look 4 |
|---|---|---|---|---|---|
| CoTracker3 | 12.7 / 0.003 | 2.34 / 0.242 | 2.23 / 0.284 | 2.24 / 0.284 | 2.27 / 0.281 |
| TAPIP3D | 12.7 / 0.003 | 4.10 / 0.077 | 3.60 / 0.109 | 3.68 / 0.110 | 3.79 / 0.102 |

More looks stall at about 2 px: a reader on the 49 x 49 tables gets the motion
roughly, not to the pixel. TAPIP3D's CNN reads worse through tables than
CoTracker3's, so its features suit its own 3D matcher, not this one.

## Does forecast training help tracking? (standard scoring, every frame seen, 100 val clips)

| Model (step 1,500) | All points | Moving hidden |
|---|---|---|
| fyJ, physics, tracking only | 0.871 | 0.316 |
| fyM, physics, tracking and forecasting together (windows) | 0.881 | 0.303 |

No: hidden points do not improve. fyM's higher hidden numbers before were
scored with frames 12-23 blank.

## Drift on the long joint model (fyO, step 5,000; scoring only, PER_FRAME)

Still proportional to how far a point has moved, but about half of fyC's:
error / distance moved 2.6% at 16-32 px, 1.8% past 64 px (fyC 3-4%, TAPIP3D
released 0.6%). Moving points within 4 px fall from 0.997 at frame 1 to 0.70
at frame 23 (fyC 0.55); predicted motion is 99% of the true motion, no undershoot.

## Drift by refinement round (fyO, step 5,000; scoring only, PER_FRAME)

Error against distance moved, before the first round and after each of the 4.
Far points are found in round 1 (32-64 px moved: 33.6 px off at the start,
2.5 px after round 1, 1.07 after round 2) and rounds 3 and 4 change almost
nothing (0.94, 0.92). Points the second window starts near (64-128 px moved,
6.8 px off at the start) end further off (1.58) than ones that start 34 px
off, so the final error follows how far a point has moved, not how far off
its guess started. The matcher sees the point; it settles slightly off it.

## Drift by how far the point turned (fyO, step 5,000; scoring only, PER_FRAME)

At the same distance moved, error / moved by how far the point turned towards
or away from the camera since frame 0 (from the rotation of its rigid
companions; 98% of seen points have enough):

| moved | 0-5 deg | 5-15 | 15-45 | 45+ |
|---|---|---|---|---|
| 32-64 px | 0.011 | 0.013 | 0.021 | 0.042 |
| 64-128 px | 0.008 | 0.008 | 0.011 | 0.029 |
| 128+ px | 0.008 | 0.007 | 0.008 | 0.026 |

Points that move far without turning are 0.8% off (TAPIP3D released 0.6%
over all points); turning 45+ deg makes it 3-4 times worse, and most points
that move far have turned that much. Getting nearer or further changes
nothing (0.013-0.022 at any depth change).

## Matcher probe (scoring only, `scripts/probe_reader.py`)

fyO's 3D matcher frozen (and, apart, TAPIP3D's released matcher weights in the
same model); only a small reader trained (3,000 steps) to say, from the
matcher's output alone, how far the truth is from a guess pushed up to 16 px
off on each axis, then four reads from its own guesses. Val, 96k visible
moving point-frames. Median error after the fourth read, by how far the point
turned since frame 0:

| matcher | 0-5 deg | 5-15 | 15-45 | 45+ | all |
|---|---|---|---|---|---|
| fyO's | 0.54 px | 0.71 | 1.32 | 6.19 | 1.17 |
| TAPIP3D released | 0.84 px | 1.07 | 2.04 | 7.67 | 1.74 |

The matcher itself loses points that have turned: 79% within 1 px when the
surface faces the camera as in frame 0, 5% after 45+ deg. Training it inside
our model made it better, not worse, and TAPIP3D's own matcher fails the same
way, so it is the frame-0 template, not our training. The whole fyO model is
better on turned points than its matcher alone (64-128 px moved, 45+ deg:
2.9% of the distance against 6.2%), so the rest of the model makes up part
of it.
