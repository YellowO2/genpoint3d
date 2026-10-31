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
| fyP | fyO with each frame also compared against the frame before it (`--fine-prev 1`), the second comparison started as a copy of the first | turned moving points closer than fyO at the same step | val seen moving (frames 12-23 blank) 0.872 / 0.892 / 0.903 at steps 1k / 2k / 3k vs fyO 0.841 / 0.868 / 0.881; moving hidden 0.575 vs 0.524 at 3k; forecast moving 0.376 vs 0.361; 10.8 s/step vs fyO 6.8 | does a recent-frame template fix the turned points |
| fyQ | fyP with each point also given SEA-RAFT's flow since the frame before at the 5 x 5 CNN cells around its guess (`--flow-patch 2`) | turned moving points closer than fyP at the same step | ran to its 12 h limit (step ~3,800; the VPN was down at step 2,000). Same as fyP at every step: seen moving 0.872 / 0.891 / 0.902 at 1k / 2k / 3k vs fyP 0.872 / 0.892 / 0.903; moving hidden 0.577 vs 0.575 at 3k; forecast moving 0.374 vs 0.376. 11.2 s/step | the flow around a point adds nothing the model did not already have; turn test skipped |
| fyR | fyP's weights trained on (`--init`), 1,000 steps, lr 2e-4, the whole-frame match off, 256 points a clip in batches of 4 | the reference for fyS | seen moving 0.912 / 0.913 / 0.915 / 0.916, moving hidden 0.597 / 0.602 / 0.605 / 0.607 at 250-1,000 (fyS 0.916 / 0.611 at 1,000); 9.4 s/step | the whole-frame match is not missed |
| fyS | fyR with each point moved towards its soft group's rigid motion (`--rigid-fit 16`, after RAFT-3D) | turned moving points closer than fyR at the same step | 1,000 steps: seen moving 0.912 / 0.914 / 0.916 / 0.916, moving hidden 0.594 / 0.603 / 0.610 / 0.611 at 250-1,000; fyR at 250: 0.912, 0.597. The gate never opened (its weights 0.02): the groups were only trained through it | the fit unused; redesigned dense (docs/tasks/rigid-motion-2026-10-30.md) |
| fyT | fyU's step-500 weights trained on without the fit, the dense cache, 500 points a clip in batches of 2 (gdev, cut at 2 h) | the reference for fyV | at 250 / 500: seen moving 0.928 / 0.931, moving hidden 0.695 / 0.700, forecast moving 0.393 / 0.393, all 0.872 / 0.875 | |
| fyU | fyP's weights, the dense cache, with the rigid fit through a gate and trained directly: the vectors against the truth's rigid pairs (`--group-loss 1`), the fitted positions against the tracks (`--fit-loss 0.5`) | the gate opens; turned points closer | stopped at step ~500 (a clip with 501 points, fewer than the 512 asked). The gate stayed shut (0.02). Scored at step 500 with the gate held open (the fit alone) against shut: median error 1.23 vs 1.93 px past 128 px moved, 0.85 vs 1.15 at 64-128, moving hidden APD 0.592 vs 0.536, but 2.01 vs 0.55 px at 4-8 px moved, all-point APD 0.926 vs 0.956. Fit in rounds 1-2 only, the guess after (scoring only): in between (4-8 px 0.85) | the fit carries large motion and turns, and is too coarse for small motion; train with it in the first rounds only |
| fyV | fyU's step-500 weights, the fit placing the points in rounds 1-2 and the rounds after refining (`--fit-rounds 2`, no gate), 500 points a clip | turned and hidden points closer than fyT, small motion as good | at 250 / 500 / 750 / 1,000: seen moving 0.905 / 0.914 / 0.918 / 0.922, moving hidden 0.716 / 0.739 / 0.752 / 0.760, forecast moving 0.350 / 0.358 / 0.370 / 0.371, all 0.859 / 0.860 / 0.863 / 0.866; 12 s/step | hidden points +0.04 over fyT, but seen moving -0.02 and forecast -0.035: not a clean win |
| fyW | fyV's weights, the matcher turning frame 0's neighbourhood by the fit's rotation (`--rot-match 1`) | seen moving back to fyT's or better, hidden kept | | does comparing with frame 0 turned the right way recover the precision the fit costs |
| fyV2 | fyV's weights trained on as they were (gdev, cut at 2 h) | the reference for fyW | | |
| fyX | fyT's weights (no fit, groups trained); the rigid state carried round to round as input -- where the last fit put the point, how its surface turned, its group vector and confidence -- and the fit never placing the points (`--fit-rounds 0 --fit-input 1`), group and fit losses (gdev, cut at 2 h) | far and hidden points as fyV, small motion as fyT | at 250 / 500: seen moving 0.919 / 0.924, moving hidden 0.679 / 0.692, forecast moving 0.374 / 0.384, all 0.864 / 0.869; stopped at ~500 | the fit as an input is not used: behind fyT2 on every score, and without fyV's gain on hidden points |
| fyT2 | fyT's weights trained on as they were (gdev, cut at 2 h) | the reference for fyX| at 250 / 500: seen moving 0.929 / 0.931, moving hidden 0.696 / 0.703, forecast moving 0.393 / 0.392, all 0.874 / 0.875 | |
| fyY | fyT's weights + the visibility output fed back each round (`--vis-loss 1 --vis-feedback 1`) + frame t-1 against frame t over a whole-frame 24 x 24 grid, read by every point at its 16 nearest cells (`--grid-cmp 24`) | turned and hidden points closer than fyZ; the ball test's spin nearer its sliding | at 250 / 500 / 750 / 1,000: seen moving 0.922 / 0.928 / 0.931 / 0.932, moving hidden 0.673 / 0.692 / 0.697 / 0.700, forecast moving 0.375 / 0.390 / 0.393 / 0.393, all 0.855 / 0.870 / 0.873 / 0.872; 10.5-12.6 s/step | level with fyZ at 250 and 500, and at 1,000 back to fyT's own scores (0.931 / 0.700 / 0.393): the grid adds nothing, trained longer too |
| fyZ | fyT's weights + the visibility output fed back each round (gdev, cut at 2 h) | the reference for fyY | at 250 / 500: seen moving 0.923 / 0.928, moving hidden 0.678 / 0.689, forecast moving 0.370 / 0.387, all 0.845 / 0.871; 9.2-10.7 s/step |  |
| fyY2 | fyY without the per-point frame-before comparison (`--fine-prev-skip 1`: not made, zeros in its place so fyT's weights load): the whole-frame grid instead of it, not beside it (gdev, cut at 2 h) | as fyY or better | at 250 / 500 / 750: seen moving 0.872 / 0.887 / 0.893, moving hidden 0.545 / 0.584 / 0.596, forecast moving 0.342 / 0.359 / 0.368, all 0.839 / 0.857 / 0.863; 6.2-8.1 s/step | far behind fyZ: the per-point frame-before comparison is not redundant; the coarse grid cannot replace it |
| fzA | fyZ (fyT's weights + visibility fed back) + each point saying a group vector every round, trained against the truth's rigid pairs (`--group-loss 1`), fed back to the next round into its token and as an attention bias towards points with close vectors (`--group-feedback 16`); no fit (gdev, cut at 2 h) | hidden moving above fyZ; on the ball, hidden points keep turning with the visible face | at 250 / 500: seen moving 0.915 / 0.923, moving hidden 0.662 / 0.680, forecast moving 0.365 / 0.378, all 0.854 / 0.869; 9.6-11.3 s/step | a little behind fyZ, and the new inputs unused: the group bias 0.01 in every block and head (about 1 needed), the fed-back visibility as small in fyZ too. Weights that start at zero move about the learning rate a step under Adam: 0.1 in 500 steps. Re-run scaled (fzC) |
| fzC | fzA with what the fed-back visibility and groups add multiplied by 10 (`--feedback-scale 10`), so they can grow to a useful size in the run; still zero at step 0 (gdev, cut at 2 h) | the group bias grows past ~1; hidden moving above fyZ | at 250 / 500: seen moving 0.915 / 0.924, moving hidden 0.661 / 0.681, forecast moving 0.360 / 0.378, all 0.827 / 0.869; 9.1 s/step | as fzA. The bias could grow now and did not: at step 250 at most 0.08 (x10 counted), mixed signs. The model finds no use for it yet; maybe the learned groups are still poor. Next fzD with the true groups |
| fzD | fzC with the attention bias built from the truth's pairs that move as one (`--group-oracle 1`, in training and scoring) in place of the learned vectors' affinity: a diagnostic | the bias grows; hidden moving above fyZ | | do the points use a right grouping at all, apart from learning one |
| fzB | fyY2's step-750 weights (the whole-frame grid in place of the per-point frame-before comparison) trained on for 2,000 more steps (g1) | closes on fyZ / fyT's plateau (seen moving 0.931, moving hidden 0.700) | at +500 / +1,000 / +1,500: seen moving 0.896 / 0.902 / 0.906, moving hidden 0.609 / 0.627 / 0.638, forecast moving 0.375 / 0.381 / 0.383, all 0.866 / 0.866 / 0.875; 6.8 s/step; stopped at ~1,600 | closing by 0.01-0.02 hidden a 500 steps, still 0.05 behind fyZ hidden and 0.02 seen: the coarse grid does not replace the per-point comparison here. Not tested: a finer grid, or training with the grid from scratch |
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

## Decision 2026-10-29: drop the whole-frame match from the next run on

`--correlate 0`. It hands the model only where the best-looking patch is,
not what the patch shows; DINOv3's meaning reaches the points through the
cross-attention, which stays. About 15% of a step. Not tested on its own.

## Drift by turn on the frame-before matcher (fyP, step 3,000; scoring only, PER_FRAME)

Error / moved for points turned 45+ deg, fyP (3,000 steps) against fyO (5,000):
16-32 px moved 0.082 vs 0.091, 32-64 px 0.042 vs 0.042, 64-128 px 0.024 vs
0.029, 128+ px 0.020 vs 0.026. Points that barely turned unchanged (0.007-0.008).
Moving points by frame: frame 12 0.765 vs 0.739, frame 23 0.701 vs 0.679.
The frame before closes about a fifth of the gap between turned and unturned
points; turned points are still 3 times further off.

## Points carried by optical flow alone (scoring only, `scripts/flow_chain.py`)

SEA-RAFT (spring-M, frozen) flow between neighbouring frames, chained from
each query point, at the frames cache's depth; nothing looks back at frame 0.
fyO's 100 val clips and points. Reading the depth at the nearest 4 px cell
alone (true pixels) costs 0.6 px and leaves moving APD 0.83 at frame 23.

| | chained flow | fyO |
|---|---|---|
| moving APD, frame 12 / 23 | 0.411 / 0.340 | 0.739 / 0.679 |
| error / moved, 64-128 px, turned 0-5 deg | 0.023 | 0.008 |
| error / moved, 64-128 px, turned 45+ deg | 0.095 | 0.029 |

Far worse than the model everywhere, turned points included (4 times worse
turned than unturned, as fyO's 3.6). Flow chained on its own does not follow
turned points better; the all-clips flow cache is held.

Each point moved with its object instead (`--object`): every step, one rigid
move per object fitted to its visible points' flow and depth (objects and
visibility from the truth), applied to all its points. Moving APD frame 12 /
23: 0.509 / 0.409 (own-pixel chain 0.411 / 0.340, fyO 0.739 / 0.679).
Error / moved at 64-128 px: 0.017 turned 0-5 deg, 0.048 turned 45+ (chain
0.023 / 0.095, fyO 0.008 / 0.029). Half the chain's error on turned points,
still worse than the model everywhere: chained over 23 steps with nothing
looking back at frame 0, the drift dominates.

## The fit against its control, by distance moved (scoring only, PER_FRAME, dense cache, 500 points)

fyV (fit in rounds 1-2, step 1,000) against fyT (no fit, step 500), median
error: 1.44 vs 1.83 px past 128 px moved, 0.76 vs 1.00 at 64-128, turned 45+
at 64-128 px 1.4% vs 1.8% of the distance; moving hidden APD 0.608 vs 0.542,
moving seen late 0.759 vs 0.746; but 0.52 vs 0.29 px at 1-4 px moved, 0.82 vs
0.37 at 4-8. The fit carries the large motions and the turns and costs the
small ones. The turn-aware matcher (fyW) against fyV continued (fyV2) at
250 / 500: seen moving 0.915 / 0.922 vs 0.913 / 0.921, moving hidden 0.749 /
0.766 vs 0.744 / 0.758: close to neutral.

Groups of fyV against the truth (`scripts/check_groups.py`): of the points a
point's group holds, 95% or more truly move with it to 5 mm for points moving
4 px or more (80% at 1-4 px), so the small-motion cost is not bad grouping.
Scored with 6 rounds instead of 4 (fit in 2): small motion unchanged (0.82 px
at 4-8 px), far worse. A placed fit is coarser than a close guess and the
rounds after do not win it back.

## Where forecasting fails (scoring only, `scripts/forecast_split.py`, frames 12-23 forecast)

Points split by their true track over frames 11-23: at rest (moves < 1 cm),
smooth (acceleration never jumps by more than 0.02 m/frame^3), impact (it
does). Median error in metric px over the forecast frames, fyT / fyV, against
holding the last seen position (stop) and keeping the last speed:

| kind | share of point-frames | fyT | fyV | stop | last speed |
|---|---|---|---|---|---|
| at rest | 74% | 0.02 | 0.04 | 0.00 | 0.00 |
| smooth | 6% | 1.78 | 2.37 | 1.81 | 2.28 |
| impact | 20% | 11.8 | 11.2 | 19.2 | 15.5 |

Almost no free motion is left by frame 12 (Kubric's objects land early), and
the little smooth motion is slow. The forecast error is the points that hit
something in the forecast frames: the model beats both baselines there but is
still 11-12 px off. Spin and momentum would help little; contact, the unseen
shape that makes it, and more than one possible outcome are where it is lost.

## The spinning-ball test (scoring only, `scripts/ball_test.pbs`)

A textured ball over a textured floor, ray-traced with exact tracks and true
depth (`scripts/render_ball.py`), 3 textures per motion, 24 frames; queries
on the ball and floor at frame 0. Median error in metric px for points that
moved 32-64 / 64-128 px, fyT (no fit) and fyV (fit in rounds 1-2):

| motion | fyT | fyV |
|---|---|---|
| sliding, no spin | 0.39 / 0.50 | 0.67 / 0.92 |
| spin about the vertical axis, 5 deg a frame | 1.04 / 1.28 | 0.61 / 0.84 |
| the same, 10 deg a frame | 1.00 / 1.63 | 0.68 / 1.45 |
| spin about the horizontal axis, 10 deg a frame | 0.93 / 1.64 | 0.80 / 2.69 |
| rolling | 1.85 / 2.43 | 1.94 / 3.02 |
| either spin at 20 deg a frame | 49-65 px: lost | 36-56 px: lost |

Sliding without turning is tracked to 0.4% of the distance; the same ball
spinning is 3 times further off, rolling 5 times. The fit helps slow spin
about the vertical axis and costs sliding, rolling and spin about the
horizontal axis. At 20 deg a frame both lose the ball altogether.

## Whole-frame flow and a rigid fit on the spinning ball (no model, `scripts/ball_flow_fit.py`)

SEA-RAFT between neighbouring frames over the whole frame, every ball cell
seen in both (which cells are the ball's from its true centre, standing in
for learned grouping; true depth), the ball's rotation and shift fitted each
step and composed; the points carried by it from frame 0. Median error in
metric px at 32-64 / 64-128 px moved, against the models on the same clips:

| motion | flow + rigid fit | fyT | fyV |
|---|---|---|---|
| sliding | 0.32 / 0.57 | 0.39 / 0.50 | 0.67 / 0.92 |
| spin, vertical axis, 5 deg a frame | 0.50 / 0.75 | 1.04 / 1.28 | 0.61 / 0.84 |
| spin, vertical axis, 10 deg | 0.30 / 0.50 | 1.00 / 1.63 | 0.68 / 1.45 |
| spin, horizontal axis, 10 deg | 0.30 / 0.40 | 0.93 / 1.64 | 0.80 / 2.69 |
| spin, vertical axis, 20 deg | 2.44 / 2.51 | lost (50-65) | lost (40-56) |
| spin, horizontal axis, 20 deg | 1.67 / 1.45 | lost (44-58) | lost (30-46) |
| rolling | 0.63 / 1.02 | 1.85 / 2.43 | 1.94 / 3.02 |

With the whole visible surface compared frame to frame and one rigid motion
fitted to it, the spinning ball is tracked as well as the sliding one, and
the fast spin the models lose entirely is followed to 2 px. The core of the
scene-points design holds; what the learned version must supply is the
grouping (here the truth) and its own frame comparison.

## On the spinning ball, drawn position against depth (fyT, `scripts/error_split.py`)

Median error for seen points that moved more than 16 px, in metric px: the 3D
error, the gap between the drawn pixels (256 px frame), the gap in depth.

| motion | 3D | 2D | depth |
|---|---|---|---|
| sliding | 0.44 | 0.30 | 0.22 |
| spin, vertical axis, 5 deg | 1.02 | 0.74 | 0.43 |
| spin, vertical axis, 10 deg | 1.13 | 0.82 | 0.48 |
| spin, horizontal axis, 10 deg | 1.02 | 0.69 | 0.53 |
| rolling | 1.95 | 1.45 | 0.99 |
| spin, vertical axis, 20 deg | 58.4 | 26.4 | 33.6 |

Both grow with the turn, about 2.5 times sliding's; the drawn position is the
larger part. The model puts a turned point in the wrong place in the image,
not only at the wrong depth.

## The step tracker, our own comparison (step_a, `scripts/train_step.py`)

TAPIP3D's comparison (its CNN frozen) between each point in frame t-1 at its
true position and frame t around a guess, 3 refinements, trained 1,500 steps
on the dense cache from true previous positions; a group head on DINOv3, the
CNN and position. One step ends about 1 px off (median, 384 px frame).
Chained over clips it does not work: with each point's own step even still
points drift 6-7 px and moving ones are lost; with the rigid fit it blew up
(the fit's weights collapsed, now floored). One step at 1 px against
SEA-RAFT's far finer flow, and trained only from true previous positions, so
it never learns to recover from its own errors. Next: SEA-RAFT's flow for the
steps with the same learned groups, to tell the comparison from the rest.

## Does fyT send points round to the ball's back? (`scripts/ball_backside.py`)

Of the ball's points truly hidden (turned to the back) in a frame, the share
whose prediction is behind the ball's centre along the line of sight: spin
about the vertical axis 5 deg 59%, 10 deg 88%; about the horizontal axis 10
deg 78%; rolling 66%. Of seen points, 0-3%. The model mostly does carry a
turning point round to the back; the videos draw every prediction on top, so
they look as if it stays in front. Points just turned past the edge sit near
the centre's plane, which is most of the 5 deg misses.
