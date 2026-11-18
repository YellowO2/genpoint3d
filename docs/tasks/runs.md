# Run tracker

Fast setup unless noted: 24-frame Kubric clips, 128 points, 2,500 steps,
batch 8. Score: APD on visible moving points (val), higher is better.
TAPIP3D's released model scores 0.926 on the same points.

## Run names

From 5 Nov runs are named by what they are. The earlier ones, as they are referred to now:

| code | name |
|---|---|
| fyT | old_design (the reference new runs start from) |
| fzP | new_design |
| fzQ | new_design_fixed_ruler |
| fzR | new_design_reversed |

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
| fzD | fzC with the attention bias built from the truth's pairs that move as one (`--group-oracle 1`, in training and scoring) in place of the learned vectors' affinity: a diagnostic | the bias grows; hidden moving above fyZ | at 250 / 500: seen moving 0.913 / 0.923, moving hidden 0.662 / 0.681, forecast moving 0.369 / 0.375, all 0.849 / 0.867; 9.2-12 s/step | no better than fyZ (hidden 0.689 at 500), with the bias grown this time (about +-0.2-0.3 x10 counted at 250, mostly positive): right groups are used a little and do not help. The groups are not the gap; next, an explicit pull of hidden points to their seen group-mates (fzE, fzF) |
| fzB | fyY2's step-750 weights (the whole-frame grid in place of the per-point frame-before comparison) trained on for 2,000 more steps (g1) | closes on fyZ / fyT's plateau (seen moving 0.931, moving hidden 0.700) | at +500 / +1,000 / +1,500: seen moving 0.896 / 0.902 / 0.906, moving hidden 0.609 / 0.627 / 0.638, forecast moving 0.375 / 0.381 / 0.383, all 0.866 / 0.866 / 0.875; 6.8 s/step; stopped at ~1,600 | closing by 0.01-0.02 hidden a 500 steps, still 0.05 behind fyZ hidden and 0.02 seen: the coarse grid does not replace the per-point comparison here. Not tested: a finer grid, or training with the grid from scratch |
| fzE | fzD (true groups) + a second bias towards group-mates scaled by how hidden the point is and how seen the other one is in that frame (`--group-vis-bias 1`) (gdev, cut at 2 h) | hidden moving above fyZ and fzD | at 250 / 500: seen moving 0.914 / 0.924, moving hidden 0.664 / 0.681, forecast moving 0.366 / 0.376, all 0.845 / 0.869; 9.4 s/step | no better than fyZ / fzD: with right groups and a pull towards seen group-mates, moving hidden is where every variant is at step 500 (0.68). How points attend to each other is not what holds hidden points back on Kubric |
| fzF | fzE with the learned groups (no oracle): the version that runs on real video (g1, 1,000 steps) | as fzE | at 250 / 500 / 750: seen moving 0.915 / 0.923 / 0.926, moving hidden 0.665 / 0.681 / 0.686, forecast moving 0.364 / 0.379 / 0.384, all 0.836 / 0.868 / 0.872; standard scoring: all 0.913, non-visible 0.534, forecast 0.627 / moving 0.372 | no gain: as fyZ, below fyP on the standard scoring |
| fzG | fyZ (fyT's weights + visibility fed back) with each window's new frames starting at the point's speed over the shared frames instead of held at frame 15's position (`--reach continue`), in training and scoring (g1, 1,000 steps) | moving hidden and forecast above fyZ; on the ball, hidden points keep turning past frame 16 | at 250 / 500 / 750 / 1,000: seen moving 0.923 / 0.928 / 0.930 / 0.931, moving hidden 0.674 / 0.687 / 0.695 / 0.695, forecast moving 0.368 / 0.377 / 0.381 / 0.382, all 0.864 / 0.872 / 0.873 / 0.873; standard scoring: all 0.915, non-visible 0.549, forecast 0.614 / moving 0.364 | level with fyZ / fyT: the handover is not what costs hidden points on Kubric; forecasting a little worse |
| fzH | fyT's weights, 3 refinement rounds instead of 4, the visibility head trained (`--vis-loss 1`, nothing fed back) (g1, 1,000 steps) | the reference for fzI; 3 rounds from now on | at 250 / 500: seen moving 0.915 / 0.921, moving hidden 0.670 / 0.680, forecast moving 0.382 / 0.389, all 0.864 / 0.869; 7.1-7.4 s/step (4 rounds: 9-11); stopped at ~550 | level with the 4-round runs at the same steps (fyZ 0.678 / 0.689 hidden, 0.923 / 0.928 seen): 3 rounds lose nothing, about 30% faster |
| fzI | fzH with hidden points blanked from round 2 on, as forecast frames are (`--hidden-blank 1`): one path for hidden points and forecasting | moving hidden above fzH; the ball's hidden points keep its shape better | at 250 / 500: seen moving 0.916 / 0.921, moving hidden 0.665 / 0.670, forecast moving 0.381 / 0.389, all 0.863 / 0.872; visibility head right on 89% of hidden point-frames; 7.4 s/step; stopped at ~550. Spin up 10 (one clip, median px, fzH / fzI): seen facing 0.95 / 0.85, edge 1.72 / 1.54, hidden 19.8 / 22.2, returned 79 / 92 | no gain: hidden points a little worse on Kubric and the ball, seen ones a little better. Carried as forecasts, hidden points are only as good as the forecast, which the drop test shows is weak; the path stays (one mechanism), the forecasting itself is what to fix |
| fzJ | fzI (3 rounds, hidden points blanked) trained on the new Kubric clips only: MOVi's own worker with a spin at each throw and rolling balls and cylinders (`scripts/kubric_movi_tracks.py`), cut to 48 frames, 240 of 518 kept (`scripts/crop_raw.py`, at least 150 moving tracks seen in frame 0), 500 points, batch 1, forecast from frame 24 at validation (g1, 1,000 steps) | the drop clip forecast above "keeps its last speed"; on spin up 10 the hidden points nearer the truth than fzI's | | is missing physics and turning in the training data what holds forecasting and hidden points back |
| fzK | fzJ with 2 forward rounds and then a backward pass over the windows, 2 rounds each, the model told which way time runs (`--refine-iters 2 --back-iters 2`); still every point queried at frame 0 (g1, 1,000 steps) | above fzJ on hidden and forecast | not run | cancelled before it started: fzL against fzJ answers whether the design helps on the same clips, and fzK's sweeps (ffbb) were not fzL's (fbfb), so it would not have split the credit cleanly |
| fzL | fzJ's clips with points from every frame: the same clips cached with each point's query the first of frames 0, 8, 16, ... it is seen in (`--query-stride 8`), its template, pin and steps there, a window holding the points first seen by its end, and the four rounds a window as alternating sweeps, forward, back, forward, back (`--schedule fbfb`): each sweep starts from what the one before left, so later frames inform earlier ones twice and late points get their past from the first backward sweep -- the first version of the scene-points design (g1, 1,000 steps) | above fzK; late points tracked back to the start | val (its own clips, forecast from 24) at 250 / 500 / 750 / 1,000: seen moving 0.443 / 0.487 / 0.523 / 0.525, moving hidden 0.096 / 0.110 / 0.120 / 0.125, forecast moving 0.161 / 0.213 / 0.259 / 0.265; 16 s/step. Fall test (forecast from 3) moving 0.104 (fzJ 0.106). Spin up 10 median px: facing 0.97, edge 2.28, hidden 30.6, returned 60.1 (fzI 0.74 / 1.64 / 25.8 / 91.4). Standard Kubric: all 0.857, non-visible 0.417; forecast 0.585, moving 0.334 (fzI 0.913 / 0.519; 0.623 / 0.377) | no gain on falls; returned points on the ball nearer (60 px against 91), everything else worse, and Kubric scores well down: trained on the 216 new clips alone, it lost some of what Kubric taught. Points joined only every 8th frame. Next: points at any frame (fzN) and the clips mixed with Kubric |
| fzM | fzJ with every clip levelled: turned about the frame-0 camera so up is always -y (`--level 1`, each clip's up from Kubric's cameras, `scripts/add_up.py`), gravity one fixed direction instead of one each clip's camera tilt sets (g1, 1,000 steps) | the fall test above fzJ's 0.106 | val at 250 / 500: seen moving 0.504 / 0.520, moving hidden 0.184 / 0.188, forecast moving 0.272 / 0.313 (fzJ 0.525 / 0.536, 0.204 / 0.203, 0.286 / 0.315). Fall test (forecast from 3) moving 0.105 (fzJ 0.106) | no gain: with up the same axis in every clip, forecasting falls is unchanged. Not knowing which way is down is not what holds forecasting back |
| fzN | fzL with points joining at exactly the frame they are first seen (`--query-stride 1`, not rounded to every 8th frame), the matcher reading each point's template in its own frame in one pass, and each point's group vector (`--group-feedback 16 --group-loss 1 --feedback-scale 10`) carried through every window and sweep (g1, 1,000 steps) | as fzL or above, at fzL's speed or better | | the design as intended: a point joins the frame it appears |
| fzO | fzN with the attention between points encoding each point's current guess in each frame instead of where it was first seen (`--rope-current 1`), so how near another point is means near now (g1, 1,000 steps) | the drop test's object stops at the floor; hidden points on the ball follow the turn better | | do points need to see where the others are now, for collisions and turning |
| fzP | The fixed design (the fixes after the cleanup, 4 Nov): the visibility head's last call fed back instead of blanking hidden points for good; training clips that lose a quarter, half, three quarters or all of their frames over a gap or from a cutoff, seen again after; the matcher's real frame before; new points' group vectors from zero; the attention between points on each point's current guess (`--rope-current 1`, now the default); our 240 clips and the old Kubric (fyT's dense cache) drawn alike, from fyT's weights (g1, 1,000 steps) | returned points re-found on the ball; Kubric scores kept (no fzL-style loss) | standard (old Kubric, 100 clips): all 0.873, non-visible 0.472, moving non-visible 0.369; forecast from 12: all 0.601, moving 0.345, seen moving 0.881 (fyT 0.920 / 0.546 / 0.448; 0.625 / 0.385 / 0.907). Fall test moving 0.076. Drop test: still falls through the floor. Ball spin_up_10 median px: facing 1.07, edge 2.42, hidden 29.9, returned 75.9, none within 4 px (fyT 0.58 / 1.54 / 21.8 / 82.9) | below fyT everywhere, visible points too. fyT's own weights under the new code score 0.915 / 0.553 / 0.444, so the code is sound and the drop is fzP's training; it predates the audit's six fixes |
| fzQ | fzP with the audit's fixes (visibility loss on the window's points only, gaps ending before the clip does, the fed-back visibility bounded, no group bias for points without a vector, points handed over only in a whole frame, the pin removed) and one ruler for every clip (9 m a unit, inputs held within 60 m, instead of each clip's median distance over all its frames, forecast ones included) (g1, 1,000 steps) | as good as fyT on standard scoring; returned points on the ball re-found | standard: all 0.877, non-visible 0.478, moving non-visible 0.369; forecast from 12: all 0.596, moving 0.349, seen moving 0.882. Fall test moving 0.079; the drop still goes through the floor. Ball spin_up_10: facing 108 px, every class lost | the fixes changed nothing on Kubric; the ball broke because the matcher still measured offsets with each clip's own frame-0 spread while the rest used the 9 m ruler (the ball scene is 3 m) |
| fzR | fzQ with the matcher on the same 9 m ruler (offsets over a typical Kubric frame's spread, 0.35, not each clip's), and the backward sweeps run forward over the clip played backwards, the steps one running sum in the order a window is run: a step always means 'from the frame before', where before, backward rounds had to flip every output keyed off a direction input starting at zero, and the last round is backward (g1, 1,000 steps) | visible points back to fyT's level (0.92 standard); the ball working again | standard: all 0.877, non-visible 0.466, moving non-visible 0.356; forecast from 12: all 0.584, moving 0.349, seen moving 0.875. Own val (step 1,000): hidden moving 0.204 (fzQ 0.168), seen moving 0.483 (0.511). Ball spin_up_10 works again: facing 1.41 px, hidden 35.5, returned 90.7. fyT's weights on this code: 0.903 standard, seen moving 0.883 (0.915 / 0.902 before the matcher's ruler) | the backward fix helps hidden points on our clips; the matcher's ruler costs visible points about 0.02; old Kubric level with fzQ, so the main drop from 0.920 is not the scale |
| reversed_no_rope_current | fzR with the attention between points on each point's start again (`--rope-current 0`), same code (927d14b) (g1, 1,000 steps) | back towards 0.92 if the moving position encoding is what costs the new design | standard: all 0.879, non-visible 0.462, moving non-visible 0.358; forecast from 12: all 0.584, moving 0.344, seen moving 0.875 (rope-current on: 0.877 / 0.466 / 0.356; 0.584 / 0.349 / 0.875) | no difference: rope-current is not the drop |
| reversed_old_recipe | new_design_reversed's model trained the old way: the old Kubric only, whole-frame cutoffs only (`--max-gap 0 --partial-masks 0`), no rigid proposal; validated on the old Kubric's split from 12 as old_design was (g1, 1,000 steps) | back near 0.92 if the drop is the training recipe (mixed data, masks); still near 0.877 if it is a model piece (the alternating sweeps, the group bias) | standard (every frame seen): all 0.871, non-visible 0.472, moving non-visible 0.368. Own val (old Kubric split, from 12, 500 points), step 1,000: all 0.863, seen moving 0.907, forecast moving 0.365, hidden moving 0.647 (old_design's own val: 0.875, 0.931, 0.393, 0.700). Ball: facing 1.40 px, hidden 23.9, returned 85.3 | no recovery with the old recipe: the drop is a model piece, not the data or the masks |
| old_recipe_forward_rounds | reversed_old_recipe with four forward rounds a window instead of the alternating sweeps (`--schedule=`, no direction input) (g1, 1,000 steps) | back near 0.92 if the alternating sweeps are the drop; still near 0.87 if it is the group bias or the fed-back inputs | standard (every frame seen): all 0.907, non-visible 0.532, moving non-visible 0.431. Own val, step 1,000: all 0.867, seen moving 0.923, forecast moving 0.378, hidden moving 0.676 (old_design 0.875, 0.931, 0.393, 0.700). Ball: facing 1.25 px, hidden 24.7, returned 70.6 | the alternating sweeps were the main drop (0.871 -> 0.907); the rest is about the matcher's ruler (old_design's own weights score 0.903 on this code) |
| old_recipe_sweeps_positions | reversed_old_recipe (sweeps forward and back) with the model correcting each frame's position directly instead of steps summed from a held frame; only the query frame held; a new output layer from zero (g1, 1,000 steps) | about 0.907 or better: the held frame was why the sweeps hurt (reversed_old_recipe 0.871, forward rounds 0.907) | standard (every frame seen): all 0.846, non-visible 0.382, moving non-visible 0.291. Own val, step 1,000: all 0.833, seen moving 0.893, forecast moving 0.289, hidden moving 0.580. Ball: facing 5.34 px, hidden 49.2, returned 81.6 | worse, not better: not like-for-like, as its output layer starts from zero while the runs it is set against keep old_design's, and frames no window had reached started standing still; the representation alone is not the answer |
| old_recipe_unified_positions | old_recipe_sweeps_positions with nothing held but what is known: the query frame corrected only along its line of sight (on TAPVid-3D only its pixel is given), and frames no window has reached starting at the point's last speed instead of standing still (d82bf16) (g1, 1,000 steps) | above old_recipe_sweeps_positions (0.846); level with forward rounds (0.907) if a held frame and a standing start were why the sweeps hurt | standard (every frame seen): all 0.845, non-visible 0.386, moving non-visible 0.296. Own val (old Kubric split, from 12) at 500 / 1,000: all 0.777 / 0.823, seen moving 0.879 / 0.895, forecast moving 0.207 / 0.269 (stop 0.331, last speed 0.334), hidden moving 0.555 / 0.590. Fall test moving 0.074. Ball spin_up_10: facing 3.60 px, hidden 46.6, returned 86.6 | level with old_recipe_sweeps_positions (0.846): neither the free query depth nor the moving start changes anything; forecasting below simply stopping |
| old_recipe_unified_forward_rounds | old_recipe_unified_positions with four forward rounds a window instead of the sweeps (`--schedule=`, no direction input), as old_recipe_forward_rounds was for the steps (g1, 1,000 steps) | the positions design's own level without backward sweeps: what the sweeps runs are measured against, as both start their output layer from zero | standard (every frame seen): all 0.898, non-visible 0.491, moving non-visible 0.401. Own val, step 1,000: all 0.838, seen moving 0.914, forecast moving 0.326 (stop 0.331), hidden moving 0.641 (old_recipe_forward_rounds 0.867, 0.923, 0.378, 0.676). Fall test moving 0.075. Ball spin_up_10: facing 3.76 px, edge 4.73, hidden 27.5, returned 80.4. Run on gdev (g1 allows one job a user), validated at step 1,000 only (`--val-every 1000`) to fit its two hours | the positions design costs about 0.01 against the steps without sweeps (0.898 against 0.907), more on hidden points (0.491 against 0.532); with fbfb it is 0.845, so the sweeps still cost 0.05 |
| old_recipe_unified_lock | old_recipe_unified_positions with each window given two rounds before the next starts from it, forward then back (`--schedule ffbb`; fbfb hands each window on after one), the same four rounds a window (c8a0ca8) (g1, 1,000 steps) | level with old_recipe_unified_forward_rounds, above old_recipe_unified_positions: a window locked before it is handed on, so the next one starts near the truth | standard (every frame seen): all 0.875, non-visible 0.405, moving non-visible 0.309. Own val at 500 / 1,000: all 0.796 / 0.835, seen moving 0.880 / 0.897, forecast moving 0.266 / 0.294, hidden moving 0.556 / 0.594. Fall test moving 0.073. Ball spin_up_10: facing 3.44 px, edge 10.3, hidden 42.0, returned 88.9 (forward rounds: 3.76, 4.73, 27.5, 80.4) | locking wins back 0.03 of the sweeps' 0.05 (0.845 -> 0.875 against 0.898) and forecasts better than fbfb, but hidden points stay far below forward rounds (0.405 against 0.491): what is left is the backward rounds themselves, which play the clip backwards, a world old_design's weights never saw (query frame last, time and gravity reversed) |
| old_recipe_lock_video_time | old_recipe_unified_lock with each backward sweep visiting the windows from the clip's end, every window still in the video's own time, instead of running forward over the clip played backwards; no direction input (6a561be): a round is the same task whichever way the windows are visited (g1, 1,000 steps) | level with old_recipe_unified_forward_rounds (0.898, non-visible 0.491): the reversed playback was the rest of the sweeps' cost | standard (every frame seen): all 0.892, non-visible 0.488, moving non-visible 0.386. Own val at 500 / 1,000: all 0.820 / 0.842, seen moving 0.902 / 0.913, forecast moving 0.306 / 0.329, hidden moving 0.622 / 0.638. Fall test moving 0.072. Ball spin_up_10: facing 3.40 px, oblique 3.68, edge 4.13, hidden 25.9, returned 74.8, the best of these runs | yes: the same ffbb with the clip played backwards scored 0.875 / 0.405; in video time it is level with forward rounds (0.898 / 0.491, own val 0.838 / 0.914 / 0.326 / 0.641). The sweeps no longer cost tracking |
| old_recipe_lock3_video_time | old_recipe_lock_video_time with three forward rounds a window before the backward sweep's one (`--schedule fffb`, still four rounds a window); gdev, validated at step 1,000 only (6a561be) | above old_recipe_lock_video_time and near 0.898 if how far a window is locked before it is handed on is what is left | standard (every frame seen): all 0.899, non-visible 0.494, moving non-visible 0.400. Own val, step 1,000: all 0.839, seen moving 0.914, forecast moving 0.330, hidden moving 0.640. Fall test moving 0.073. Ball spin_up_10: facing 3.51 px, oblique 4.09, edge 4.42, hidden 31.8, returned 74.4 | level with ffbb and forward rounds: once the backward rounds run in video time, two rounds lock a window well enough. ffbb stays the schedule, as it gives late points two backward rounds before the start of the clip, fffb one |
| full_design_fixed_sweeps | The design as it stands with the sweeps fixed, every part on: positions corrected each round, nothing held but the query frame's depth, ffbb in video time, a late point's past starting at its earliest speed; our 240 clips (late points, turning) mixed with the old Kubric at 25% of the draws instead of 50% (`--mix 0.75 0.25`); partial masks and gaps; the rigid proposal; from old_design's weights (7235de9) (g1, 1,000 steps) | against new_design_reversed (same data and masks, the broken sweeps, no rigid proposal: 0.877 standard; ball hidden 35.5 px, returned 90.7): tracking at least as good, hidden points on the ball nearer | standard (every frame seen): all 0.888, non-visible 0.485, moving non-visible 0.380. Own val (old Kubric split, from 12) at 500 / 1,000: all 0.808 / 0.837, seen moving 0.901 / 0.916, forecast moving 0.264 / 0.271 (stop 0.331), hidden moving 0.614 / 0.641. Fall test moving 0.064 (stop 0.061, last speed 0.076). Drop test: still falls through the floor, more slowly than free fall. Ball spin_up_10: facing 4.11 px, oblique 3.85, edge 4.03, hidden 30.7, returned 83.0. 7.3-7.7 s/step | against new_design_reversed: tracking and hidden points up (0.888 / 0.485 against 0.877 / 0.466; ball hidden 30.7 against 35.5). Against old_recipe_lock_video_time (old Kubric only, no masks, no rigid proposal): level on tracking and hidden points (0.892 / 0.488), worse on the ball's hidden points (25.9) and on forecasting (0.271 against 0.329). The rigid proposal, the masks and our clips together add nothing yet; whole-frame cutoffs are now one draw in eight, so forecasting is practised less. Returned points still not found again; facing points on the ball 3.4-4.1 px in every positions run against 1.25-1.41 for the steps |
| full_design_object_groups | full_design_fixed_sweeps with the group vectors trained on the object each point is on (our new clips, `--group-labels object`) instead of on pairs whose distance stays within 2 cm; our clips rendered again with object labels, 48 frames and moving objects given half the points (`cache/kubric_obj48`, the fall test's scenes left out), at 25% of the draws; points beyond 30 m dropped (19c74a4) (g1, 1,000 steps; queued behind the render and cache jobs; 441 clips kept) | groups right above 0.81 for moving points, and the rigid proposal beating the model's own guess on hidden points (full_design_fixed_sweeps: 0.189 against 0.380) | standard (every frame seen): all 0.885, non-visible 0.464, moving non-visible 0.383 (full_design_fixed_sweeps 0.888 / 0.485 / 0.380). Own val (old Kubric, from 12) at 500 / 1,000: all 0.793 / 0.821, seen moving 0.896 / 0.911, forecast moving 0.279 / 0.304, hidden moving 0.617 / 0.639, groups right 0.93 / 0.95, found 0.97 / 0.94 (all pairs, motion labels: the old Kubric has no object labels). Fall test moving 0.058 (stop 0.061). Ball spin_up_10: facing 4.57 px, edge 4.93, hidden 33.9, returned 95.7 (30.7 / 83.0 before). Rigid check, hidden moving, guess / proposal / proposal with the true groups: old Kubric 0.383 / 0.207 / 0.461, our clips 0.128 / 0.094 / 0.142, ball 0.062 / 0.080 / 0.121 | no: the proposal is still half as good as the guess on Kubric (0.207 against 0.189 before). Only our clips, a quarter of the draws, have object labels, and the pair-by-pair group loss scores 0.95 while the few wrong mates still ruin the fit: the loss rewards the wrong thing. Its groups against the true objects (scripts/check_groups.py, 60 of the new clips, each drawn about once or never in training): of the points it puts with a moving point (over 64 px), 0.894 are on its object, and it finds 0.907 of its object's points; 32-64 px 0.876 / 0.842; 1-4 px 0.640 / 0.900; still points 0.862 / 0.986. The wrong mates of fast movers are other objects, not the floor (3% on the background) |
| full_design_window_groups | full_design_object_groups with every window's groups trained on the pairs that keep their distance within that window's own frames (4c59306), on every clip, ours too (`--group-labels motion`): one rule for the label, rigid motion in what the window sees, instead of object labels on our clips and the whole clip's motion on the old Kubric (g1, 1,000 steps; rigid check chained after) | the rigid proposal nearer the model's own guess on hidden points than full_design_object_groups' 0.207 against 0.383 (old Kubric hidden moving) | standard (every frame seen): all 0.886, non-visible 0.456, moving non-visible 0.378. Own val at 500 / 1,000: all 0.758 / 0.811, seen moving 0.891 / 0.911, forecast moving 0.267 / 0.282, hidden moving 0.611 / 0.639, groups right 0.93 / 0.95, found 0.97 / 0.96. Fall test moving 0.066. Ball spin_up_10: facing 3.81 px, edge 4.07, hidden 34.6, returned 89.9. Rigid check, hidden moving, guess / proposal / proposal with the true groups: old Kubric 0.378 / 0.202 / 0.457, our clips 0.131 / 0.100 / 0.141, ball 0.067 / 0.082 / 0.122 | no: the proposal is as far behind the guess as with object labels (0.202 against 0.207) or the whole clip's motion (0.189). The label was not what kept the groups from serving the fit: groups right 0.95 by the pair-by-pair loss, and the few wrong mates still ruin a least-squares fit |
| full_design_clip_spread_3k | full_design_window_groups with the matcher measuring offsets at each clip's own frame-0 spread again (TAPIP3D's isotropic scale, which its weights expect) and the model told that spread against a typical one (8cc0337), trained 3,000 steps instead of 1,000 (g1, from old_design's weights; rigid check chained) | standard all back towards 0.90 and moving visible points above 0.80: old_design's weights lost 0.017 to the fixed ruler (0.903 against 0.920), and the longer run is the lever the tap_short test pointed at (TAPIP3D's 0.965 is long training) | standard (every frame seen): all 0.910, moving 0.823, non-visible 0.515, moving non-visible 0.426 (full_design_window_groups 0.886 / 0.80 / 0.456 / 0.378). Own val at 500 / 1,000 / 1,500 / 2,000 / 2,500 / 3,000: all 0.781 / 0.809 / 0.815 / 0.838 / 0.843 / 0.845, seen moving 0.888 / 0.902 / 0.916 / 0.925 / 0.929 / 0.931 (old_design 0.931), hidden moving 0.608 / 0.645 / 0.668 / 0.693 / 0.706 / 0.710 (old_design 0.700), forecast moving 0.283 / 0.310 / 0.333 / 0.342 / 0.341 / 0.342 (stop 0.334; old_design 0.393). Rigid check, old Kubric hidden moving: guess 0.426, proposal 0.282, with the true groups 0.484. Ball spin_up_10 broken: facing 76.6 px, hidden 91.8, returned 124 (3.8 / 34.6 / 89.9 before) | the biggest gain yet on Kubric: visible tracking back to old_design's own val, hidden points past it, the fit nearer the guess. But the 3 m ball scene breaks, as fzQ's did: the matcher's offsets are in the clip's own scale and the model, trained only on Kubric's spreads, cannot turn a spread it never saw into real units. Needs scale variety (a whole clip scaled with its camera looks the same) or corrections made in the matcher's scale | does the matcher's own scale and three times the steps lift visible tracking |
| full_design_more_kubric_scaled | full_design_clip_spread_3k's weights trained 3,000 steps more on 80% more clips (the ~2,400 downloaded old-Kubric clips kubric_dense never cached, cache/kubric_more_dense, less 93 that repeat its validation clips), drawn 0.45 old / 0.35 new / 0.2 ours, and the correction made in the matcher's unit, the clip's own size, then turned into ours by that size (bb88847): fixed arithmetic, not learned, so the same at any scale (g1; rigid check after). A first launch without the rescale crashed on the new cache's query frames (densify kept the source's 256; fixed) | visible moving points above full_design_clip_spread_3k's (0.823 standard), and the ball working again (77 px facing there) | stopped at step 600 for full_design_clip_units_gravity. Own val at 500: all 0.790, seen moving 0.896, hidden moving 0.664, forecast moving 0.316 (its start, full_design_clip_spread_3k, ended at 0.845 / 0.931 / 0.710 / 0.342) | the rescale upset the starting weights; not run long enough to say anything about the extra data, which is kept regardless | does more data lift visible tracking, and does the rescale make tracking hold at any scene size |
| full_design_clip_units_gravity | full_design_more_kubric_scaled's step-600 weights (its ckpt.pt) on the same data and mix, 3,000 steps, with every clip in its own unit (frame-0 spread over a typical one, about 9 m on a typical Kubric clip; the matcher's unit too) and told gravity in it: its strength 9.81 / 12 fps^2 / unit (a typical clip's about 1) always, and which way is down, a learned "unknown" vector in its place where a clip does not say and on a fifth of training clips (11a4e30, 0bb735e). Two heads beside the correction: which way is down, said for the clip, trained (1 - cos) on clips not told it and fed back in place of "unknown"; and each point's velocity and acceleration per frame, trained on the true tracks' central differences and fed back each round. Validation hides down on every other clip and reports its angle off the truth. A first launch hid the strength with the direction (85c26d4), a second lacked the two heads; both stopped within minutes. The output rescale and spread input are gone; metres only at the end. kubric_dense given its up first (scripts/add_up.py: it had none, nor has the ball) (g1; rigid check after) | Kubric back to full_design_clip_spread_3k's 0.845 / 0.931 / 0.710 or above, and the ball tracked again (76.6 px facing now; 3.8 before the clip scale) | stopped at step 1,200. Own val at 500 / 1,000: all 0.776 / 0.801, seen moving 0.839 / 0.838, hidden moving 0.621 / 0.610, forecast moving 0.311 / 0.327, down off 11.3 / 9.5 deg. Motion-state head at 1,000 on the standard test's seen moving point-frames (scripts/state_check.py, bc66493): velocity off by 0.498 of its true size against 0.144 for the track's own differences, acceleration 0.786 against 0.590; tracks the same with its state fed back or "unknown" (APD 0.7438 / 0.7431, full_design_clip_spread_3k 0.823) | the head says less than subtracting the track's positions and the model ignores it, while seen moving fell 0.06 in the first 500 steps and stayed: removed. Learning to Simulate (GNS) makes acceleration the output that positions are integrated from; a side head was the wrong form (ideas.md) | does tracking hold at any scene size with Kubric kept, now the network works in the clip's own unit and is told gravity |
| clip_units_gravity_no_state_head | full_design_clip_spread_3k's best weights (our best tracker: own val seen moving 0.931), the same data and mix, 3,000 steps, clip units and gravity as full_design_clip_units_gravity with the down head kept and trained on its angle (not 1 - cos, which barely pushes below 10 deg), and no motion-state head (g1; rigid check after) | seen moving held near 0.93 and the ball tracked again; down well below the 9.5 deg the usual tilt alone may give | own val at 500: all 0.783, seen moving 0.880, hidden moving 0.652, forecast moving 0.297, down off 12.5 deg (always guessing the average down: 18.7 on the old Kubric). Ball at 500 (step500.pt), median px: spin_up_10 facing 0.57, oblique 0.73, edge 1.26, hidden 23.6, returned 65.5 (best before 3.81 / 4.07 / 34.6 / 89.9); still facing 0.18, slide facing 0.29, hidden 2.57 | the clip's own unit fixes the ball, better than any run before; seen moving 0.04 above the motion-state head's run at the same step (every restart dips at 500, the learning rate at its peak). At 1,000: all 0.791, seen moving 0.910, hidden moving 0.681, forecast moving 0.315, down off 9.4 deg (full_design_clip_spread_3k at 1,000: 0.809 / 0.902 / 0.645 / 0.310). Rigid check at 500, standard test, hidden moving: guess 0.377, fit 0.224, fit with true groups 0.440. Stopped at ~1,400 for clip_units_gravity_fit_loss_3k | was the motion-state head what cost tracking, and does the clip's own unit fix the ball |
| clip_units_gravity_fit_loss | clip_units_gravity_no_state_head with the group vectors also trained through the rigid fit: each round, the fit from the vectors it said, the guess fixed, scored against the truth on every frame of the window, hidden ones too (`--fit-loss 1`; the gradient reaches the weights through the fit's centres, not its turn); same start, schedule and data (gdev, 2 h: compared at step 500) | the fit with the model's groups nearer the guess on hidden moving points (rigid check), groups right up, tracking kept | own val at 500: all 0.801, seen moving 0.890, hidden moving 0.661, forecast moving 0.316, groups right 0.96 (without: 0.783 / 0.880 / 0.652 / 0.297, 0.95). Rigid check at 500, hidden moving: guess 0.387, fit 0.287, fit with true groups 0.441 (without: 0.377 / 0.224 / 0.440); with 10+ seen mates 0.532 / 0.430 / 0.639 (without 0.515 / 0.350 / 0.640); seen moving fit 0.619 (0.537) | training through the fit makes the groups the fit needs: the fit 0.06-0.08 better, tracking a little better; still 0.10 below the guess, and the groups, not the fit, are the gap (0.287 against 0.441 with true groups). Kept: --fit-loss 1 the default | does training the groups through the fit make the fit worth using |
| clip_units_gravity_fit_loss_3k | clip_units_gravity_fit_loss run to 3,000 steps on g1 (the same start, data and flags, --fit-loss 1 now the default): the new main reference, end-of-run tests and rigid check after | seen moving near 0.93, hidden moving above 0.71, the fit closer to the guess, the ball kept | own val at 500 / 1,000 / 1,500 / 3,000: all 0.784 / 0.791 / 0.820 / 0.847, seen moving 0.881 / 0.905 / 0.912 / 0.936, hidden moving 0.664 / 0.685 / 0.696 / 0.735, forecast moving 0.314 / 0.314 / 0.337 / 0.353 (stop 0.334, constant speed 0.332), down off 13.8 / 10.7 / 7.5 / 6.1 deg, groups right 0.96, AJ 0.955. Standard: all 0.905, moving 0.824, non-visible 0.519, moving non-visible 0.437, AJ 0.838, OA 0.968. Fall test moving forecast 0.076 (stop 0.061, constant speed 0.076). Drop test: the forecast follows the fall to frame 7 and then stops near the floor (-2.1 to -2.5 m against the truth's -2.1 to -2.6; free fall -21 m by frame 25). Ball spin_up_10: facing 0.65 px, oblique 0.76, edge 1.14, returned 40.8, hidden 23.4 Rigid check, standard test, hidden moving: guess 0.437, fit 0.372, robust fit 0.404, with true groups 0.477 / robust 0.453; with 10+ seen mates 0.594 / 0.561 / 0.652 / 0.700 / 0.715 (the robust fit 0.06 above the guess, at 500 0.574 against 0.527) | the best on our own val yet: seen moving 0.936 (full_design_clip_spread_3k 0.931, old_design 0.931), hidden moving 0.735 (0.710, 0.700); forecast still under old_design's 0.393. Down learned from the scene (6 deg, the usual tilt alone 18.7) | where the current design ends at 3,000 steps |
| clip_units_gravity_motion_check | clip_units_gravity_fit_loss with the attention between points biased by how much each pair's distance changed over the window's frames both are believed seen, as a share of the distance (`--motion-check 1`, a scale per block and head from zero); same start, schedule and data (gdev, 2 h: compared at step 500 with clip_units_gravity_fit_loss) | groups right up and the fit nearer the guess (rigid check), tracking kept | stopped after step 500. Own val: all 0.777, seen moving 0.864, hidden moving 0.650, groups right 0.95 (clip_units_gravity_fit_loss and clip_units_gravity_fit_loss_3k, the same setup twice: 0.801 / 0.784, 0.890 / 0.881, 0.661 / 0.664, 0.96 / 0.95). Rigid check, hidden moving: guess 0.386, fit 0.284, fit with true groups 0.435 (without, twice: 0.387 / 0.384, 0.287 / 0.287, 0.441 / 0.441) | no better groups and no better fit: the model already has what the cue says. Code removed | does an explicit rigidity cue make better groups |
| clip_units_gravity_fit_left | clip_units_gravity_fit_loss with the rigid proposal also telling each point how far its mates sit from where the fit puts them, over their spread (`--fit-left 1`; 0 for mates that agree, about 0.2 with one gone astray in a toy), so a wrong fit says so; rigid_in starts from zero (one input more); same start, schedule and data (gdev, 2 h: compared at step 500 with clip_units_gravity_fit_loss) | hidden moving above the fit-loss run's: the model leaning on the fit where its mates agree | stopped after step 500. Own val: all 0.799, seen moving 0.887, hidden moving 0.638 (the fit-loss setup twice: 0.661 / 0.664). Rigid check, hidden moving: guess 0.382, fit 0.294, fit with true groups 0.453 (0.384-0.387 / 0.287 / 0.441) | the model is no better at hidden points for being told; the fit barely better. Code removed | does telling the model whether its fit holds let it use the fit where it is good |
| clip_units_gravity_fit_robust | clip_units_gravity_fit_loss with the rigid fit refit 3 times, each mate counting less the further it sat from the last fit (`--fit-robust 3`), in the model's input and in the fit loss; same start, schedule and data (gdev, 2 h: compared at step 500 with clip_units_gravity_fit_loss_3k) | hidden moving above the fit-loss runs' (0.66): a fit that beats the guess where 10+ mates are seen, now given to the model | stopped after step 500. Own val: all 0.776, seen moving 0.878, hidden moving 0.660, forecast moving 0.300 (the fit-loss setup twice: 0.801 / 0.784, 0.890 / 0.881, 0.661 / 0.664, 0.316 / 0.314) | the model's own guesses no better for a better fit given as an input: it does not read the fit | does the model use a fit that is better than its guess |
| clip_units_gravity_fit_blend | clip_units_gravity_fit_robust with each point saying how far it trusts its own match in each frame (from 0.95), every round moving it the rest of the way to where its piece's robust fit puts it (where 3+ mates back the fit, never at its query frame), the fit weighting each mate by that trust, and the model told how far the piece has turned since the query frame (`--fit-blend 1`; rigid_in from zero, one input more); same start, schedule and data (gdev, 2 h: compared at step 500 with clip_units_gravity_fit_robust) | hidden moving above 0.66, turned visible points closer; a fit that beats the guess now used by construction, not left for the model to read | stopped after step 500. Own val: all 0.784, seen moving 0.872, hidden moving 0.646, forecast moving 0.321. Rigid check, hidden moving with 10+ seen mates: the blended place 0.531, the robust fit alone 0.556 (no blend: 0.527 / 0.574) | trust stayed near the own match: nothing taught it but a 5% pull on the loss. Next: trust trained on which place is nearer the truth (in long_physics) | does a learned trust, blending the own match with the object's fit, use the fit where it is better |
| long_physics | The newest design for a long run: clip_units_gravity_fit_loss_3k's best weights (step ~2,000), --fit-robust 3, --fit-blend 1 with trust trained directly (each point-frame: was the own match or the piece's place nearer the truth, `--trust-loss 1`), and --physics 1 (each point's piece carried on where its fit has too little behind it: the centre's velocity and the spin from the last two supported frames, plus the acceleration and change of spin the points say, learned through the carried places' loss, `--physics-loss 1`); 20,000 steps on g2, 3 GPUs (a step is 3 clips: 60,000 clips, 20 times a usual run), two chained 24 h jobs with --resume, rigid check after | visible moving towards TAPIP3D's precision with the clips (turned points), hidden and forecast moving above the 3,000-step runs (0.70 / 0.34) | stopped at step ~250 for long_physics_spawn: 9.6-10.2 s a step of 3 clips (2.3 times one GPU's clips an hour) | | how far the design goes trained long, and does carrying the pieces on lift hidden and forecast points |
| long_physics_spawn | long_physics with points also started where new surface comes into view (`--spawn 150`: up to 150 of a training clip's 500 points, frame by frame from tracks that come into view farther than 5% of the frame from every tracked point seen there) and validation shared across the 3 GPUs (fe1841a). A 2-GPU check of the whole setup, 40 steps: validations at 20 and 40 ran, files written; at 40 forecast moving 0.348 (stop 0.334, constant speed 0.332), down off 5.4 deg | as long_physics, and hidden points whose old mates are all hidden placed by new ones (0.138 to 0.312 when given) | stopped at step ~650. Own val at 500 (1,500 clips): all 0.762, seen moving 0.895, hidden moving 0.709, forecast moving 0.353, forecast all 0.609 (the main run at 1,500 clips: 0.820 / 0.912 / 0.696 / 0.337 / 0.705), down off 5.7 deg; validation shared over 3 GPUs 202 s (466 alone). 11.2 s a step | stopped because physics_3k, the same package on one GPU at the full learning rate, broke (below); this run's warm-up (1,000 steps) had not yet reached it | as long_physics, with the fit always having seen points to work from |
| physics_3k | clip_units_gravity_fit_loss_3k's start and data with the long run's package: --fit-robust 3 --fit-blend 1 --physics 1, trust trained directly, 3,000 steps on g1 (one clip a step): the matched test of the package | hidden and forecast moving above the main run at the same steps | at 500: all 0.560, seen frames all 0.683, seen moving 0.737, hidden moving 0.598, forecast moving 0.283, AJ 0.603 (the main run: 0.784 / 0.941 / 0.881 / 0.664 / 0.314 / 0.906) | At 1,000: all 0.744, seen frames all 0.900, seen moving 0.841, hidden moving 0.663, forecast moving 0.319, forecast all 0.611 (the main run: 0.791 / 0.948 / 0.905 / 0.685 / 0.314 / 0.672); validation 591 s (466). At 3,000: all 0.819, seen moving 0.917, hidden moving 0.726, forecast moving 0.340, forecast all 0.708, down off 6.6 deg (the main run at 3,000: 0.847 / 0.936 / 0.735 / 0.353 / 0.731) | recovered, still behind at equal training: the package as built does not help | broken even in frames with images: the blend pulls points to wrong places at the full learning rate. To find before physics goes back in; left running overnight to see whether it recovers | does the trust-blend-physics package help at equal training |
| physics_hidden_blend | physics_3k with the blend moving only points the round believes hidden (a frame without its image included), trust trained only there; seen points keep their own match (`--blend-hidden 1`); same start and data (gdev, 2 h: compared at step 500 with physics_3k and the main run) | no collapse in frames with images (seen frames all near the main run's 0.941) and hidden moving above 0.66 | at 500: all 0.624, seen frames all 0.923, seen moving 0.873, hidden moving 0.663, forecast moving 0.292, forecast frames all 0.464 (physics_3k: 0.560 / 0.683 / 0.737 / 0.598 / 0.283 / 0.449; the main run: 0.784 / 0.941 / 0.881 / 0.664 / 0.314 / 0.665) | the collapse in frames with images was the blend moving seen points (gone here); the carried physics itself wrecks forecast frames (0.464): the made-up centre's spin and velocity, the form to replace | is the collapse the blend moving seen points |
| long_spawn | clip_units_gravity_fit_loss_3k's design and final weights with spawning (`--spawn 150`), 20,000 steps on g2, 3 GPUs, three chained 24 h jobs with --resume, validation shared over the GPUs, rigid check after | seen and hidden moving above the main run's 0.936 / 0.735 with the clips (TAPIP3D's lesson for turned points) | own val at 500: all 0.848, seen moving 0.936, hidden moving 0.740, forecast moving 0.358, forecast all 0.737, down off 4.6 deg, groups right 0.97, AJ 0.954; 9.5-9.8 s a step of 3 clips, validation 160 s. At 2,000 (6,000 clips): all 0.849, seen moving 0.938, hidden moving 0.762, forecast moving 0.373, forecast all 0.736, down off 4.6 deg, groups right 0.97 | the best yet on hidden and forecast moving | how far the working design goes trained long Its continuation (lspawn2, lspawn3, the rigid check) cancelled on 13 Nov: its fit (64 mates picked once, carried from the query frame) is superseded; lspawn1 finishes |
| points_1000_fit_last_seen | long_spawn's step-5,500 weights with 1,000 points a clip, up to 500 added on new surface (`--points 1000 --spawn 500`), the whole-piece fit carried from the last seen frame, refit 3 times (`--fit-robust 3`), same data and mix, 3,000 steps on g1 | the model follows the new fit: hidden moving up on own val, the ball's hidden points nearer with test-time spawning to 1,000 | stopped at launch (14 Nov): its fit was replaced before it trained | | does the model learn to use the fit when trained the way it is tested |
| points_1000_object_pose | long_spawn's step-6,000 weights with 1,000 points a clip, up to 500 added on new surface, and the fit input and fit loss from each object's rigid shape and pose with fresh mates counting most (994b63d: --fit-robust 3, --fit-fresh 30), same data and mix, 3,000 steps on g1 | the model follows the new fit: the ball's hidden points from its 0.21 toward the fit's 0.88 | first two launches lost: the first ran out of memory at its first steps; the second hung 8 h (the robust refit's gradient was NaN where a point sits exactly in place, the weights went NaN, the grouping looped; 3cb3725). Third at 15 s a step. Own val (1,000 points) at 500: seen moving 0.947, hidden moving 0.794, forecast moving 0.391 (stop 0.332), moving 0.699, groups right 0.95, down off 2.7 deg; long_spawn at its 500 (500 points): 0.936 / 0.740 / 0.358. At 1,000: 0.954 / 0.815 / 0.395, moving 0.704, groups right 0.96, down off 2.3 deg. Stopped at step 1,080 for points_1000_centre_arrows (its step-1,000 weights carried on). Standard (step 1,000, 256 asked, filled to 1,000, every frame seen): all 0.943, moving 0.894, hidden 0.635, moving hidden 0.595, AJ 0.897, OA 0.977 (the best before, fyP in week 11's report, unfilled: 0.921 / - / 0.556 / 0.454; TAPIP3D 0.965 / - / 0.678 / -). Fall test moving forecast 0.080 (stop 0.061). Ball rigid check: hidden moving, own guess 0.331, the fit 0.816; ball hidden 9.0 px (long_spawn 8.85). Rigid check filled to 1,000 (eb1f15c, the audit's fixes), hidden moving, guess / object / true image / true both / errors dealt: standard 0.595 / 0.633 / 0.738 / 0.906 / 0.707, fall test 0.285 / 0.358 / 0.488 / 0.921 / 0.476, ball 0.313 / 0.815 / 0.833 / 0.930 / 0.827 | | does the model learn to use the object's pose when trained with it |
| points_1000_centre_arrows | points_1000_object_pose's latest weights with each point also saying, every frame, an arrow from its guess to its object's centre of mass (`--centre-loss 1`: the arrow's tip, from the guess held fixed, against the true centre, each round). Kubric's object position is the centre of mass (it falls at exactly g while spinning). Labels only on our clips: `cache/kubric_obj48_centre`, kubric_obj48 rebuilt with each point's centre from the renders' metadata; the old Kubric and the floor have none. Same data mix (0.45 / 0.35 / 0.2), 1,000 points, up to 500 added, 3,000 steps on g1. Scored by scripts/centre_check.py on our clips' held-out tenth (44 clips), with 1,000 points as at test. A first launch died on import (a syntax error in RunningLoss, 317b05c); relaunched on 8174388, with the audit's fixes (the visual mask drawn before spawning, frames not yet looked at not seen in the fit, neighbours among live points). Own val from 9de57ac on is a new series, not comparable with this run's or any earlier row's: points minus spawn asked (500), the rest filled by the spawning rule training uses (fill_frames), only the asked scored. Training's spawning, counted on 10 clips each: before 9de57ac 0-13 points added a clip (the rest of the 1,000 drawn as given), from it 13-60 added on top of the 500 asked, so 513-560 points a clip on the old Kubric and 395-500 on ours: the gap, not the 1,000, sets how many. From 1530ad6, the first frame is filled farthest-first to total - reserve with no gap (reserve: --spawn in training, the same share of --fill at test), later frames by the gap rule. Points reached, 10 clips each, before / after 1530ad6: test (points_1000_object_pose, --fill 1000) standard 670-1,000 (mean 874) / 698-1,000 (mean 884), ours 578-1,000 (mean 815) / 587-1,000 (mean 799); training unchanged at 513-560 (old Kubric) and 395-500 (ours), as its 500 asked already fill the first frame's 500. Neither side reaches 1,000: after the first frame, the gap rule finds little new surface | the arrows' centre clearly nearer than the middle of the object's seen points on objects partly seen, and tracking as points_1000_object_pose | Own val (old series, 1,000 points) at 500: seen moving 0.942, hidden moving 0.792, forecast moving 0.392 (stop 0.332), moving 0.696, groups right 0.95, down off 4.0 deg; centre loss 0.13 at step 500 (its start, points_1000_object_pose at 1,000: 0.954 / 0.815 / 0.395); stopped at step ~850 (its ckpt.pt, 21:18) for points_1000_all_fixes. scripts/centre_check.py on its ckpt.pt (step ~800), 44 held-out clips filled to 1,000, centre error cm (median), arrows / middle of the seen points / middle of all points (guesses): under half seen 31.6 / 52.9 / 30.8, half or more 20.2 / 25.1 / 23.7, none seen 94.8 / - / 93.0; moving the same (31.5 / 52.0 / 30.7, 18.1 / 23.8 / 21.3) | the arrows beat the middle of the seen points everywhere and the middle of all points where half or more is seen (20 against 24 cm), but are level with the middle of all points where under half is seen: not yet the clear win on partly seen objects | can the points learn where their whole object's centre is from the part they see |
| points_1000_all_fixes | points_1000_centre_arrows' latest weights (ckpt.pt, step ~500-800) on the newest code (8f1686e): every fix of 14 Nov in (the audit's, frames and point-frames no round looked at not seen in the fit, the parallel fit, each round handed the fit of the guess it corrects (8d996d4), one spawning rule for training and test, own val scoring only the asked points) and the carry-forward in the fit (speed, spin about the arrows' centre, gravity where falling free); --points 1000 --spawn 100 (900 asked, the first frame filled, 100 kept for new surface: about 950 points a clip on the old Kubric, ~500 on ours), same data and mix, 4,000 steps on g1. The final model for 15 Nov | the model follows the fit now it is right in every round: ball hidden moving from its 0.37 toward the fit's 0.89, standard non-visible above 0.635, forecast moving above stop | Own val at 500 (new series): seen moving 0.952, hidden moving 0.838, forecast moving 0.391 (stop 0.331), moving 0.700, groups right 0.96, down off 2.7 deg. Stopped at step ~500 for points_1000_follow_fit; its best.pt gets the standard tests on the new code as the reference | superseded: a review found two gaps in the fit it trained on
| points_1000_follow_fit | points_1000_all_fixes' weights at step 500 on 3 review fixes and one new piece (2d617cb, 854afe8, 42dda18, --follow-fit). Fixes: the fit runs over the whole clip, so a point asked late is placed by its object in earlier windows too (before, the fit was cut at each window's end and such a point got none there); an object starts once 3 of its points are given (before, one whose first frame held under 3 never got a pose); test-time filling tracks twice, not three times. New: each point says how far to go along its fit's offset, a number added to its correction, starting at 0. Same data, mix and flags as points_1000_all_fixes, 4,000 steps on g1. The final model for 15 Nov | hidden points follow the fit: ball hidden moving from 0.37 toward the fit's 0.89, a late point's far past above 0.12, standard non-visible above 0.635 | Own val at 500: seen moving 0.948, hidden moving 0.834, forecast moving 0.394 (stop 0.331), groups right 0.96, AJ 0.938 (all_fixes at 500: 0.952 / 0.838 / 0.391): level. follow.weight norm 0.02 at 600: learning slowly. Resumed from step 600 on 92ca148 (carry_fixes). At 1000: seen moving 0.960, hidden moving 0.860, forecast moving 0.397, AJ 0.955: better seen and hidden, forecast level. follow.weight norm still 0.023: the follow number is barely used. Standard test at 1000 (after_training): all 0.9375, moving 0.902, non-visible 0.651, moving non-visible 0.619, AJ 0.893 (points_1000_object_pose: 0.9435 / 0.896 / 0.639 / 0.606 / 0.897): non-visible and moving are the best so far, all a little lower. Ball hidden median 15.8 px (was 19.7), returned points 49 px. Fall test forecast moving 0.083 (stop 0.061). Drop test: follows the fall and stops at the floor (true -2.20 at frame 7, forecast -1.85, then level near -2.3). Own val at 1500: 0.965 / 0.862 / 0.398, AJ 0.959; at 2000: seen moving 0.969, hidden moving 0.873, forecast moving 0.406, AJ 0.966. Standard test at 2000: all 0.9488, moving 0.911, non-visible 0.657, moving non-visible 0.632, AJ 0.906: above points_1000_object_pose on every one, the best so far. Ball hidden 16.8 px, fall test forecast moving 0.089. Own val at 2500: 0.971 / 0.881 / 0.405, AJ 0.967; at 3000: 0.973 / 0.886 / 0.409, AJ 0.969. Standard test at 3400 (ckpt.pt; the run's own best.pt stayed at 2000): all 0.9493, moving 0.916, non-visible 0.679, moving non-visible 0.647, AJ 0.909: the best so far, non-visible level with TAPIP3D's 0.678. Ball hidden 15.9 px, returned 53 px; fall test forecast moving 0.091; ball clip 0 video (every point drawn) APD moving 0.947, all 0.860 (fyT's video of 2 Nov, 500 points: 0.851 / 0.887). Own val at 3500: 0.975 / 0.888 / 0.409, AJ 0.971; at 4000 (the run's best.pt): 0.975 / 0.889 / 0.410, AJ 0.971. Final, standard test at 4000: all 0.9507, moving 0.916, non-visible 0.685, moving non-visible 0.648, AJ 0.911: the best model so far (week 11: 0.921 / 0.556; TAPIP3D 0.965 / 0.678). Ball hidden 15.2 px (28% within 4 px), returned 51 px; fall test forecast moving 0.092; ball video 0.947 / 0.868. The weekly report's other columns (now in after_training): the standard test on 100 clips it trained on, all 0.967, non-visible 0.704 (week 11: 0.933); the standard clips forecast from 12, forecast all 0.636, forecast moving 0.400, seen moving 0.957 (week 11: 0.635 / 0.386). The ball test had never added points (no --fill; all 334 points asked at frame 0, none of the ball's seen from frame 16 to 20): filled to 1,000 as the design runs, hidden 4.67 px median, 48% within 4 px (unfilled 15.2 px, 28%), returned 28.9 px (51.3), seen under 0.4 px; clip 0's video APD moving 0.960, all 0.873 (0.947 / 0.868). after_training's ball test and video fill from now on | do the fit's fixes and the follow number make hidden points follow their object |
| review_code_or_weights | The standard tests (scripts/after_training.pbs) on the review's code (854afe8 and after) for points_1000_all_fixes' step-500 weights and for points_1000_object_pose's best.pt (step 1,000), and scripts/rigid_check.py on the first, filled to 1,000 (gdev) | the previous best reproduces on the new code: the review's changes cost nothing | standard all / moving / hidden / moving hidden / AJ: object_pose 0.9435 / 0.896 / 0.639 / 0.606 / 0.897 (before the review 0.943 / 0.894 / 0.635 / 0.595 / 0.897); all_fixes at 500: 0.931 / 0.888 / 0.635 / 0.602 / 0.880. Ball, unfilled, hidden median px: 19.7 / 21.0 (the 9.0 px in points_1000_object_pose's row was another setting). Fall test moving forecast 0.086 / 0.084. Rigid check on all_fixes at 500, hidden moving, guess / object: standard 0.602 / 0.604, fall test 0.306 / 0.381, ball 0.469 / 0.860 (object_pose: 0.595 / 0.633, 0.285 / 0.358, 0.313 / 0.815). Fall test, hidden moving point-frames by how the fit places them, share: guess / object / seen points true: no fit 0.04: 0.24 / 0.24 / 0.24; pose fitted in the frame 0.81: 0.36 / 0.46 / 0.73; pose carried on 0.16: 0.15 / 0.13 / 0.19; seen moving there, guess 0.57. Standard, by the true piece's seen points among the asked ones (old table): none 0.19: guess 0.31, object 0.27; 1-9 0.38: 0.56 / 0.56; 10+ 0.43: 0.79 / 0.80; seen moving 0.888 | the code is not it: the best weights score the same or a little better on the new code. all_fixes' 500 steps lost 0.012 on all points (seen ones; hidden level), and its model follows the fit more on the ball (0.47 from 0.31). Where an object has 10+ seen points the fit is near the seen points' own score on the standard test (0.80 against 0.89); hidden points are held down by the frames where their object shows few or none of its points (57% of them, 0.27-0.56). On our clips the seen points themselves are poor (0.57), and true seen points lift the fit from 0.46 to 0.73 | did the review's code or the weights cost the standard score, and what holds hidden points down |
| hidden_by_placement | scripts/rigid_check.py on points_1000_object_pose's best.pt (step 1,000), hidden moving point-frames split by how the fit places them: by a pose fitted in that frame, by a pose carried on (under 3 of the object's points seen there), or not at all; and the fit with the truth only at truly seen point-frames ('seen true': 'true both' also handed unplaced hidden points their truth). Filled to 1,000 and unfilled (every input's truth exists) (gdev, 854afe8) | which hidden points are low, and where the fit with the truth reaches 1 | share: guess / fit / seen true. Standard, filled: fitted pose 0.79: 0.68 / 0.72 / 0.83; carried pose 0.19: 0.38 / 0.26 / 0.27; no fit 0.02: 0.38; seen moving 0.90. Standard, unfilled: fitted 0.71: 0.68 / 0.71 / 0.87 (all true 0.94); carried 0.26: 0.37 / 0.23 / 0.28 (all true 0.49). Our clips (fall test), unfilled: fitted 0.75: 0.41 / 0.52 / 0.84 (all true 0.96); carried 0.14: 0.18 / 0.14 / 0.18 (all true 0.30); no fit 0.11: 0.18; seen moving 0.59. Ball, filled: all fitted: 0.37 / 0.86 / 0.94; unfilled 0.21 / 0.34 / 0.39 (all true 0.83) | three separate losses. (1) Frames where the object shows under 3 points (a fifth of hidden point-frames on the standard test): the carried pose is worse than the model's own guess (0.26 against 0.38) and low even with every input true: it only carried forward from the last fitted frame, though the object is often seen again after. (2) Frames with a fitted pose: the fit is limited by the seen points' places (0.72, with them true 0.83-0.87; on our clips 0.52 to 0.84, where seen moving points are only 0.59). (3) The ball: the fit is good (0.86) and the model does not follow it (0.37). True groups and true visibility change almost nothing | why the fit for hidden points is far below the seen points |
| carry_fixes | Two fixes to the carry-forward, each from a synthetic throw first (tiny CPU scripts): 243c6b3, the noise bar is the pose's own uncertainty (misfit over the root of the points backing it), not one point's misfit; 92ca148, an object unseen over a gap is carried from both sides, switching where the two paths come nearest | | synthetic, 1.5 cm tracking noise, 60 points: a throw seen 3 frames, error 12 frames on 61 cm -> 8 cm (the old bar held an object moving 5 cm a frame still); a still object holds (0.4 cm); a gap of 10 frames, free flight 2-3 cm (from before only: up to 6), with a bounce inside 1-3 cm (from before only: 125-627 cm). With exact inputs the carry is exact. An object moving 1 cm a frame is still held (two frames cannot tell it from noise). Real clips: rigid_check queued | both in points_1000_follow_fit from step 600 (resumed on 92ca148). Real clips, rigid_check on points_1000_object_pose's best.pt, filled (92ca148): standard, carried-pose row, fit 0.257 -> 0.277 (true both 0.30 -> 0.36; the model's guess 0.377); fall test 0.135 -> 0.149; fitted rows unchanged (0.71, 0.53); ball 0.864 -> 0.884 | right on made-up clips, a small gain on real ones: the carried pose is still below the model's own guess where the object is unseen. Both in points_1000_follow_fit from step 600 (resumed on 92ca148). scripts/check_fit.py keeps the made-up clips as tests
| forecast_free_or_contact | scripts/rigid_check.py --cutoff on our clips' held-out tenth (44 clips of kubric_obj48_centre, 500 asked, filled to 1,000), points_1000_object_pose's best.pt, forecast from 3 and from 12; the forecast frames of moving points split by whether their object is still in free flight (its true centre within 5 cm of a thrown body's path from the last two seen frames) or has met something (scripts/forecast_check.pbs, gdev) | where the forecast is lost: in free flight (the carry-forward wrong) or after contact (no contacts yet) | share of forecast moving point-frames, guess / fit / true both. From 3: free flight 1-4 frames on 0.04: 0.32 / 0.40 / 0.53, 5-12 0.01: 0.09 / 0.21 / 0.29; in contact 1-4 0.05: 0.48 / 0.44 / 0.48, 5-12 0.17: 0.18 / 0.15 / 0.16, 13+ 0.73: 0.040 / 0.037 / 0.051. From 12: free flight 1-4 0.02: 0.34 / 0.39 / 0.65; in contact 1-4 0.10: 0.35 / 0.34 / 0.49, 5-12 0.22: 0.14 / 0.12 / 0.17, 13+ 0.67: 0.06 / 0.03 / 0.06. All forecast moving, from 12: 0.109 / 0.088 / 0.132 | contact is the forecast's problem on our clips: 95% of forecast moving point-frames are of objects that have already met something (the floor, another object), and there nothing is above 0.2 after 4 frames, with true inputs either. Free flight is 5% of them; there the fit beats the model's guess (0.40 against 0.32) but is not exact even with true points (0.53-0.65): the helpers' guesses and the arrows' centre are still the model's (a made-up throw: a centre 20 cm off costs 4-29 cm 4 frames on, by the spin); the unfilled check with every input true is queued | is the forecast lost in free flight or after contact |
| forecast_all_true | forecast_free_or_contact again, unfilled (500 asked, so every input has its truth), points_1000_follow_fit's step-1000 weights, after fixing a leak in the check (the true-visibility columns were handed the truly seen points inside forecast frames, so the earlier 'all true' 0.99 was no test); new column 'true centre': all true and the arrows' tips at the object's true centre of mass | with every input true, is the carry-forward exact in free flight | free flight, forecast from 3, frames 1-4 / 5-12: guess 0.37 / 0.06, fit 0.45 / 0.25, all true 0.58 / 0.27, true centre 0.72 / 0.42; from 12, frames 1-4: 0.34, 0.37, 0.73, 0.77. In contact 1-4: guess 0.49, true centre 0.50; 13+: all under 0.06. Hidden (not forecast) moving, all true: fitted pose 0.995, carried pose 0.48-0.68 | not exact even with every input true (0.72-0.77 in the first 4 frames), so part of the loss is in the carry-forward's own rules; the true centre adds 0.04-0.15 over the arrows'. The same frames from the truth at the last seen frame (forecast from 3, frames 1-4 / 5-12): the thrown path with the true turn 0.99 / 0.96, the thrown path without turning 0.31 / 0.16: the path is right, the turn is what is lost (the fit's carried turn reaches 0.72 / 0.42). In contact 1-4: true centre without turning 0.57, above everything else there. The thrown path with the turn carried on from the last two true frames: 0.74 / 0.48 (from 12: 0.83 / 0.53), level with the fit on true inputs (0.72 / 0.42; 0.77 / 0.53): the fit does its rule right, and the rule (an object keeps its last turn) is the limit, since real objects tumble (a made-up body with unequal inertia under Euler's equations: 13 cm off at +4 frames at 5 rad/s; a ball: exact). The turn's change over three frames carried too: 0.80 / 0.50 from 3, 0.76 / 0.30 from 12: no better. No rule added to the fit; the turn of a tumbling body is left for the network to learn (ideas.md) |
| handed_fit | scripts/rigid_check.py (its round-by-round table; first run as its own script, since folded in), points_1000_follow_fit's step-2000 weights, filled to 1,000: for each round of each window, on moving point-frames, the error of the guess the round starts from, of the fit as handed to it (guess + offset), how often the fit is the nearer, and the final answer; the ball (spin_up_10) and 40 standard-test clips | is the fit the model is handed wrong in some rounds, so it learned not to use it | cm, median, hidden moving, start guess / fit handed / fit nearer, by round. Ball: 21.4 / 9.2 / 0.93; 22.5 / 8.5 / 0.97 (44% have a fit); 15.8 / 3.3; 11.8 / 2.3; 9.3 / 2.0 / 0.85; 3.4 / 1.8; 2.4 / 0.8 / 0.95; final 6.0 (frames 8-23) and 1.2 (0-15). Standard clips: 38.7 / 38.4 / 0.59; 34.9 / 38.7 / 0.63 (40% have a fit); 16.2 / 13.9; 11.5 / 11.0; 9.3 / 9.3 / 0.51; 5.9 / 5.5; 5.2 / 4.9 / 0.59; final 8.0 and 4.5 | on the ball the fit handed is far nearer than the guess in every round (2 cm against 9 at the last forward round) and the model ends at 6 cm: it does not follow a good fit. On Kubric-like clips the fit handed is no nearer than the guess (a coin flip, 51-63%), so training gives no reason to follow it. The input is not wrong on the ball; it is uninformative where the model trains |
| groups_no_fit_loss, groups_fit_loss_twin | Twin runs from points_1000_follow_fit's best.pt (step 4000), 400 steps, same data, mix and flags; the first with `--fit-loss 0` (the groups trained only on which points are on the same piece), the twin with the fit loss kept (g1 and gdev) | a group means one thing, the points of one rigid piece; the fit loss also teaches 'points whose guesses fit together' (a true member with a wrong position is pushed out), and handling such a member is the fit's job (its robust refits). Does dropping the loss cost anything now | own val at 400, without / with the fit loss: seen moving 0.971 / 0.971, hidden moving 0.877 / 0.877, forecast moving 0.404 / 0.406, AJ 0.966 / 0.966, groups right 0.966 / 0.966, found 0.937 / 0.937. Standard test: all 0.945 / 0.946, non-visible 0.682 / 0.686, moving non-visible 0.638 / 0.639, AJ 0.904 / 0.905. Without: train clips 0.965, forecast from 12 moving 0.399, fall test 0.090, ball (filled) hidden 4.06 px, 50% within 4 px (the twin was cut by its 2 h limit before these). Both a little under their start (0.951 / 0.685): a fresh warm-up | level: the fit loss on the groups is removed from the code (a group means the points of one rigid piece; a member with a wrong place is the fit's to handle) |
| ball_exact | The spinning ball (spin_up_10 clip 0, filled to 1,000, points_1000_follow_fit's best.pt) should be near exact: one rigid object that keeps getting seen points. The fit taken apart offline from the clip's saved inputs (rigid_check --dump, object_places on a laptop), one input swapped at a time | which input keeps the ball from exact | error of the fit on the ball's own points, cm, median, by frame: frames 1-19 hidden 0.1-1.6 (the model's own guess 0.2-3.7: it trails the fit from frame 16, when none of the asked points is seen); frames 20-23 hidden 4.4-6.0, returned points 8. Frames 20-23 with: true visibility 2.9-5.9; true places where truly seen 4.4-6.1; both 0.4-2.7; freshness off 5.3-6.0; refits off 6.0-6.5; only the added points counted 2.6-4.0; added points not counted 5.8-5.9. At frames 20-23, 16-25 of the asked points are believed seen though hidden (their guess 8-17 cm off) and the 7-37 truly back in view are 8 cm off. No point is added after frame 12: of the 666 added, 567 go to frame 0 (mostly floor) and the 100 kept for new surface are used up by frame 12 | the fit is right while its inputs are (1 cm up to frame 19). It breaks when asked points come back: they are not found again (8 cm off) and some still hidden are believed seen, and both back the pose as fresh sightings. Behind that: the model's guess trails the fit while the points are hidden, so they return far from where they should be; and new surface gets no points after frame 12. With half the filled points kept for new surface (--spawn-share 0.5; points then arrive through frame 23, 880 in all): the fit on clip 0 stays within 0.9 cm to the last frame, returned points 1.0 cm; the model's own guess 1.6-2.7 cm hidden, 3.3-4.4 returned; over the 3 ball clips hidden moving guess 0.552 / fit 0.857 (with the run's share of 0.1: 0.506 / 0.730). The what-if of following the fit exactly (rigid_check --follow: a point-frame believed hidden with a fit takes the fit's place each round), guess hidden moving / seen moving, with / without: ball 0.707 / 0.964 against 0.506 / 0.963 (frames 8-23 hidden end at 1.9 cm, not 7.9); standard clips 0.557 / 0.905 against 0.648 / 0.916; fall test 0.322 / 0.594 against 0.369 / 0.620 | two causes on the ball. The reserve for new surface (100 of 1,000) runs out at frame 12, and with points arriving to the end the fit is exact: the rule, not the fit, deviates from the design. And the model does not follow a good fit. But following the fit by rule is no fix: on Kubric clips it costs 0.09 on hidden moving points, since there the fit is worse than the model's guess. The larger reserve (--spawn-share 0.5) on Kubric, guess hidden moving / seen moving against the run's 0.1: standard clips 0.644 / 0.912 against 0.648 / 0.916, fall test 0.354 / 0.613 against 0.369 / 0.620: level to a little lower (little new surface there, fewer points in frame 0). On Kubric the fitted poses' fit is level with the guess (0.72 both) and the carried poses (18-20% of hidden moving point-frames) are where both are low (guess 0.37-0.40, fit 0.27-0.29). The fill has two rules: frame 0 is filled farthest-first to 900 however dense, later frames by the gap rule from a reserve of 100; one gap rule for every frame would leave room for new surface with no reserve number, but training draws 900 asked points at frame 0, so it needs the user's say and a retrain |
| kubric_fit_by_seen_points | The fit taken apart offline on 10 standard clips (rigid_check --dump, points_1000_follow_fit's best.pt, filled to 1,000): hidden moving point-frames split by how many of their true piece's asked points are truly seen there; errors as a share of the scene's spread; and the seen points' error split into the part shared by the piece (the best rigid motion from their true to their predicted places) and the rest | why the fit is no better than the model's guess on Kubric when on the ball it is exact | seen mates, share of hidden moving point-frames, median error guess / fit (% of the scene's spread), how often the fit is the nearer: none 28%, 13.6 / 60.8, 0.14 (90th percentile 39 / 267: the carried pose flies off); 1-2 16%, 3.7 / 6.6, 0.33; 3-5 18%, 4.2 / 8.1, 0.42; 6-15 16%, 1.8 / 1.9, 0.51; 16+ 22%, 0.62 / 0.48, 0.65. The pose's turn is off by 10 deg with 3-5 seen points, 3.3 with 6-15, 1.4 with 16-40, 0.3 with 41+, in no steady direction (not a lag: its part along the true turn is 0.00 of it, negative half the time) | the fit is as good as its seen points are many: better than the guess only with 16 or more (the ball's case), worse with 5 or fewer, which is 62% of Kubric's hidden moving point-frames, and far worse where the object is unseen and carried. The seen points' shared error is the noise of a pose fitted to a few points, swung out along the object, not a drift. The model is told how much backs a fit as s/(s+1), which barely tells 6 from 44. For the user: tell the model the pose's own uncertainty instead (the fit already computes it); whether the carry-forward should be handed at all where it is worse than the guess |
| old_kubric_object_labels | Each old Kubric point's object and its centre of mass, from the clip's MOVi-F twin (TAPIP3D's Kubric is MOVi-F's own videos): Google's shards give which object each pixel shows and each object's pose; a point's object is the one in whose frame it stays still, its pixel breaking ties between things at rest (scripts/movi_objects.py, label_old_kubric.py, movi_objects.pbs). Written into the raw clips and the caches kubric_dense, kubric_more_dense and the standard test's, features untouched | every old clip has a twin and every point an object | trial on 4 shards (17 Nov): 22 videos gave 40 old clips their objects; 99.96% of 81,920 points stay within 1 mm in their object's frame; in 6 cached clips each point's distance to its centre holds to 0.006 cm. Full run (17 Nov): 5,893 clips labelled, 0.021% of points still in no object's frame, none rejected; 5 clips have no points file; 590 training clips of kubric_more_dense are second copies of a video | | one definition of a rigid group (same_group) on every clip, and the arrows trained on the old Kubric too |
| kubric_own_points | Each old Kubric clip's own points from its MOVi-F twin instead of TAPIP3D's 2,048: a track at every 4th pixel of every frame, kept by the adding rule at half its spacing (scripts/movi_points.py), then a copy of each cache with them, features reused (scripts/own_points_cache.py, own_points.pbs; cache/kubric_dense_own, kubric_more_dense_own). Why: TAPIP3D's points leave 11.3% of the seen surface (median of 40 clips, worst 34.3%) with no point within the rule's spacing at the 5% gap, 39.8% at 2.5% | trial of 35 clips: about 1,500 points a clip (803 to 2,008); the clip's cameras put each point on MOVi-F's pixel (0.000 px); seen surface left empty 0.0% at the 5% gap, 0.2% at 2.5%. MOVi-F's object position is the centre of gravity: in free fall its second difference is gravity's 6.8 cm to 0.13 cm (median, 592 objects). Full run (18 Nov): kubric_dense_own 3,493 clips, kubric_more_dense_own 2,399, median 1,472 and 1,484 points a clip (least 132, most 3,044), cameras within 0.001 px | | training with the adding rule free to put a point anywhere |
| adding_rule_counts | What the adding rule gives on 60 clips of kubric_dense_own with no model, picked as training picks (256 asked, scripts/count_points.py), and the same as a point cloud to look at (scripts/points_view.py, outputs/points_view.html) | gap 5%: 440 points a clip, 216 in the first frame, an object's points median 7, 31% of objects under 4, surface left empty 1.1%. Gap 3.5%: 655, 315, median 11, 23% under 4, 3.9% empty. Gap 2.5%: 1,369 (limit 4,000), 790, median 21, 8% under 4, 1.0% empty; held to 1,000 points, 90% of clips reach the limit, 15% of objects under 4, 9.7% empty. A training step on an A100 (scripts/profile_step.py, points_1000_follow_fit's best.pt, the two own-points caches): gap 5% limit 1,000 7.0 s and 12.4 GB; gap 2.5% limit 1,000 12.5 s; limit 1,500 17.2 s; limit 2,000 18.1 s and 16.1 GB | | the gap: 5% leaves a third of the objects too few points to hold a shape; 2.5% needs about 1,400 points a clip and its pool rebuilt at half that spacing |
| engine_against_kubric | Our engine's step with nothing touching (tools/engine.py) against Kubric's own physics, PyBullet: the pose a frame on from the two true poses before, on MOVi-F's true object poses, 300 clips, 77,704 object-frames (scripts/check_engine_kubric.py) | flying (our step within 0.5 cm): 8.2% of object-frames, their centre 0.13 cm off (median). At rest: 42.0%. Pushed or rolling: 49.8%, our step 6.8 cm off there, one frame of gravity | | the step agrees with the real engine where nothing touches; 92% of object-frames touch something: what the model says of the touches carries most of the physics |
| touch_against_kubric | Whether the engine's touch can say what Kubric's physics does on the floor: for 400 object-frames that are not flying (of 71,330 in 300 clips), the best touch with the floor searched for (where under the object, how wide its mass is spread, and how far the floor stops it; no push), then the pose a frame on against the true one (scripts/check_touch_kubric.py). First with one number for how far the two stick, then (e25cf21) with two: into the floor (hold) and along it (grip) | at rest (194): 6.81 cm off with no touch, 0.07 cm and 0.04 deg with the best one, all within 0.5 cm and 1 deg, either way. Moving (206): 6.85 cm and 1.38 deg with none; one number 0.60 cm, 0.95 deg, 42% within; hold and grip apart 0.26 cm, 0.59 deg, 55% within | | a resting object is said exactly. A moving one better with hold and grip apart, as an engine's contact and friction, but 45% are still not within 0.5 cm and 1 deg: other objects' touches, bounces and the object's true shape were not in the search |
| world_smoke | A smoke test of the stored world tracked a frame at a time (--world 1 --rounds 3, 1e11d4b and the matcher fix): points_1000_follow_fit's weights (489 of 504 carried; the new parts from nothing), the own-points caches, --points 2000 --spawn-gap 0.025, 100 steps on gdev | 18.8 s a step (the windows: 18.1 s at the same points), no crash; train loss 1.10. Own val at 100 (349 clips, forecast from frame 12, 4.7 s a clip): observed frames all 0.950, moving 0.779; forecast all 0.472 (held still 1.000), moving 0.214 (held still 0.177, constant speed 0.160); groups right 1.00 found 0.89; AJ 0.914; down off 21.3 deg | | it runs and tracks what it sees. The forecast is broken as expected of parts that start from nothing: every body starts half free, so the still world falls; 100 steps with the rate falling to 0 cannot teach that |
| world_step_time | A --world 1 --rounds 3 training step timed at --points 2000 --spawn-gap 0.025 on the own-points caches (scripts/profile_step.py, 15 steps after 5), every point reading every other (--patch 0) and only the 64 near it (--patch 64). Before it, where the time went (cProfile, 2 steps): the matcher rebuilt every point's template once per frame a point was ever given in, 72 calls a step: 127 s a step, 34 s after building each once (1b271ea), both with CUDA blocking on | --patch 0: 21.3 s a step, 7.0 GB. --patch 64: 24.3 s, 7.0 GB. The windows at the same points: 18.1 s, 16.1 GB | | as fast as the windows at under half the memory, before any speed-up. Near-only reading is slower here: the reading between points is not where the time goes; the weight update through the matcher is |
| parallel_fit_check | The object fit's poses for every frame at once instead of frame by frame, on points_1000_object_pose's best.pt (step 1,000), the ball (spin_up_10) filled to 1,000 points, scripts/rigid_check.py's "object" column (gdev). Tried in order: the shape from the object's first points only, then a refit with every point placed through it; the same repeated until no point joins; the same with every joined point placed again each pass; and with a point backing poses only after its query frame (49283e9) | the frame-by-frame walk's 0.815 hidden moving and 0.969 seen moving within noise, faster | walk 0.815 / 0.969, 108 ms a fit. First points then one refit 0.553 / 0.959 (10 ms); repeated until none joins 0.553; placed again each pass 0.590 (64 ms); a point backing poses only after its query frame 0.827 / 0.970 (65 ms). Committed code, with the model also tracking on it: 0.838 / 0.970, hidden with no seen mates 0.795 (walk 0.775) | matches and a little better, 40% faster. The walk's order was doing one job: the pose that gives a point its place never held that point. A pose that holds the point it places only confirms the place it gave, so the shape drifts with its own errors (what bundle adjustment did, ideas.md) | can the fit be parallel like the tracker |
| backward_sweep_check | points_1000_object_pose's best.pt on our clips' held-out tenth (44 clips of kubric_obj48, 500 points, points asked at the frame they are first seen, 167 of 500 after frame 0), every frame seen, scripts/evaluate.py with the moving points' frames split at their query frame (c45e13e). The own val clips (old Kubric) were tried first: every point there is asked at frame 0, so nothing is before a query | the frames before a point's query frame, which only the backward sweep reaches, near the frames after | moving: seen after 0.595 (none seen before: a point is asked where it is first seen), hidden after 0.303, hidden before 0.381; before by distance: 1-4 frames 0.471, 5-12 0.233, 13+ 0.116. All 0.785, moving 0.610 | the past fades with distance as a forecast does (0.47 near, 0.12 far); hidden before is no worse than hidden after. Own validation never scores a late point's past: every old-Kubric point is asked at frame 0 | how good the backward sweep is, and does a late point reach frame 0 |
| carry_forward_check | The object fit carries a pose on where too few of its points are seen (not yet tracked, hidden, forecast) instead of holding it: its centre of mass (from the points' arrows; without them the middle of its shape) keeps its speed, gains gravity where its fitted frames fall free, and it keeps its spin about the centre; a move the two poses cannot tell from their own noise is no move (b13552a, 0d7ed4a, ba2a2ce). Synthetic throw with spin, 5 frames of evidence: placed exactly to frame 15 (0.00 cm), holding 23-608 cm; shape middle instead of the centre 0.7-39 cm; at rest and backwards exact. Measured with scripts/rigid_check.py --cutoff 3 on the fall test (24 clips, filled to 1,000), points_1000_centre_arrows' step-500 weights, carry-forward against a copy holding the pose, both on ba2a2ce (with 8d996d4: earlier runs here took the stale round input, alike for both) | the fit's forecast above holding on moving points, level on still ones | object, forecast moving: carry 0.062 / hold 0.050 (frames 1-4 0.326 / 0.183, 5-12 0.096 / 0.089, 13+ 0.022 / 0.025); still 0.919 / 0.927; all 0.630 / 0.634. With the seen points exactly true (true both): moving 0.415 / 0.405, frames 1-4 0.650 / 0.485. The model's own guess, moving: 0.061 (stop 0.061 on this test). Before the noise bar, still points 0.88 against 0.95: a still object drifted on at its noise's speed | the carry-forward nearly doubles the fit's forecast in the first 4 frames and costs nothing on still points; beyond 12 frames it is no better than holding, as objects meet the floor (contacts, next). With true seen points it reaches 0.65 in the first 4 frames: the rest is tracking noise and speed from only 3 frames | does carrying an unseen object on (speed, spin about its centre, gravity) forecast better than holding its pose |
| stepper_check | A GNS-style stepper (`model/Stepper.py`, ae3f7de) trained 20,000 steps on the training clips' true 3D tracks alone (1,000 points, no images), rolled out frame by frame on the own validation clips from 12, from the true past; plain and with each true piece kept rigid each step (shape matching); pieces told by the truth (scripts/stepper_check.py, gdev) | forecast moving clearly above stop 0.334 / constant speed 0.332 (our tracker 0.353, from its own past) | a first launch gave no pieces (kubric_dense and kubric_more_dense have no objects; now from motion): 0.269 at 2,000. With pieces, plain / kept rigid: 2,000 0.249 / 0.252, 4,000 0.328 / 0.332, 6,000 0.359 / 0.374, 8,000 0.357 / 0.364, 10,000 0.373 / 0.380 (stopped by the 2 h limit) | above stop by 0.046 from the true past and still rising slowly; our tracker is 0.019 above it from its own past. A real but modest gain, not yet the jump physics should give | does stepping one frame at a time with gravity and neighbours forecast better than our windows |
| spawn_test | long_spawn's best.pt on the standard test (100 clips, 256 points), without and with up to 150 points added at test time where no point covers a seen frame (883c423, evaluate --spawn, g1) | hidden moving up, seen moving not down | step 3,500, without / with: all 0.924 / 0.922, seen moving 0.853 / 0.856, hidden 0.570 / 0.568, hidden moving 0.496 / 0.502, AJ 0.865 / 0.864 | void: the spawning then spent its whole budget on frame 0's gaps (spawn_count found it on the ball), none on new surface; fixed, to be rerun | does adding points at test time help hidden points, as new_points_check suggests |
| ball_spawn | the spinning ball (spin_up_10) by how each point faces the camera, last night's clip_units_gravity_fit_loss_3k and long_spawn (step 3,500), without and with up to 150 points added at test time (ball_view_angle --spawn, g1) | hidden and returned points nearer | median px (within 4 px), long_spawn without / with: facing 0.48 / 0.48 (0.995), edge 1.10 / 1.06 (0.968 / 0.978), returned 29.8 / 26.3 (0.000), hidden 21.7 / 20.5 (0.146 / 0.154); last night's: edge 1.14, returned 40.9, hidden 23.3 (0.139) | seen points on the turning ball are now accurate to about 1 px; hidden ones are still about 20 px off, and points that come back are never found again: the gap is not the seen points' drift. The added points' part is void: all 150 went to frame 0's background, none to the ball (spawn_count); fixed, rerunning | is the ball's trouble still the seen points, and do new points fix it |
| ball_rigid | rigid_check on the spinning ball (spin_up_10, 3 clips, 1,000 points), long_spawn step 3,500, no added points (g1) | which link breaks: the model's guess, its own fit, the fit from the true pieces | hidden moving APD: guess 0.171, model's fit 0.200 (robust 0.185), true pieces 0.266 (robust 0.320). By seen points of the true piece: none for 32.5% of hidden point-frames (all fits ~0.02), 10+ for 55%: guess 0.287, model's fit 0.336, true pieces robust 0.560; seen moving 0.90-0.94 | every link loses: a third of hidden point-frames have no seen mate at all (the tracked side has turned away); even the true-piece fit is 0.56 against 0.93 seen; the model's fit is below that; its guess below its own fit | where hidden points on a turning object go wrong |
| ball_spawn_fixed | ball_spawn's long_spawn rows again with test-time spawning as training's (from frame 1, only new surface; spawn_count) | hidden nearer | 117-122 points added a clip, about 85% on the ball, a few each frame and most as its old side turns away (frames 13-19). Median px without / with: facing 0.48 / 0.48, edge 1.10 / 1.07, returned 29.8 / 20.0 (within 4 px 0.000 / 0.016), hidden 21.7 / 16.7 (0.146 / 0.155) | the new points land where they should and help, but hidden points stay ~17 px off: the fit carries each point from its query frame, where the new mates were never seen (their place there is a guess) |
| ball_chained | rigid_check with each point carried frame by frame (chained_places, 0f0a0b4-era commit) beside the one fit from the query frame; long_spawn step 3,500; the ball with 150 added, without, and the standard test | chained far better on the ball's hidden points | hidden moving APD, ball with added: guess 0.198, fit 0.206, robust 0.194, chained 0.218; ball without, true pieces: robust 0.320, chained 0.314 (10+ seen mates 0.560 / 0.522); standard: guess 0.496, true robust 0.477, true chained 0.482 | chaining is not it: even with the true pieces and 10+ seen mates the fit places the ball's hidden points at 0.52-0.56 while seen ones are 0.90; on the standard test the model's own guess beats every fit. Chained fit removed |
| fit_ladder | rigid_check with the true pieces, then the true visibility choosing the mates (true seen), then the true places of the mates too (all true); long_spawn step 3,500, the ball and the standard test | which input costs the fit | hidden moving with 10+ seen mates, true robust / true seen / all true: ball 0.560 / 0.520 / 0.887, standard 0.758 / 0.760 / 0.989 | the fit is right and which mates count is not the problem: the mates' predicted places are. Points each about 1 px off do not move as one rigid set, and the fit carries that to the hidden points. The ball's 0.887 with every input true is likely the 64-mate cap (picked once for the clip, not by who is seen); --mates 1000 to test |
| fit_ladder_all_mates | fit_ladder with every point a mate (--mates 1000), long_spawn step 4,000 | the cap is why all-true stopped at 0.887 on the ball | ball hidden moving: guess 0.188, model's fit 0.313 (robust 0.346), true robust 0.344, all true 0.976 (10+ seen mates: guess 0.310, model robust 0.591, all true 0.996); standard: as with 64 (guess 0.507, all true 0.850) | the 64-mate cap, picked once for the clip, halves the fit on an object with many points: uncapped, the model's own fit nearly doubles on the ball's hidden points (0.18 to 0.35); the model's output (0.19) does not follow it; what is left is the mates' predicted places (0.59 against 0.996 with 10+ seen) |
| fit_whole_piece | rigid_check with the fit over every point the groups put on the piece (likeness over 0.5), seen in the frame (e096ceb), long_spawn step 4,000 as trained (it was fed the old fit), the dev copy | the fit right with true inputs, better with the model's | ball hidden moving: guess 0.205 (old fit fed in: 0.188), model's fit 0.278 (old 0.217), robust 0.293, true groups 0.279, all true 0.976 (10+ seen: guess 0.341, model robust 0.497, all true 0.996); standard hidden moving: guess 0.528 (0.507), model's fit 0.515 (0.442), all true 0.850 | the fit is right; the model's groups are now as good as the true ones for it; even untrained for it, the model's own hidden points improve. Left: the model does not follow its fit (0.205 against 0.293 on the ball), and its seen points as a set (0.50 against 0.996 with 10+ seen) |
| fit_whole_piece_spawn | fit_whole_piece's setup with points added at test time (rigid_check --spawn 300 / 600, one budget from frame 0), the ball; and, without spawning, the seen mates' accuracy and the fit's turn for hidden points with 10+ seen mates | the design we are going for, measured | ball hidden moving, spawn 300: guess 0.269, model's fit 0.367, robust 0.392 (10+ seen: 0.412 / 0.583 / 0.621); spawn 600: guess 0.222, robust 0.267 (seen moving, plain fit, falls to 0.754). Without spawning: those mates' APD 0.903 (all seen 0.932), the turn since the query frame 120.0 deg true, 123.0 from predicted places, 3.6 apart (standard: 52.8 / 52.1, 1.9 apart) | with 300 added, the best yet (guess 0.19 at the start of the day, the fit now 0.39); 600 is past what the model trained with (150) and hurts. The mates are fine; the fit's error is a small turn error (3.6 deg) carried over the whole turn from the query frame: carry from where the point was last seen instead Spawning here was starved: one budget, the first frame's background and frames 1-4 took it all, none was left for the ball's back (frames 10-19); rerun with the first frame outside the budget. |
| fit_last_seen | the whole-piece fit carrying each point from the nearest frame it is believed seen in (last_seen), mates seen at both ends (83e0b51); spawning 300, the ball and the standard test, long_spawn step 4,000, dev copy | well above carrying from the query frame (the turn error no longer carried over the whole turn) | hidden moving, guess / fit from the query frame (robust) / from last seen: ball 0.261 / 0.367 / 0.385 (10+ seen: 0.398 / 0.575 / 0.555; none seen: 0.080 / 0.149), standard 0.535 / 0.490 / 0.511 | a small gain, not the jump expected: the turn carried over is not the main cost. The largest gap left is the model not following its fit (ball 0.26 against 0.37-0.39), which only training can change Spawning here was starved: one budget, the first frame's background and frames 1-4 took it all, none was left for the ball's back (frames 10-19); rerun with the first frame outside the budget. |
| fit_depth_split | fit_last_seen's last-seen fit with the clip's own points (mates and the point's start) at their true depth with their predicted image place, at their true image place with their predicted depth, and both; spawning 300 | which part of the seen points' error costs the fit | hidden moving, last seen / true depth / true image / both: ball 0.385 / 0.509 / 0.473 / 0.993 (10+ seen: 0.555 / 0.662 / 0.663 / 0.997), standard 0.511 / 0.550 / 0.598 / 0.746 | neither alone: depth and image error cost about the same, and only both restore the fit. Seen points within the benchmark's thresholds (0.90) are not precise enough for a fit that reaches across the object; the fit should lean on the mates most likely precise Spawning here was starved: one budget, the first frame's background and frames 1-4 took it all, none was left for the ball's back (frames 10-19); rerun with the first frame outside the budget. |
| fit_errors_dealt | fit_depth_split with the seen points' own errors dealt out at random among the points seen in each frame, on true places (a first version dealt them among all points, hidden ones too: void) | near 0.99 if the errors' pattern is what costs, near 0.39 if their size | hidden moving, last seen / errors dealt / true: ball 0.385 / 0.443 / 0.993 (10+ seen: 0.555 / 0.649 / 0.997), standard 0.511 / 0.551 / 0.746 | mostly their size: the fit is standard but carries seen points' errors of about 1.6 cm (each within the benchmark's thresholds) to several cm at hidden points across the object; a pattern among them adds a little. The fit inspector page (scripts/fit_view.py) shows one ball clip Spawning here was starved: one budget, the first frame's background and frames 1-4 took it all, none was left for the ball's back (frames 10-19); rerun with the first frame outside the budget. |
| fit_points_1000 | the fit checks with test-time spawning filling each clip to 1,000 points (the next run's setting; spawn_count: about 98% of the points added after frame 0 land on the ball, every frame from 3 to 23), long_spawn step 4,000 | | ball hidden moving: guess 0.135 (trained with 500 points: 0.26 with 150 added), fit from the query frame 0.173, from the last seen frame 0.381, errors dealt 0.474, both true 0.877; standard: guess 0.539, last seen 0.522 | the model must train with the setting it is tested in; carrying from the last seen frame holds where carrying from the query frame collapses (its mates must be seen in that frame): it is now the only fit (8fe8be5) |
| fit_object_pose | the fit checks at 1,000 points with two new placements: frame by frame (each step from the piece's points seen in both frames) and each object kept as one rigid shape whose pose is fit every frame from its seen points (object_places, 6136440; objects = the groups' pieces joined through likeness over 0.5); each from the model's places and from true ones | hidden near the seen points' 0.9 | ball hidden moving, model's guess / old fit / frame by frame / object pose: 0.207 / 0.434 / 0.558 / 0.697 (true places: frame by frame 0.966, object pose 0.997); the ball's seen moving under object pose 0.962 against 0.853 its own. Standard: object pose 0.518 hidden, 0.790 seen (true places 0.692 / 0.867) | one rigid shape per object is the right placement: on the ball it lifts hidden points from 0.21 to 0.70 and seen ones from 0.85 to 0.96. On the standard test the pieces are wrong: joining points through likeness over 0.5 merges objects (true places only reach 0.69) |
| fit_object_pose_mates | object_places with each point's object its own group-mates, no chains (c-commit after 6136440) | the standard test's objects no longer merged | ball as before (hidden 0.688, seen 0.961); standard hidden 0.559 (chains 0.518), seen 0.866 (0.790; own 0.843), true places 0.751 / 0.954 (10+ seen mates: 0.781, true 0.952) | no chains fixes most of the merging; what the standard test lacks with true places is the 19% of hidden point-frames with no seen mate at all, where no fit can help |
| ball_whole | the ball re-rendered with tracks over the whole ball, each queried in the first frame it is seen (render_ball, tapvid3d_cache --one-clip), spin_up_10, points drawn as training draws them (1,000, up to 500 new surface, rigid_check --train-spawn 500), long_spawn step 6,000 | the fits with truth for every point | seen moving only 0.563 (the old ball, same checkpoint: 0.853); the predicted turn 40 deg against the true 80 | the model tracks these clips far worse: their points start through the clip, not at frame 0. Either the new cache is wrong or the model handles late-starting points badly; nothing from this test until that is settled |
| fit_drift | object_places on the ball (filled to 1,000), long_spawn step 6,000: error by frame for seen points in the fit, a fifth kept out of it, and hidden ones; error around the frame each point hides in; and the shape refined from every frame (bundle adjustment, robust) | hidden points worse because they leave the fit, or because of something about the frames they are in | by frame (cm, median), in the fit / kept out / hidden: frame 1 0.2 / 0.2 / 0.9, frame 9 0.9 / 1.0 / 1.9, frame 15 1.5 / 0.7 / 4.6, frames 21-23 ~8 / ~8 / ~5.7; around hiding, frames -3 .. +8: 0.5, 0.6, 0.8, 0.9, 1.2, ... 3.5 (no jump). Refined from every frame: hidden 0.629 against 0.688, seen 0.884 against 0.961 | leaving the fit costs nothing and nothing jumps when a point hides: the pose itself drifts as the spin goes on, every point with it, and hidden points live in the later frames, farther from the seen ones. Refining the shape from every frame made it worse: the drift is shared by the seen points (the model's tracks drift together), which no fit to them can remove. Refinement removed |
| fit_fresh | object_places on the ball (filled to 1,000) with each mate weighted 1 / (1 + turn / F)^2, its turn how far its object has turned since it was given (the model's own error on seen points: 0.27 cm within 15 degrees of that, 0.58 at 15-30, 0.78 at 30-60, 1.24 at 60-90, 2.47 past 90) | fresh mates steady the pose | hidden moving / seen moving: F off 0.688 / 0.961, F 12 deg 0.787 / 0.956, F 30 deg 0.876 / 0.976 | weighing mates by how fresh they are lifts the ball's hidden points from 0.69 to 0.88, near the model's own seen points (0.85): the design's fit (994b63d, --fit-fresh 30) |
| clip_units_gravity_latest_template | clip_units_gravity_no_state_head with each window's matcher template the point's latest clear look the window before settled (a frame seen whole where it is believed visible) instead of its query-frame look, the first window keeping the query look (`--template latest`, 68cfc42); same start, schedule and data (gdev, 2 h: compared at step 500) | seen moving above clip_units_gravity_no_state_head's at the same step: turned points are matched against a look a few frames old, not up to 23 | stopped after step 500. Own val: all 0.758, seen moving 0.879 (0.783 / 0.880 without). Standard test at 500: all 0.841, moving 0.729, non-visible 0.451, moving non-visible 0.351 (without: 0.864 / 0.768 / 0.464 / 0.377). On the no-template run's step-500 weights, untrained for it (scripts/template_check.py, e18fb32), standard all / moving / moving in frames 12-23: query look 0.864 / 0.768 / 0.666, latest look where guessed 0.817 / 0.709 / 0.543, latest look where true 0.891 / 0.817 / 0.775 | a fresh look is worth 0.11 on late moving points when it sits where the point is, and costs 0.12 where the model guessed it: the guess's error becomes the matcher's target. Placing it, not the idea, is wrong; the only exact looks are query points (new points, ideas.md). Code removed | does a fresh template recover the turned points' precision |
Standard scoring (100 clips, all 256 points, like run33's 0.878 / 0.928): fyJ at step 1,500
val 0.871 / train 0.876; moving hidden val 0.316 (run33 0.283; last seen 0.188, last speed 0.267).

## More visible points for the fit (13 Nov, scripts/new_points_check.py, 6334377)

clip_units_gravity_fit_loss_3k's step-500 weights, the standard test's clips
from the dense cache: the 256 scored points tracked alone; in a pass of their
own, the clip's next points (median 247) given as new points in frame 8 where
seen. The scored points' hidden moving point-frames from frame 8, placed by a
robust fit of the seen points of their true piece (truth for seen), APD:

| mates | all | 10+ old seen | none old seen |
|---|---|---|---|
| none (own guess) | 0.540 | 0.564 | 0.138 |
| old points | 0.664 | 0.698 | - |
| new points | 0.594 | 0.603 | 0.312 |
| both | 0.613 | 0.625 | 0.312 |

Where every old point of the piece is hidden (it turned or moved away from
them), new points more than double the placement; where old ones are seen,
they are better (carried from their exact query, the new ones carry the old
point from where it was guessed at frame 8). Spawning gives the fit points
where it had none: in long_physics_spawn.

## Training faster (13 Nov)

Two clips a step (`--batch 2`) fails before any memory limit: clips of
different sizes cannot be stacked (24 and 48 frames, fewer points where the
30 m reach drops some), which is likely why runs went to one clip a step when
our renders joined. Training on several GPUs instead (66990f6): one process a
GPU, each on its own clips, gradients averaged before every step, the weights
checked alike every 100 steps; 2 GPUs ran 80 steps in sync. Queues: g1 is 1
GPU for up to 24 h, g2 2-3 GPUs for 24 h, gdev 1-4 GPUs for 2 h, each with its
own one-job limit, so a g2 run can go beside a g1 run.

## The groups now (13 Nov, scripts/check_groups.py)

clip_units_gravity_fit_loss_3k's best weights, 500 points a clip. Of the mates
the model puts with a point, the share truly on its piece (2 cm), by how far
the point moves. Fall test (24 held-out clips): 0.979 still, 0.742 at 1-4 px,
0.707 at 4-8, 0.88-0.95 at 8-64, 0.966 past 64 px (2,821 of the 3,755 moving
points); found 0.71-0.99. Our renders (60 clips trained on), against the 2 cm
rule and against the true objects: 1-4 px 0.626 and 0.490 (wrong mates 73% on
the background), 4-8 px 0.808 and 0.798, past 64 px 0.958 and 0.958. Most
moving points have the right mates now (0.61-0.81 on full_design_fixed_sweeps);
slow movers are taken for the background, and they match the objects no better
than the 2 cm rule, so the label rule is not what confuses them: they look
still. The few wrong mates left are what the robust refit removes.

## The fit made robust (13 Nov, scoring only)

On clip_units_gravity_fit_loss_3k's step-500 weights (scripts/rigid_check.py,
1902b79): the fit refit 3 times, each mate's weight cut by how far it sat from
the last fit against the typical mate. Hidden moving: guess 0.384, fit 0.287,
robust fit 0.361, with true groups 0.441, robust 0.430. With 10+ seen mates
(43% of those point-frames): guess 0.527, fit 0.434, robust 0.574, true groups
0.640, robust 0.665. Seen moving: 0.772 / 0.617 / 0.729. A few wrong mates
were what dragged the fit: refit without them it beats the model's own guess
wherever 10 or more mates are seen. In a toy with 3 of 20 mates wrong, the
fit's error on the right ones goes 0.145, 0.047, 0.004, 0 over 0-3 refits.

## Turned points: fresh looks and new points (12 Nov, scoring only)

On clip_units_gravity_no_state_head's step-500 weights. A template refreshed
to the latest clear look the window before settled, standard moving in frames
12-23: 0.666 with the query look, 0.543 read where the model guessed the point,
0.775 read where it truly is; the last partly hands over the answer (frames
12-15 compare against a look at frame 15 or before, at the true position), so
the gain of a fresh look is smaller than it shows.

New points (scripts/new_points_check.py, 5551d74): the dense versions of the
100 standard clips, their tracks past the 256 scored given again in frame 12
where seen (median 243 a clip, true position, look read there). The scored
points' moving APD in frames 12-23: alone 0.653, beside the new points 0.624,
placed by a rigid fit of the new points on their true piece from frame 12
0.626 (the fit reaches 35% of those point-frames with 3+ seen mates). New
points hurt a model that rarely trained with late points, and carrying the old
ones by exact new mates gains nothing: the old point's own position at frame
12 already holds its drift.

So turned points are not to be fixed by a fresher look. TAPIP3D, with the same
frame-0 template, drifts 0.6% of the distance moved when trained long and 7-8%
at our budget (tap_short; ours 2.6%): the drift on turned points is what long
training removes.

## When the rigid fit helps (scripts/rigid_check.py, 12 Nov)

full_design_clip_spread_3k (its own code, 8cc0337), old Kubric val, hidden
moving point-frames by how many points of their true piece are seen there. APD,
no rescale:

| seen mates | share | guess | fit, model's groups | fit, true groups |
|---|---|---|---|---|
| none | 0.186 | 0.238 | 0.104 | 0.184 |
| 1-9 | 0.381 | 0.417 | 0.228 | 0.409 |
| 10+ | 0.433 | 0.573 | 0.428 | 0.706 |

The fit is worth having only with many seen mates and the right groups (0.706
against the guess's 0.573); with few or none it is no better than the guess,
and the model's groups make it worse everywhere. A fifth of hidden moving
point-frames have no seen mate at all.

## Does the full design use its rigid proposal (scripts/without_rigid.pbs, 8 Nov)

full_design_fixed_sweeps with and without its rigid proposal: the same
weights, the proposal switched off.

| test | with | without |
|---|---|---|
| standard: all / non-visible / moving non-visible | 0.888 / 0.485 / 0.380 | 0.884 / 0.475 / 0.376 |
| same clips, forecast from 12: moving | 0.295 | 0.297 (stop 0.322) |
| fall test, forecast from 3: moving | 0.064 | 0.063 |
| ball: facing / edge / hidden / returned px | 4.11 / 4.03 / 30.7 / 83.0 | 4.10 / 4.11 / 30.6 / 83.4 |

The model barely uses it: 0.01 on Kubric's non-visible points, nothing on
forecasting or the ball.

Why (scripts/rigid_check.py): the proposal's own position against the truth,
beside the model's final guess, and the proposal built from the true groups
(the training labels' pieces) instead of the model's. APD, no rescale:

| clips, point-frames | guess | proposal | proposal, true groups |
|---|---|---|---|
| old Kubric, hidden moving | 0.380 | 0.189 | 0.453 |
| old Kubric, seen moving | 0.802 | 0.492 | 0.813 |
| our clips, hidden moving | 0.123 | 0.093 | 0.129 |
| our clips, seen moving | 0.415 | 0.290 | 0.328 |
| ball, hidden | 0.072 | 0.088 | 0.123 |
| ball, seen moving | 0.521 | 0.528 | 0.477 |

- The model is right to ignore it: with its own groups the proposal is
  worse than its guess almost everywhere (half as good on old Kubric's
  moving hidden points).
- On old Kubric the groups are what is wrong: with the true ones the
  proposal beats the guess on hidden points (0.453 against 0.380) and
  matches it on seen ones. The fit itself is right (exact on a toy turning
  piece).
- On our clips and the ball even the true groups help little: the proposal
  is built from the mates' tracked positions, and those are poor there
  (seen moving 0.41 and 0.52).
- Its backing says about 0.96 whether it is right or wrong: it counts how
  much weight the mates carry, not how well they fit one rigid motion.

The groups themselves (scripts/check_groups.py, old Kubric, 100 clips, 256
points): of the mates the model puts with a point, the share truly on its
piece (2 cm) is 0.93 for still points, 0.61-0.72 for points moving 1-16 px
and 0.75-0.81 for those moving more; it finds 0.81-0.97 of its true mates.
Mostly right, but a fifth of a moving point's mates are on another piece
(their motion is 44 px off for points moving over 64 px), and one least-
squares fit over all of them is pulled far off by that fifth. Fixes to try:
weight mates more sharply than the affinity (only near-certain ones count),
a fit that drops mates that disagree with it, and a backing that is how well
the mates fit one rigid motion.

New clips for group labels by object (8 Nov, 54ce190): neither TAPIP3D's
Kubric clips nor ours kept which object a point is on, so ~500 of ours are
being rendered again (`~/scratch/kubric_obj`, seeds 0-495) with each point's
object saved, 48 frames (the length training reads; before, 48-96 were
rendered and the rest cut), and the moving objects given half the points
(30-50% of points move, against 2-20% before, when half the clips failed the
moving filter). Four review clips: depth within 0.1% (median), every query
frame visible, every clip past the filter (241-665 moving points seen in
frame 0); `scripts/render_raw_clip.py` draws them coloured by object.

From 8 Nov, every cache serves only the points within 30 m of the camera in
their query frame (`REACH`): Kubric's sky is the inside of a dome 40 m out
with a photo on it, and its points taught the model the sky is 40 m away.
Dropped: 2.6% of the old Kubric's points, 14% of our clips', 3.4% of the
standard test's, none of the ball's. Standard scores from here on are on the
near points only, so earlier runs' are not exactly comparable.

## Where the sweeps lose their tracks (scripts/sweep_curve.py, 7 Nov)

Standard test (100 old-Kubric clips, 256 points, every frame seen), the step
design's checkpoints on its own code (c2eec27). Windows 0-16 and 8-24. APD of
the seen points, no rescale:

| frame | fbfb after sweep 1 (f) | after 2 (b) | after 3 (f) | after 4 (b) | forward rounds |
|---|---|---|---|---|---|
| 4 | 0.868 | 0.932 | 0.962 | 0.964 | 0.968 |
| 8 | 0.786 | 0.848 | 0.906 | 0.903 | 0.934 |
| 16 | 0.798 | 0.801 | 0.876 | 0.864 | 0.904 |
| 23 | 0.787 | 0.787 | 0.837 | 0.837 | 0.890 |

Whole clip (median rescale): fbfb 0.751 / 0.793 / 0.871 / 0.871, forward
rounds 0.907. Points lost (beyond 16 px at their depth) at frame 23: fbfb 7.0%
after sweep 1, 3.2% at the end; forward rounds 1.6%. Hidden points at frame 23:
0.454 against 0.513.

- The first window (frames 0-7) ends level with forward rounds; the whole
  drop is in the second window and grows towards the clip's end.
- One round, then handed on: the second window starts from a rough first
  window and its first round ends at 0.79, where forward rounds' second window
  starts from a finished first one and ends its first round at 0.85-0.93.
- The backward sweeps cannot fix the clip's end: frame 23 is a backward
  window's held frame (where its steps start), so it never moves (0.787 ->
  0.787, 0.837 -> 0.837) and the frames near it barely do. The second window
  gets two useful rounds, the forward ones, not four.
- Both are what the code did, not the idea: the positions design holds no
  frame (old_recipe_unified_positions), and ffbb locks each window before
  handing it on (old_recipe_unified_lock).

The same on old_recipe_unified_positions (positions, nothing held, fbfb), on
the current code:

| frame | after sweep 1 (f) | after 2 (b) | after 3 (f) | after 4 (b) |
|---|---|---|---|---|
| 4 | 0.826 | 0.911 | 0.954 | 0.958 |
| 8 | 0.780 | 0.840 | 0.871 | 0.902 |
| 16 | 0.778 | 0.815 | 0.860 | 0.863 |
| 23 | 0.735 | 0.785 | 0.789 | 0.819 |

Whole clip 0.675 / 0.785 / 0.840 / 0.845; lost at frame 23: 5.9% (step fbfb
3.2%, forward rounds 1.6%).

- With no held frame, the backward sweeps now move the clip's end (frame 23
  0.735 -> 0.785, and 0.789 -> 0.819), where the step design's never did.
- But every frame is lower than the step design's, the first window too
  (frame 4: 0.958 against 0.964; first sweep 0.826 against 0.868): this model
  corrects less per round. old_recipe_unified_forward_rounds says whether that
  is the positions design itself, with its output layer started from zero, or
  the sweeps.

The same on old_recipe_unified_lock (ffbb, clip played backwards) and
old_recipe_unified_forward_rounds. For ffbb, stage 1 is the clip after the
forward sweep and stages 2-3 the backward rounds; for forward rounds each
stage is every window's round k:

| frame | ffbb after f, f | after b, b | forward rounds after 2 | after 4 |
|---|---|---|---|---|
| 4 | 0.931 | 0.959 | 0.942 | 0.965 |
| 7 | 0.867 | 0.927 | 0.883 | 0.941 |
| 8 | 0.869 | 0.911 | 0.904 | 0.914 |
| 16 | 0.860 | 0.872 | 0.895 | 0.897 |
| 23 | 0.815 | 0.845 | 0.862 | 0.879 |

- The backward rounds work: on the first window they add as much as forward
  rounds 3-4 do (frame 7: +0.060 against +0.058), the clip played backwards
  or not.
- The rest of the gap is the handover again: the second window starts from a
  first window two rounds into its four (frame 7 at 0.867, against 0.941), so
  its first rounds end lower (frame 23 0.815 against 0.862) and the backward
  rounds do not close it. Two rounds do not lock a window.
- Hidden points follow the same pattern: level in the first window, below in
  the second (frame 23: 0.409 against 0.471).

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

## TAPVid-3D, 10 Panoptic Studio clips (quick check, `scripts/tapvid3d_score.pbs`, `scripts/tapvid3d_ref.pbs`)

Every 8th query frame, each query's track scored as its forward and backward
half-clips (322 half-clips), Depth Anything 3 metric depth, queries starting
at their pixel at that depth, our scorer (the benchmark's metric, median
rescale). TAPIP3D is its published checkpoint at its defaults (384 x 512, 16 x
16 support grid) on the same half-clips, frames, depth and queries.

| | APD | AJ | OA | moving hidden APD |
|---|---|---|---|---|
| nothing moves | 0.202 | | | |
| fyT | 0.271 | | | 0.080 |
| fyZ | 0.272 | 0.175 | 0.822 | 0.083 |
| TAPIP3D | 0.287 | 0.181 | 0.815 | 0.097 |

Real video is hard for both at this depth: TAPIP3D, trained far longer, is
only 0.015 APD ahead of us, and both barely beat a prediction that nothing
moves. The depth (Depth Anything 3, per frame) is the first suspect.

## Learned groups against the truth (`scripts/check_groups.py`, fzC, 50 val clips)

Of the points a point's group holds, 96-99.6% truly move with it for points
moving 8 px or more, and 85-97% of its true mates are in it; points moving
1-8 px are put with the still floor (54-66%). The groups are right where it
matters, so they are not why the group bias goes unused.

## Against last week's report (standard scoring: 100 val clips, 256 points; `scripts/fit_check.pbs`)

Tracking with every frame's image; forecasting with frames 12-23 blank
(CUTOFF=12). fyO is last week's reported model (week 9).

| | all points | non-visible | moving non-visible | forecast (all) | forecast (moving) | seen moving, forecast run |
|---|---|---|---|---|---|---|
| fyO, last week | 0.911 | 0.488 | 0.401 | 0.632 | 0.379 | 0.892 |
| fyP, frame-before comparison | 0.921 | 0.556 | 0.454 | 0.635 | 0.386 | 0.908 |
| fyT, fyP's line on the dense data (the ball model) | 0.920 | 0.546 | 0.448 | 0.625 | 0.385 | 0.907 |
| fyV, rigid fit in rounds 1-2 | 0.906 | 0.546 | 0.463 | 0.609 | 0.346 | 0.871 |

fyP is the single best: +0.010 on all points, +0.068 on non-visible points,
forecasting level or a little better. fyT is level with it; the dense data and
the shut gate added nothing on this protocol. fyV's fit helps moving
non-visible points and costs everything else.

## Overnight, 1 Nov (queued on NSCC)

- fzE (true groups + hidden pull, gdev) and fzF (learned groups + hidden pull,
  g1) at step 250: moving hidden 0.664 / 0.665, as fyZ / fzD. To step 500 (fzE)
  and 1,000 (fzF).
- fzG: the window handover fix trained in (`--reach continue`), g1, 1,000 steps.
- After each run ends: a ball render to `~/scratch/renders/ball_<run>` (spin up
  10 / 20, roll), and for fzF and fzG the standard scoring against last week
  (`fit_check_val_3d_median_m0_p256{,_c12}.json` in the run's outputs).

fzE at step 500: moving hidden 0.681, the same as fyZ, fzA, fzC and fzD. Every
change to how points attend to their group lands at the same score. The ball
movies showed hidden points stopping at frame 16, the window handover; fzG
tests that directly.

## On the spinning ball, error by how the point faces the camera (fyT, `scripts/ball_view_angle.py`)

Median error in metric px (share within 4 px). Facing / oblique / edge count
only points never hidden so far; returned = seen again after hiding.

| motion | facing <45 deg | oblique 45-70 | edge 70-90 | hidden | returned |
|---|---|---|---|---|---|
| slide | 0.25 (1.00) | 0.37 (1.00) | 0.83 (0.99) | 2.2 (0.89) | - |
| spin up 10 | 0.58 (0.97) | 0.79 (0.97) | 1.54 (0.98) | 21.8 (0.12) | 82.9 (0.00) |
| spin up 20 | 0.85 (0.75) | 1.08 (0.82) | 2.24 (0.74) | 41.4 (0.02) | 70.2 (0.00) |
| spin side 20 | 0.87 (0.81) | 0.99 (0.83) | 2.54 (0.70) | 42.7 (0.02) | 59.4 (0.01) |
| roll | 0.70 (0.88) | 1.35 (0.80) | 2.68 (0.67) | 13.3 (0.11) | 28.0 (0.00) |

A point is tracked to about a pixel for as long as it stays in view, turned
or not; worse towards the edge (2-3 px). Once hidden it is lost (13-43 px),
and when it comes back into view it is not found again: returned points are
further off than hidden ones, none within 4 px. The turn's cost is hiding and
the missing re-find, not the turn itself.

## On Kubric, always seen against seen again after hiding (fyP, `scripts/return_split.py`, standard 100 val clips)

Moving points, every frame given its image. Median error in metric px (share
within 4 px).

| class | frames 1-7 | frames 8-15 | frames 16+ |
|---|---|---|---|
| always seen | 0.34 (0.98) | 0.81 (0.92) | 1.00 (0.89) |
| seen again after hiding | 0.73 (0.88) | 1.30 (0.76) | 1.73 (0.67) |
| hidden | 1.47 (0.78) | 4.80 (0.43) | 8.01 (0.25) |

Unlike the spinning ball, points that come back are mostly found again on
Kubric (a point leaving an occluder reappears near where it is expected),
though they stay about twice as far off as points never hidden. Points never
hidden also drift, 0.34 to 1.0 px over the clip: smaller, but real. By frame
16 about a third of the seen moving point-frames are returned ones. Hidden
points are the large error, growing to 8 px.

Hidden points' images blanked wherever they are truly hidden (`--hide-hidden`,
the forecast frames' blank, scoring only): hidden points get worse, not better.
Spin up 10: hidden 21.8 to 41.7 px, returned 82.9 to 132.9; slide hidden 2.2 to
9.6; seen points unchanged. The match a hidden point gets is not what bends
the ball: it helps (its 3D neighbourhood still holds the turning front of the
same ball). With no image the model falls back on what it does for forecast
frames, which carries the points worse. Hidden points and forecasting share
one missing ability: moving a point from what the rest of its object does.

## Forecasting a bounce in plain view (withdrawn)

The hand-made drop (`render_ball.py --only drop`) fell under about 0.8 m/s^2,
set by "lands at frame 15", not real gravity: a model that learned Kubric's
gravity is wrong on it by construction. Its numbers (fyP 0.273 against
"keeps its last speed" 0.291) are withdrawn and the test removed. In its
place `scripts/fall_test.pbs`: forecasting from frame 3 on the held-out val
clips of our Kubric set, where thrown objects fall, bounce and tumble under
Kubric's own physics.

## Overnight, 3 Nov (queued on NSCC)

The 512 Kubric clips rendered (4.7 min each at 256 px, 32 samples); 240 kept.
Queued in a chain: caching them (DINOv3, frames), fzJ, then for fzJ and fzI
each: the drop test (re-rendered, the ball in view), the view-angle check on
spin up 10, and the standard scoring (val, and forecast from frame 12).
PointOdyssey's sample (3.3 GB) and val (20.4 GB) are downloaded to
~/scratch/pointodyssey, not yet unpacked.

## Forecasting falls on held-out Kubric clips (`scripts/fall_test.pbs`)

The 24 val clips of our Kubric set, frames 3-47 with no image: objects thrown
at frame 0 fall, bounce and tumble in the forecast, under Kubric's physics.

| | forecast, moving | forecast, all | seen moving (frames 0-2) |
|---|---|---|---|
| "stops where last seen" | 0.047 | | |
| "keeps its last speed" | 0.057 | | |
| fyP (best, old data) | 0.107 | 0.498 | 0.773 |
| fzI (3 rounds, hidden blank, old data) | 0.106 | 0.469 | 0.763 |
| fzJ (fzI trained on these clips' training split) | 0.106 | 0.499 | 0.770 |
| fit_truth | points_1000_object_pose's step-1,000 weights, scripts/rigid_check.py on the standard clips (100 old-Kubric val clips) without filling, so every point has its truth and the all-true column exists; once with --points 1000 --spawn 500 (the cache holds 256 tracks a clip, so the same as the second) and once with 256 points (gdev, 49283e9 and after: the parallel fit) | all true near 1: the fit's arithmetic right on these clips too | hidden moving, guess / object / true image / true both / errors dealt / true groups / true seen / all true: 0.568 / 0.594 / 0.739 / 0.983 / 0.698 / 0.604 / 0.601 / 0.998; seen moving 0.889 / 0.893 / 0.951 / 0.993 / 0.983 / 0.895 / 0.895 / 0.994. Filled to 1,000 (rigid_object_pose): guess 0.595, object 0.633 | the fit is right (all true 0.998) and the learned groups and visibility cost almost nothing (true both 0.983): the whole gap is the seen points' guessed positions, mostly their place in the image, and their errors move together (dealt at random 0.698 against 0.594). The 0.906 'true both' with filling was the helpers, which have no truth. The model's own helpers lift hidden moving 0.568 -> 0.595 | is the fit's logic or its inputs what holds hidden points at 0.6 on the standard clips |
| ball_ignores_fit | points_1000_object_pose's step-1,000 weights on 0d7ed4a, every round logged at hidden moving point-frames (true visibility, over 4 px): the fit input's offset (clip units and px at 256), its support channel, the visibility belief, the share of the gap (fit placement - the round's input guess) the round's correction closes, and the errors before / of the fit's placement / after (px, median). Ball spin_up_10 filled to 1,000; 30 standard clips filled to 1,000; 20 of our clips and 20 old-Kubric clips as training draws them (1,000 points, up to 500 from true tracks). Rounds 0-1 forward, 2-3 backward, each window's first round 0 / 2 (one-off probe, gdev) | why the model's final guess on the ball (0.31) stays far from the fit's placement (0.82) when on the standard clips it nearly follows (0.595 / 0.633) | Median px, before / fit / after, and the share closed: ball round 2 4.92 / 1.04 / 2.66 (0.37), round 3 2.66 / 3.52 / 2.69 (0.05); standard round 2 2.69 / 1.95 / 1.96 (0.62), round 3 1.96 / 2.63 / 1.94 (0.09); old Kubric round 2 3.15 / 2.29 / 2.34 (0.79), round 3 2.34 / 2.99 / 2.29 (0.08); our clips round 2 9.15 / 11.6 / 8.13 (0.14), round 3 8.13 / 11.9 / 7.87 (0.00). Belief at those point-frames: ball median 0.01-0.02 (share over 0.5 at most 0.03), standard 0.04-0.11, so the model knows the ball's points are hidden. Support near 1 on the ball from round 1 on (no support 0.0%). Offsets in round 2, median / 90th, clip units: ball 0.049 / 0.123, standard 0.0047 / 0.031, old Kubric 0.0048 / 0.030, our clips 0.017 / 0.35 | two causes. (1) A bug: in a window's second round the fit's offset is taken from the clip's guess before the window's first round (Sweeps.__iter__ calls proposal on self.whole, which is only updated after the window's rounds) but added to the window's guess after it, so it is stale by exactly the first round's correction: the fit's placement handed in goes from 1.04 px to 3.52 px on the ball between rounds 2 and 3 though its inputs improved, the same on every clip set, and the model learned to ignore second rounds' fit (closes 5-9%). The last round, which gives the final answer, is a second round. The fit loss already splices the window's guess in (regress.py, `run.proposal(torch.cat([run.whole[:, :r.start], g, run.whole[:, r.end:]], 1)`); the input should do the same. (2) In first rounds, where the offset is right, the model follows a large offset less: on our clips (the only training set with large offsets) the fit is worse than the guess (11.6 against 9.15 px), so training teaches that a large offset is wrong, and the ball's offsets are ten times old Kubric's. To confirm (1): hand in the offset from the spliced guess and rerun this probe: round 3's fit error should match round 2's (about 1 px on the ball) at once, and after training the share closed in second rounds should rise. Confirmed on 8d996d4 (the offset from the window's own guess; also carries the parallel fit, carry-forward and the new fill rule), same weights, untrained: the fit placement handed in, median px, round 2 / round 3: ball 0.76 / 0.47 (was 1.04 / 3.52), standard 1.89 / 1.73 (was 1.95 / 2.63); the share closed in round 3 still 0.03 on the ball and 0.06 on the standard clips, the model trained to ignore it. Final hidden moving APD filled to 1,000, guess / fit: ball 0.372 / 0.892 (eb1f15c: 0.313 / 0.815), standard 0.606 / 0.618 (0.595 / 0.633). The handed fit is now right in every round; the model must be trained on it to follow | why does the model not follow the fit on the ball |

The models forecast real falls about twice as well as either guess, so
something of falling is learned, but at a low level; 1,000 steps on 216 clips
of physics did not move it at all.

Code cut on 3 Nov, the experiments recorded above: the rigid fit and all
around it (`--rigid-fit`, `--fit-rounds`, `--rot-match`, `--fit-input`,
`--fit-loss`), the whole-frame grid (`--grid-cmp`, `--grid-window`,
`--fine-prev-skip`), the true-groups and hidden-pull diagnostics
(`--group-oracle`, `--group-vis-bias`), the window-start option (`--reach`),
`--back-iters` (the schedule does it), and the step tracker. Checkpoints that
used them no longer load.

## Design audit, 4 Nov (an independent read of the code against the design)

Bugs found and fixed (`ff7cc56`):

- Every run with `--hidden-blank` (fzI, fzJ, fzL, fzM, fzN, fzO) blanked every
  point in the new frames of each window after the first, on the first
  forward sweep: the visibility store started at 0, read as "hidden". Those
  runs tracked frames 16+ blind on their first pass at every step. Their
  results, and the conclusions drawn from them (hidden blanking no gain, the
  physics data no help, the design no better), are not valid tests.
- Forecast scores from fzL on counted points first seen inside the forecast,
  which sat at their true position there: inflated. Scoring now keeps only
  points first seen before the cutoff.
- The per-point loss weight used depth at frame 0, wrong (even negative) for
  points first seen later; now the query frame's, held to at least 10 cm.
- Frames before a point is first seen were matched against its template;
  now always blanked.
- The tracking loss was divided by the number of sweeps a second time.

Open design question: a point first seen inside a window has steps that run
from its query frame both ways, but the model is told one direction for the
whole window.

## Cleanup, 4 Nov

The code now holds the current design only: the model is the CNN matcher (3D
neighbourhoods against the template and the frame before), DINOv3
cross-attention and the motion input, and training and scoring share one
sweep loop (`regress.Sweeps`). Removed: flow matching, the whole-frame match
and its head, the cost volume, the locality prior, upsampling, causal
attention, the flow patch, the depth head, fed-back visibility, buddies,
virtual tracks, TAPIP3D's updater, the position and acceleration params,
carry, augmentation, best-of-k, support points, and about 30 scripts of those
or of finished checks. 8600 lines out, 900 in.

The training flags' defaults are now fzN's design, so a run names only what
it changes (`scripts/train.pbs`). Only fyT and fzN are kept loadable (their
two dead finder weights stripped); fyP and older checkpoints no longer load.
`scripts/snapshot.pbs` checks that fyT and fzN predict, and train one step,
exactly as before the cleanup: predictions identical, the step's loss
identical and its gradients within the GPU's run-to-run noise (6e-5). A
100-step run of the default design (`cleanup_check`, from fyT) trained,
validated and saved end to end; validation is one pass now, 80 s for 24 clips.

## Fixes after the cleanup, 4 Nov (`f35cca2`, `c0d0077`)

From a review of the lean code against the design:

1. Unseen is never final. The visibility head used to blank a point's image
   in a frame it called hidden, for every later round: with no image there,
   nothing could call it seen again, so a point that came back was never
   re-found. Now what the head last said about every frame is fed back to
   the token (carried through windows and sweeps), the image stays, and a
   fact beside it says whether the frame is before the point's query frame,
   or is it. `--hidden-blank` is gone. Half the training clips now lose a
   span of 2 to `--max-gap` (12) frames in the middle instead of their end,
   so re-finding a point after it was unseen is practised by every point.
2. The matcher's frame-before comparison compared a window's first frame
   with itself (always a perfect match); it now reads the clip's frame
   before, and is off unless both frames have their image.
3. A point first seen in a later window started from the group vector it
   said before it was in any window; now from zeros.
4. Scoring measured from frame 0: the static baseline, moving points and the
   hidden baseline's fallback now use each point's query frame. Only the
   scores in use are kept (26 a clip).

These change what fyT and fzN predict; no run of the fixed design exists yet.
