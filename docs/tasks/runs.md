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
Standard scoring (100 clips, all 256 points, like run33's 0.878 / 0.928): fyJ at step 1,500
val 0.871 / train 0.876; moving hidden val 0.316 (run33 0.283; last seen 0.188, last speed 0.267).

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
