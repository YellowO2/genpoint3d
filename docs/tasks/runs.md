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
| fyN | fyM + DINOv3 (whole-frame match, cross-attention over the 49 nearest patches) | forecast moving 0.302 vs 0.286; seen moving 0.781 vs 0.793 | 1000: forecast moving 0.322 vs 0.304 (true-freeze 0.321); seen moving 0.840 vs 0.842; hidden moving 0.431 vs 0.436 | DINOv3 helps the forecast, tracking unchanged | forecast frames above fyM = DINOv3 helps forecasting |

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
