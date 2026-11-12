# Ideas

The one place for what is decided to come later, and what was ruled out. One
line each; a line leaves when it is built (runs.md then has it) or dropped.

## Later

- Camera roll in our renders, so finding down is real work: only if the down output's angle error is poor on rolled real video (ADT).
- Real video's frame rate in the gravity strength (fixed at Kubric's 12 fps now), and frame-rate variety in training.
- Two corrections: tracking in the clip's unit, physics in real units.
- Forecast as Learning to Simulate (GNS) does: in frames without an image, each point's acceleration is the output and its position comes from integrating it (velocity from past positions as input, the loss on acceleration at the data's scale, noise on the inputs in training); our side head beside the position correction was the wrong form.
- Noisy-depth training with a random global depth-scale error, so gravity can correct the scale.
- Rigid proposal through the fit: not detached, a loss on the proposal's own positions, so the groups learn what the fit needs (grouping plan item 2).
- Pairwise motion check as a group input: how much each pair's distance changed over the frames both are seen.
- The fit's backing from its leftover error over the mates, not their weight, so a wrong fit says so.
- Two group outputs: an object vector on object labels, a rigid vector on motion.
- SAM 3 masks as an appearance input for real video, trained on its masks of our clips (EntitySAM lost to our groups on Kubric).
- Matcher template from the latest clear look (or a memory the model updates) instead of frame 0: visible drift follows turning.
- Spawn rule: new points where new surface appears.
- Whole-frame comparison, removed in a cleanup: needed eventually?
- More data: PointOdyssey or SynthVerse.
- The full scene-points design (docs/tasks/scene-points-2026-10-31.md).
- A long run once the design settles: TAPIP3D trains on about 28x our samples, with augmentation.
- Spatial attention in every block or once at the end; virtual points; model size (dim 256, depth 6, 4 heads is what trained, never chosen).

## Ruled out

- More forward rounds before the backward sweeps (ffffbb): rounds 3 and 4 moved the error only 1.07 to 0.92 px (fyO, by round), and ffbb already gives each window 4 rounds; turning, 3-4 times the error, is the bigger target.
- DINOv3 (77%) and EntitySAM (74%) groups: worse than the model's own (89%) on Kubric.
- The matcher on the fixed 9 m ruler (fzR): Kubric lost 0.017 against the clip's own spread.
- Caching cross-attention keys and values: 9.4% slower, memory-bound (flow-matching design, 2026-09).
