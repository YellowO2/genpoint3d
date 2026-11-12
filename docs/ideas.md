# Ideas

The one place for what is decided to come later, and what was ruled out. One
line each; a line leaves when it is built (runs.md then has it) or dropped.

## Later

- Camera roll in our renders, so finding down is real work: only if the down output's angle error is poor on rolled real video (ADT).
- Real video's frame rate in the gravity strength (fixed at Kubric's 12 fps now), and frame-rate variety in training.
- Two corrections: tracking in the clip's unit, physics in real units.
- Physics per group: version 1 built (`--physics`, object_paths: the fit's centre of each point's mates and its turn carried on from the last two supported frames, plus learned accelerations; the reference is the mates' centre, not the true one, so a spin about another point drifts a little). Still to come from the fuller idea: per group, not per point's mates; the true centre. The idea: physics per group, not per point, as a rigid-body engine has it: each group's centre velocity and acceleration and its spin about the centre, carried forward in frames without an image, every point following through the rigid fit (FIGNet keeps objects rigid by fitting one turn and centre per object, "shape matching"; GNS's per-particle acceleration suits sand and fluids). A per-point velocity and acceleration head was the wrong form (runs.md). Needs the groups right. The true centre (the centre of mass, where gravity acts) from tracks alone: in free flight the one point of the piece that follows a parabola at g while the rest spin about it, a small least-squares fit over the piece's turns and shifts. Train only on what 3D tracks give; Kubric's object centres and poses only to measure it.
- Noisy-depth training with a random global depth-scale error, so gravity can correct the scale.
- Rigid proposal through the fit: not detached, a loss on the proposal's own positions, so the groups learn what the fit needs (grouping plan item 2).
- Groups in two cuts: objects first, from appearance (frame 0 is enough), then rigid parts by motion only within an object (a human into limbs); the physics (centre motion, spin) on each part. Motion never joins across objects, so a ball resting on the floor is never the floor's (today slow movers' wrong mates are 73% background, runs.md 13 Nov). Joined parts (limbs) are not independent bodies; on Kubric every object is one rigid part.
- SAM 3 masks as an appearance input for real video, trained on its masks of our clips (EntitySAM lost to our groups on Kubric).
- New points where new surface appears (the spawn rule), trained with many late points: given at test time to a model that rarely saw them, exact new points made the old ones worse (0.653 to 0.624) and a rigid fit from them gained nothing (runs.md, 12 Nov).
- A long run once the design settles is also the fix for turned points: TAPIP3D's matcher, the same frame-0 template, drifts 0.6% of the distance moved trained long, 7-8% at our budget (TAPIP3D trains on about 28x our samples, with augmentation).
- Whole-frame comparison, removed in a cleanup: needed eventually?
- More data: PointOdyssey or SynthVerse.
- The full scene-points design (docs/tasks/scene-points-2026-10-31.md).
- Spatial attention in every block or once at the end; virtual points; model size (dim 256, depth 6, 4 heads is what trained, never chosen).

## Ruled out

- Telling the model how far its mates sit from its rigid fit (`--fit-left`): no better hidden points (runs.md, 13 Nov).
- A pairwise motion check as an attention bias between points (how much each pair's distance changed): no better groups or fit at step 500 (runs.md, 13 Nov).
- A point's own template refreshed to its latest clear look at its guessed position (`--template latest`): standard moving 0.729 against 0.768 at step 500; the guess's error becomes the target (0.543 late, against 0.775 with the true position).
- More forward rounds before the backward sweeps (ffffbb): rounds 3 and 4 moved the error only 1.07 to 0.92 px (fyO, by round), and ffbb already gives each window 4 rounds; turning, 3-4 times the error, is the bigger target.
- DINOv3 (77%) and EntitySAM (74%) groups: worse than the model's own (89%) on Kubric.
- The matcher on the fixed 9 m ruler (fzR): Kubric lost 0.017 against the clip's own spread.
- Caching cross-attention keys and values: 9.4% slower, memory-bound (flow-matching design, 2026-09).
