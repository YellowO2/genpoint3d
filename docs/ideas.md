# Ideas

The one place for what is decided to come later, and what was ruled out. One
line each; a line leaves when it is built (runs.md then has it) or dropped.

## Later

- Camera roll in our renders, so finding down is real work: only if the down output's angle error is poor on rolled real video (ADT).
- Real video's frame rate in the gravity strength (fixed at Kubric's 12 fps now), and frame-rate variety in training.
- Two corrections: tracking in the clip's unit, physics in real units.
- Physics redesign (13 Nov): forecast frames stepped one frame at a time, one round, as GNS: each point's acceleration from its last two positions, gravity and the points near it (the model's own support points too: floor, walls), its position integrated; tracking keeps its windows. No object centre.
- Noisy-depth training with a random global depth-scale error, so gravity can correct the scale.
- Groups trained through the fit's turn too, not only its centres (R is detached now): RAFT-3D trains its embeddings through its whole differentiable rigid solve.
- Trust as CoTracker3's and TAPIP3D's confidence: the chance the point's own guess is within a threshold of the truth, not whether it beats the object's place (a target that moves as the object's place improves).
- Groups in two cuts: objects first, from appearance (frame 0 is enough), then rigid parts by motion only within an object (a human into limbs); the physics (centre motion, spin) on each part. Motion never joins across objects, so a ball resting on the floor is never the floor's (today slow movers' wrong mates are 73% background, runs.md 13 Nov). Joined parts (limbs) are not independent bodies; on Kubric every object is one rigid part.
- SAM 3 masks as an appearance input for real video, trained on its masks of our clips (EntitySAM lost to our groups on Kubric).
- Spawning at test time, without truth: a first pass tracks the given points; where a frame shows surface that came into view with no tracked point near it (by the predicted places and visibility), a new point starts at that pixel, its depth from the depth map; a second pass tracks both, only the given points scored. Training spawns from the cache's own tracks (`--spawn`, long_physics_spawn); validation and the tests spawn nothing yet. These are also the forecast's neighbours (floor, walls) where the given points have none.
- New points, given at test time to a model that rarely trained with them, made the old ones worse (0.653 to 0.624, 12 Nov); tracked in their own pass, they more than double hidden points whose old mates are all hidden (0.138 to 0.312, runs.md 13 Nov).
- A long run once the design settles is also the fix for turned points: TAPIP3D's matcher, the same frame-0 template, drifts 0.6% of the distance moved trained long, 7-8% at our budget (TAPIP3D trains on about 28x our samples, with augmentation).
- Whole-frame comparison, removed in a cleanup: needed eventually?
- More data: PointOdyssey or SynthVerse.
- The full scene-points design (docs/tasks/scene-points-2026-10-31.md).
- Spatial attention in every block or once at the end; virtual points; model size (dim 256, depth 6, 4 heads is what trained, never chosen).

## Ruled out

- Physics carried by an object centre (v1, `--physics`, object_paths): an object's true centre cannot be found from one side of it, and floors and walls have none.
- Telling the model how far its mates sit from its rigid fit (`--fit-left`): no better hidden points (runs.md, 13 Nov).
- A pairwise motion check as an attention bias between points (how much each pair's distance changed): no better groups or fit at step 500 (runs.md, 13 Nov).
- A point's own template refreshed to its latest clear look at its guessed position (`--template latest`): standard moving 0.729 against 0.768 at step 500; the guess's error becomes the target (0.543 late, against 0.775 with the true position).
- More forward rounds before the backward sweeps (ffffbb): rounds 3 and 4 moved the error only 1.07 to 0.92 px (fyO, by round), and ffbb already gives each window 4 rounds; turning, 3-4 times the error, is the bigger target.
- DINOv3 (77%) and EntitySAM (74%) groups: worse than the model's own (89%) on Kubric.
- The matcher on the fixed 9 m ruler (fzR): Kubric lost 0.017 against the clip's own spread.
- Caching cross-attention keys and values: 9.4% slower, memory-bound (flow-matching design, 2026-09).
