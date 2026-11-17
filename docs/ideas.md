# Ideas

One line each. Now: decided, being built. Later: not yet. Ruled out: tried, with the result in runs.md.

## Now

- The model is a learned physics engine: the engine's parts are ours to reference (rigid bodies, contacts, joints, motors, a step of gravity and inertia); what an engine calculates from known shapes and masses, the model says. Mechanisms for it to learn to use, not rules enforced on its answer.
- Rigid place: a point is pulled toward the place that keeps its group's shape, as strongly as the model says the group is stiff (not enforced: a soft body gets no pull).
- Contact force: where two rigid bodies touch, the model says the force between them, a push or a pull (one vector at the touching point, equal and opposite on the two); fixed physics turns it into each body's change of speed and spin about its centre of gravity, and its points follow. Inertia and gravity are given. Two things colliding and two limbs at a joint are the same case.

## Later

- Rigid parts within an object, split by motion (limbs); on Kubric every object is one part.
- Groups trained through the rigid alignment, as RAFT-3D.
- Confidence as CoTracker3's: the chance the guess is within a threshold of the truth.
- Forecast stepped one frame at a time, as GNS.
- The turn of a tumbling object in free flight, learned.
- Merging a returned surface's old point with its new one.
- Mass as a learned object property: our generator must set density first.
- V-JEPA 2 beside DINOv3 for object properties, if DINOv3 cannot predict them.
- SAM 3 masks as an appearance input on real video.
- Noisy-depth training with a random depth-scale error, so gravity can correct the scale.
- Real video's frame rate in the gravity strength; frame-rate variety in training.
- Camera roll in our renders, only if finding down is poor on rolled real video.
- More data: PointOdyssey, SynthVerse, soft bodies.
- Model size and where spatial attention goes: never chosen.
- A long run once the design settles.

## Ruled out

- Physics carried by an object centre found by the model: a centre cannot be found from one side, and floors have none.
- Blending the guess with the rigid place by a learned trust (`--fit-blend`): broke training.
- Telling the model how far its mates sit from their rigid place (`--fit-left`): no gain.
- A pairwise motion check as an attention bias: no gain.
- A point's template refreshed to its latest look: worse.
- More forward rounds (ffffbb): no gain.
- DINOv3 or EntitySAM groups: worse than the model's own.
- The matcher on a fixed 9 m ruler: worse.
- Caching cross-attention keys and values: slower.
