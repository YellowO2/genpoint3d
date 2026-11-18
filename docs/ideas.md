# Ideas

## The design

A converter from the real world to a physics engine: the model refines the engine's world as it sees more.

1. The world is stored once: bodies, each with its points, whole shape, mass, stiffness and a record of how it moved.
2. Each frame, the engine predicts and the model corrects the stored world from what it sees: where things are, and what each body is (shape, mass, stiffness).
3. The model also says what the engine cannot know of the bodies: which are fixed, how heavy, where two touch and how far they stick there; and a push of its own where the engine's is not all (a bounce, a living thing).
4. The engine does what is exact: gravity, inertia, the push that stops two bodies where they touch, a push into motion, a stiff body keeping its shape, keeping what is out of view.
5. Any frame, past or future, is the stored world played.

The model is offered these to learn to use; nothing is enforced on its answer.

## Later

- Points as the vertices of a body's shape, as a game's collision mesh: one added only where the seen surface is not already explained by the body's points, so a flat floor needs few.
- Rigid parts within an object (limbs).
- The model says a body's properties once and the engine works out the forces.
- Confidence as CoTracker3's.
- SAM 3 masks and V-JEPA 2 as extra inputs.
- Noisy depth and real frame rates in training.
- More data: PointOdyssey, SynthVerse, soft bodies.
- A long run once the design settles.
