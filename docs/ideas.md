# Ideas

## The design

A converter from the real world to a physics engine: the model refines the engine's world as it sees more. Whatever a physics engine does, the engine here does.

1. The world is stored once: bodies, each with its points, whole shape, mass, stiffness and a record of how it moved.
2. Each frame, the engine predicts and the model corrects the stored world from what it sees: where things are, and what each body is.
3. The model imagines each whole body, its unseen side too, and says what it is: fixed or free, how heavy, how stiff. Of a living thing it also says its own push.
4. The engine does the rest: gravity, inertia, finding where bodies touch, the push that stops them there, a stiff body keeping its shape, keeping what is out of view.
5. Any frame, past or future, is the stored world played.

The model is offered these to learn to use; nothing is enforced on its answer.

Not built yet: the whole shape (only its centre of gravity is); until it is, the model says where two bodies touch and how far they stick.

## Later

- Asleep bodies skipped as a game engine skips them: one at rest or fixed costs nothing; one moving unseen keeps being stepped, without the image.
- Points as the vertices of a body's shape, as a game's collision mesh: one added only where the seen surface is not already explained by the body's points, so a flat floor needs few.
- Rigid parts within an object (limbs).
- Confidence as CoTracker3's.
- SAM 3 masks and V-JEPA 2 as extra inputs.
- Noisy depth and real frame rates in training.
- More data: PointOdyssey, SynthVerse, soft bodies.
- A long run once the design settles.
