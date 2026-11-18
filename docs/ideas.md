# Ideas

## The design

1. The world is the memory: bodies and their points are stored once, at their place.
2. Each frame, physics predicts and seeing corrects that one stored world.
3. The network turns what it sees into the world (points, bodies, how stiff, each body's whole shape as a guess) and says the forces between bodies that touch.
4. The engine does what is exact: gravity, inertia, a force into motion, a stiff body keeping its shape, keeping what is out of view.
5. Any frame, past or future, is the stored world played, by each body's record of how it moved.

The network is offered these to learn to use; nothing is enforced on its answer.

## Later

- Rigid parts within an object (limbs).
- The network says a body's properties once and the engine works out the forces.
- Merging a returned surface's old point with its new one.
- Mass as a learned property.
- Confidence as CoTracker3's.
- SAM 3 masks and V-JEPA 2 as extra inputs.
- Noisy depth and real frame rates in training.
- More data: PointOdyssey, SynthVerse, soft bodies.
- A long run once the design settles.
