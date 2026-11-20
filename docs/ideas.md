# Ideas

## The design

A model that imagines the world with a physics engine as its tool: it keeps the world stored as a game's, the engine plays it, and the model corrects it as it sees more. Whatever a physics engine does, the engine here does. The test of what is stored: the video could be drawn again from it.

What is stored:

- A body is a thing. It has one description of what it is (what kind of thing, its look, fixed or free, how heavy, how stiff; of a living thing also its own push), its points, the bars between them, and where it has been in every frame.
- A point is a place on a body, not a thing. It has its body, where on the body it is, and whether it is seen now. A point a frame gave also keeps the look of its spot, for one use: finding that spot again. A body's points are as a game's mesh: the few its shape needs.
- A bar joins two nearby points of a body. Three joined points are a wall, so the bars are the body's whole shape, and each is the engine's constraint holding the two points apart as they are (stiff bars a stiff body, bars that give a soft one).
- The camera, and the light.

How it runs:

1. Each frame, the engine plays the stored world one frame on, and the model corrects it from what it sees.
2. A body is imagined whole the moment it is first seen: its description, and from the description all its points, the unseen side too. All of it is a guess corrected every frame: the description as more is seen, a point when what is seen says it is elsewhere. A point is added only where nothing was imagined.
3. The engine does the rest: gravity, inertia, finding where bodies touch, the push that stops them there, a stiff body keeping its shape, keeping what is out of view.
4. Any frame, past or future, is the stored world played. A point someone asks to track is a place on a wall of its body.

The model is offered these to learn to use; nothing is enforced on its answer.

Not built yet: the bars, the model choosing its own points, an asked point read off its body's walls, the light. Until the bars are, the model says where two bodies touch and how far they hold and grip there.

## Later

- Asleep bodies skipped as a game engine skips them: one at rest or fixed costs nothing; one moving unseen keeps being stepped, without the image.
- The model's range, as a game's simulation distance: a body farther than it from the camera is neither stepped nor looked at; it stays stored as it was until the camera comes near again.
- Rigid parts within an object (limbs), as a robot simulator has them: a joint is two parts always held together at one place, and the model says how hard each joint turns (a turning push, not yet in the engine).
- The stored world rendered back into the video and compared with it, as CycleGAN's cycle: the model then learns by itself how many points, what look and what physics the scene needs to come out the same. Needs the light, and a body's look as a texture on its walls.
- A ready shape model as a hint for a body's unseen side, set aside: TripoSR knows a little (triposr_shape_prior). Others found: ShapeR, TRELLIS, SAM 3D Objects; closest whole design to ours, ReconPhys.
- Speed, when we need it: search the current research on speeding up first. One seen: Spark3R (arXiv 2605.06270), attention over many frames up to 28 times faster with no retraining, by merging alike tokens that ask and dropping tokens that are asked.
- Confidence as CoTracker3's.
- SAM 3 masks and V-JEPA 2 as extra inputs.
- Noisy depth and real frame rates in training.
- More data: PointOdyssey, SynthVerse, soft bodies.
- A long run once the design settles.
