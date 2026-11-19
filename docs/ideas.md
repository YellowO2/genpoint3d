# Ideas

## The design

A model that imagines the world with a physics engine as its tool: it keeps the world stored as an engine's, the engine plays it, and the model corrects it as it sees more. Whatever a physics engine does, the engine here does.

1. The world is stored once: bodies, each with its points, the bars between them, mass, stiffness and a record of how it moved. A bar joins two nearby points of a body: three make a wall, so the bars are the body's whole shape, and each is the engine's constraint holding the two points apart as they are (stiff bars a stiff body, bars that give a soft one).
2. Each frame, the engine predicts and the model corrects the stored world from what it sees: where things are, and what each body is.
3. The model imagines each whole body, its unseen side too, and says what it is: fixed or free, how heavy, how stiff. Of a living thing it also says its own push.
4. The engine does the rest: gravity, inertia, finding where bodies touch, the push that stops them there, a stiff body keeping its shape, keeping what is out of view.
5. Any frame, past or future, is the stored world played.

The model is offered these to learn to use; nothing is enforced on its answer.

Not built yet: the bars and the unseen side (only a body's centre of gravity is); until they are, the model says where two bodies touch and how far they hold and grip there.

## Later

- Asleep bodies skipped as a game engine skips them: one at rest or fixed costs nothing; one moving unseen keeps being stepped, without the image.
- The model's range, as a game's simulation distance: a body farther than it from the camera is neither stepped nor looked at; it stays stored as it was until the camera comes near again.
- Points as the vertices of a body's shape, as a game's collision mesh: one added only where the seen surface is not already explained by the body's points, so a flat floor needs few.
- Rigid parts within an object (limbs), as a robot simulator has them: a joint is two parts always held together at one place, and the model says how hard each joint turns (a turning push, not yet in the engine).
- The stored world rendered back into the video and compared with it, as CycleGAN's cycle: the model then learns by itself how many vertices and what physics the scene needs to come out the same. After the point loss is right.
- A body's look (a texture on its walls): a point's one colour is too little to tell the scene from the stored world drawn (outputs/points_view.html).
- A ready shape model as a second tool beside the engine: it proposes a body's whole shape from the points and views seen so far, the model corrects it. Best fit found: ShapeR (takes sparse 3D points and several posed views, gives the shape at true size and place); single-picture reference: TRELLIS (best on Kubric's own objects in an outside benchmark), SAM 3D Objects (32 GB). Its guess can be scored against Kubric's true shapes first. Closest whole design to ours: ReconPhys (video to points with springs, mass, stiffness and friction, taught by rendering back).
- Confidence as CoTracker3's.
- SAM 3 masks and V-JEPA 2 as extra inputs.
- Noisy depth and real frame rates in training.
- More data: PointOdyssey, SynthVerse, soft bodies.
- A long run once the design settles.
