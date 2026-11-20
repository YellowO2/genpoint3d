# To do

The steps in order. The design is in docs/ideas.md, the runs in docs/tasks/runs.md.

| # | Step | State |
|---|---|---|
| 1 | Build the new design: stored world, engine, frame-by-frame model | Done |
| 2 | Check it runs and learns on a short run | Done: world_smoke |
| 3 | Add points as the clip goes at test | Done |
| 4 | Points as vertices: fewer points on flat things | Rule built and counted (vertices_rule_counts: 562 points a clip, not 1,361); not yet in the test-time adding |
| 5 | The whole body: every body stored whole from first sight, and corrected as more is seen | Here. Built and tested on made-up clips, the short run next: each body has one description, kept and put right every frame, from which its freedom, mass, size, colour and the whole of its points (the unseen side too, up to 64, taught by the answer mesh) are read. Not yet: the model choosing its own points in place of the hand-written adding rule, the bars, an asked point read off the mesh. Next, agreed: one state loss, the stored world against the true one (bodies matched to objects, the floor a fixed flat body, field by field on one scale), in place of the separate parts; the tracked-point loss and the one-step engine check stay, the long forecast loss goes and forecast frames are the engine's alone (measured once with and without). Before this: a body stored three things and only the first was built: its points (today a point's place on its body is fixed once given, never corrected), the bars between them (none in the code), and what it is (no description of a body is stored; heavy, stiff and free are said anew each frame). The idea, agreed: at first sight the model imagines the body's whole set of points, the unseen side too, with bars; each frame it moves any of them to fit what it now sees, and adds one only where something was not imagined at all. The answer to teach it is ready: each object's low-poly shape, all sides, 8 cells across, with bars (outputs/world_view.html; 5,898 of 5,907 clips named, 924 models on NSCC, shapes on our tracks within 1.1 cm). A ready shape model as a hint is set aside (triposr_shape_prior: it knows a little). Next: the flow written out and agreed, then the build |
| 6 | Speed: skip the matcher for asleep and out-of-frame points | After 5 |
| 7 | One proper retrain | Waits for the user's say |
| 8 | Delete the old tracker's code | After the retrain shows the new one works |
| 9 | A real physics engine in place of the stand-in | After the whole shape exists |
