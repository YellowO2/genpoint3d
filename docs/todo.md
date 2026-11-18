# To do

The steps in order. The design is in docs/ideas.md, the runs in docs/tasks/runs.md.

| # | Step | State |
|---|---|---|
| 1 | Build the new design: stored world, engine, frame-by-frame model | Done |
| 2 | Check it runs and learns on a short run | Done: world_smoke |
| 3 | Add points as the clip goes at test | Done |
| 4 | Points as vertices: fewer points on flat things | Rule built and counted (vertices_rule_counts: 562 points a clip, not 1,361); not yet in the test-time adding |
| 5 | The whole shape: bars between a body's points (walls for drawing, constraints for the engine), and the model imagining the unseen side | Here. The target is seen and agreed: each object's low-poly shape, all sides, 8 cells across (outputs/world_view.html, target_detail in runs.md). Every clip's objects are named (5,898 of 5,907 clips); all 924 models are whole on NSCC (~/scratch/gso), and the shapes sit on our tracks (world_shapes_against_tracks: median 1.1 cm). Next: how the model says the unseen points and bars, the user's say |
| 6 | Speed: skip the matcher for asleep and out-of-frame points | After 5 |
| 7 | One proper retrain | Waits for the user's say |
| 8 | Delete the old tracker's code | After the retrain shows the new one works |
| 9 | A real physics engine in place of the stand-in | After the whole shape exists |
