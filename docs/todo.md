# To do

The steps in order. The design is in docs/ideas.md, the runs in docs/tasks/runs.md.

| # | Step | State |
|---|---|---|
| 1 | Build the new design: stored world, engine, frame-by-frame model | Done |
| 2 | Check it runs and learns on a short run | Done: world_smoke |
| 3 | Add points as the clip goes at test | Done |
| 4 | Points as vertices: fewer points on flat things | Rule built and counted (vertices_rule_counts: 562 points a clip, not 1,361); not yet in the test-time adding |
| 5 | The whole shape: the model imagines each body's unseen side, so the stored world drawn from any side can be told (outputs/points_view.html shows one-sided shells today) | Here: before the retrain, the user's say |
| 6 | Speed: skip the matcher for asleep and out-of-frame points | After 5 |
| 7 | One proper retrain | Waits for the user's say |
| 8 | Delete the old tracker's code | After the retrain shows the new one works |
| 9 | A real physics engine in place of the stand-in | After the whole shape exists |
