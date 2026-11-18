# To do

The steps in order. The design is in docs/ideas.md, the runs in docs/tasks/runs.md.

| # | Step | State |
|---|---|---|
| 1 | Build the new design: stored world, engine, frame-by-frame model | Done |
| 2 | Check it runs and learns on a short run | Done: world_smoke |
| 3 | Add points as the clip goes at test | Done |
| 4 | Speed: skip the matcher for asleep and out-of-frame points | Here: counting how many such points a clip has |
| 5 | Points as vertices: fewer points on flat things | Not started |
| 6 | One proper retrain | Waits for the user's say |
| 7 | Delete the old tracker's code | After the retrain shows the new one works |
| 8 | Whole-shape guess and a real physics engine | Open: last, or before the retrain |
