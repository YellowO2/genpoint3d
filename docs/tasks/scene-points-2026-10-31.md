# A persistent set of scene points: design

What this is for: tracking points on surfaces that turn as well as on ones
that slide (the spinning-ball test: sliding 0.4% of the distance moved,
spinning 3 times worse, rolling 5 times, 20 deg a frame lost), on the way to
one model that tracks, fills in what it cannot see and forecasts.

Why the current model fails at it: each point compares only the few cells
around its own guess, mostly against frame 0. Nothing compares whole frames
with each other, so a turning surface is never seen turning as a whole.

## Agreed so far

1. The internal picture. One set of 3D points for the scene, each one
   physical spot with a position in every frame, seen or not. Points on
   surfaces seen in the frames, and imagined points on surfaces not seen.

2. The input. Frame t-1 compared with frame t over the whole frame, with the
   CNN's features (4 px cells, as RAFT compares frames): something like a
   flow map. DINOv3 gives what things are (for grouping, imagined shape and
   physics), not the comparison.

3. Points over time.
   - Seen points are tied to pixels: a point starts at a visible cell, and
     its true track supervises it.
   - Imagined points: a fixed budget, trained by hiding frames on purpose:
     given frames 0-11, place the points that first appear in frames 12-23.
     No point is any particular true point, so a set loss (Chamfer distance)
     against the true hidden points.
   - The set carries over from window to window. A visible cell no visible
     point explains gets a new point; every point is refined the same way,
     seen or not, only with more evidence when seen; an imagined point whose
     surface comes into view is corrected by it.
   - Windows and single frames are the same step with a different number of
     new frames: the window size is a setting (16 now).
   - Budget: about k times one frame's worth of points, k around 4 (enough
     views to cover a convex object; more in clutter). One point a 16 px
     patch (24 x 24 a frame, about 2,300 a scene) and the CNN lookup for the
     exact spot.

4. Visibility. For every point and frame, whether it is seen: where to add
   points, which frame evidence a point trusts, and TAPVid-3D's AJ.

## Still to settle

- What a point is matched against. Frame 0's look only (now); then the latest
  frame it was clearly seen in, which looks more like it does now; then a
  template the model keeps and learns when to update (SAM2's memory of recent
  frames, MixFormer's template updates).
- The whole-frame comparison is in (fyY, `--grid-cmp`): frame t-1 against t
  at a 24 x 24 grid, once per clip, read by every point at its nearest cells.
  fyY2 tests it in place of the per-point frame-before comparison (40% faster).

- How points read the frame-to-frame comparison.
- How each piece's rigid motion is built from it, step by step (frame t's
  motion = the step from t-1 to t on top of t-1's), with frame 0 only to
  correct drift.
- Forecasting: continuing each piece's motion, contacts, more than one
  possible outcome.

## First check

Before the learned version: on the ball, the whole-frame comparison of
neighbouring frames (SEA-RAFT), every visible cell of the ball, the ball's
rotation fitted step by step, the points carried by it. No model. Near
sliding's accuracy means the core is right; drift shows what the learned
version must correct.
