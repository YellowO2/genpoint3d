# Dense rigid motion: design

## Why

Tracking gets worse the further a point moves because points that move far
have usually turned (runs.md, drift by turn). The matcher compares a point's
3D neighbourhood with frame 0's, and once the surface has turned they no
longer line up. Comparing with the frame before as well (fyP) closed a fifth
of the gap; optical flow, chained, fitted per object or as an input (fyQ),
added nothing. The model never represents how a surface turned.

A first try (fyS) grouped the 256 query points into soft rigid pieces and
moved each point by its piece's fitted motion, through a gate that started
shut. The gate never opened (its weights stayed at about 0.02): the groups
were only trained through the gate, so they never became good enough to open
it. And 256 points, about 20 an object, is far sparser than what this idea
is built on.

## The end state

One model that watches the clip and infers, densely, the scene's structure and
how every part of it moves; the query points are read off that.

1. Dense state. A grid of cells over frame 0 (the CNN's map, at a stride),
   each tracked through every frame: its 3D position per frame, a group vector
   (which rigid piece it is on) and a confidence per frame. Later also its
   depth, predicted instead of given.
2. Joint refinement in rounds, as now: every cell compares its neighbourhood
   with frame 0's and the frame before's, and the blocks attend over time and
   between cells.
3. Rigid motion, from the dense state. After each round, for each cell and
   frame, the rotation and shift that best carry its group's frame-0 positions
   onto their current positions (RAFT-3D's Dense-SE3, but from frame 0 across
   the clip rather than between two frames). The rotation is not a separate
   output to be guessed: it is computed from how many cells of one surface
   moved together.
4. The rotation is used. Each cell's position is settled towards its group's
   rigid motion, and the matcher rotates frame 0's neighbourhood by it before
   comparing.
5. Query points ride on the dense state: a query sits among cells in frame 0
   and is carried by their rigid motion, plus a small correction of its own
   from the matcher.
6. Physics carries the dense state through hiding and into the forecast.

How the existing work does the dense part: DELTA (ICLR 2025, cloned here)
tracks a coarse grid of every 4th pixel of its model resolution with a
CoTracker-style transformer, using attention within local 6 x 6 windows of
cells plus a few global tokens so that thousands of cells stay affordable,
and an attention upsampler to full resolution. SpatialTracker learns a
soft "moves rigidly with" signal between points.

## What trains what

- Positions: the true tracks. The cache has 256 a clip; the raw Kubric data
  has 2048, so a denser cache gives far more supervised points.
- Groups: pairs of tracked points that keep their distance (`rigid_pairs`)
  should have close vectors, other pairs far ones: their own loss, so the
  groups learn whatever the settling does.
- The fit: a loss on the fitted positions themselves (RAFT-3D's loss on the
  motion its fit induces), so the fit learns even before it is trusted.
- Depth, later: the dense depth loss (`tools/depth.py`, built).

## Steps, each a piece of the end state

1. Dense cells with given depth. A grid of cells tracked with the query
   points (CoTracker's support grid, trained rather than only at test), the
   group and fit losses above, positions settled by the fit, queries carried
   by it. Warm-started from fyP; reference fyR (running).
2. The rotation-aware matcher.
3. Depth predicted, not given.
4. The dense state forecast.

## Open

- Grid size against cost: 24 x 24 = 576 cells a frame with all-pairs
  attention between cells costs about 4 times what 256 points do; DELTA's
  local-window attention is the way out if that is too slow.
- Cells have no true track of their own (only the 256 or 2048 tracked points
  do): they learn through the queries they carry and the group losses. A
  2048-point cache would let most cells be tracked points instead.
