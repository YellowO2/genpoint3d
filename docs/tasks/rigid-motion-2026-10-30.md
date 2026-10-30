# Points that carry a rigid motion (after RAFT-3D)

## Why

Tracking gets worse the further a point moves because points that move far
have usually turned (runs.md, drift by turn). The matcher compares a point's
3D neighbourhood with frame 0's, and once the surface has turned the two no
longer line up. Comparing with the frame before as well (fyP) closed a fifth
of the gap; optical flow, chained, fitted per object or given as an input
(fyQ), added nothing. The model never represents how a surface turned, so it
cannot use it.

## What RAFT-3D does (Teed and Deng, CVPR 2021)

Scene flow from two RGB-D frames, built on RAFT.

1. State: every pixel carries a rigid motion, a rotation and a translation,
   not a displacement.
2. Each iteration, the update network reads the correlation at where that
   motion sends the pixel, and outputs three things: a revision (where the
   pixel should really go, and its inverse depth), a confidence for it, and a
   rigid-motion embedding, a vector per pixel.
3. Dense-SE3, a differentiable layer: for each pixel i, one Gauss-Newton step
   on the rigid motion that best explains the revised correspondences of the
   pixels j around it, each weighted by its confidence and by the affinity
   a_ij = 2 sigmoid(-|v_i - v_j|^2) of the two embeddings. Pixels with similar
   embeddings act as one rigid piece; nothing says what an object is.
4. Supervised only by optical flow and depth change. The embeddings get no
   labels: they are learned through the fit.

## The same in our model

| RAFT-3D | here |
|---|---|
| a rigid motion per pixel, frame 1 to 2 | a rigid motion per query point, frame 0 to t: which way its patch of surface faces now, and where it is |
| GRU update | the DiT blocks, each round |
| revision + confidence | the correction the model gives now, plus a confidence per point and frame |
| rigid-motion embedding | a vector per point, from its tokens |
| Dense-SE3 step | for each point i and frame t, the weighted rigid fit (Kabsch, closed form) that best carries the frame-0 positions of the points j onto their revised positions at t, weights confidence x affinity; point i's position is its own frame-0 position carried by it |
| loss on flow and depth change | the loss we have, on positions; no rotation or group labels |

What this gives the points:

- A point is placed by the motion of every point that moves with it, so a spot
  that has turned away is placed by the rest of its surface.
- A hidden point is placed the same way, by its group's visible points.
- Each point has a rotation, which the matcher can then use: rotate frame 0's
  neighbourhood offsets by it before comparing, so a turned surface lines up
  again. RAFT-3D does not need this (its frames are neighbours); we compare
  against frame 0.

## Runs

All warm-started from fyP (`--init`), with the whole-frame match off
(`--correlate 0`), about 1,500 steps, judged by the turn table and moving APD
by frame.

1. Control: fyP so, nothing else.
2. The rigid fit: embeddings, confidences and the fit layer; positions come
   from the fit.
3. Only if 2 helps: the rotation-aware matcher.

Open: 128 points per clip may be too few for good groups (about 10 per
object). If 2 is held back by that, add a grid of helper points.
