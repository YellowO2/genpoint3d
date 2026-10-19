# 2026-10-20: why hidden points fail, and rigid buddies

## Velocity helps a little, and is not the reason

Fast setup (24 grid, CNN, 2,500 steps, accum 1), moving hidden points on val:

| Run | What | APD | Moving hidden |
|---|---|---|---|
| vzA | velocity output + velocity loss | 0.762 | 0.229 |
| vzB | vzA + velocity as an input | 0.753 | 0.229 |
| vzC | vzA + rigidity loss | 0.765 | 0.241 |

Earlier batches: the velocity loss is the part that helps (position 0.165 to
0.199 at step 1,000), the velocity output adds a little, acceleration as the
output (run41) and as a loss do not help, and a heavier velocity loss does not
either. run42_vel_48 is the full setup with vzA's choices.

## Where TAPIP3D's lead comes from

Moving hidden points, 100 val clips, split by whether the point is seen again
later in the clip. The baselines use the true positions.

| | run33 | TAPIP3D | keep last speed | straight line between sightings | rigid buddies |
|---|---|---|---|---|---|
| all hidden | 0.283 | 0.635 | 0.27 | 0.26 | 0.76 |
| seen again later | 0.430 | 0.771 | 0.39 | 0.52 | 0.88 |
| never seen again | 0.214 | 0.564 | 0.20 | 0.13 | 0.71 |

Rigid buddies: each hidden point carried from where it was last seen by the
rotation and shift that move the visible points of its own object (pairs whose
true distance never changes, among the scored points), `_buddy` in
`score/track_scores.py`.

1. TAPIP3D wins on points never seen again too, so seeing the future is not
   the main reason.
2. Knowing which visible points share a point's rigid object, and following
   them, is enough to beat TAPIP3D. That is the skill we lack.
3. On points seen again, we are below a straight line between the sightings,
   so a model that reads only the past also loses some there.

## The buddies module and two-way time do not help (fast setup)

Moving hidden, same fast setup as above, all with the rigidity loss:

| Run | What | Step 2000 | Step 2500 |
|---|---|---|---|
| bw0 | + visibility head fed back (reference for bw1, bw2) | 0.232 | 0.237 |
| bw1 | bw0 + `--buddies 1` | 0.237 | |
| bw2 | bw0 + `--buddies 1 --buddy-bias 1` | | |
| vzC | rigidity loss only | 0.236 | 0.241 |
| bw3 | vzC + `--causal 0` | 0.236 | 0.244 |

All within noise (about 0.005 between repeats).

## Why: our visible moving points are not precise enough

The rigid-buddy rule applied to each model's OWN predicted tracks (`_buddypred`:
true buddies and true visibility, predicted positions):

| Moving points, 100 val clips | run33 | TAPIP3D |
|---|---|---|
| buddy rule, true tracks | 0.761 | 0.754 |
| buddy rule, own tracks | 0.341 | 0.585 |
| actual hidden score | 0.283 | 0.635 |
| visible, APD | 0.755 | 0.926 |
| visible, within 1 px | 0.476 | 0.778 |

Rigidity only places a hidden point well when its visible buddies are tracked
precisely. TAPIP3D's are and ours are not: the hidden gap is mostly inherited
from the gap on visible moving points. A module that reads our own guesses
for buddies cannot do better than those guesses, which is why `--buddies`
changed nothing.

So the next question is why visible moving points are imprecise. Queued:

- vl10k_rigid: vzC for 10,000 steps instead of 2,500. If moving APD keeps
  climbing well past vzC, we are undertrained (CoTracker-family models train
  50k-200k steps; we train 6k, and our curves are still rising when the
  schedule ends).
- run33 scored in 2D, and with the true z or the true x swapped in
  (`fit_check` with SPACE=2d, AXIS=z, AXIS=x): whether the error is mostly
  depth or mostly position in the image.

## Built tonight

- `--buddies 1` (`model/Buddies.py`): from the model's own guess, points whose
  gap has kept its frame-0 length are buddies, weighted by how sure the model
  is it sees them; each point gets where its buddies' rotation and shift say it
  is, as an input. `--buddy-bias 1` also lets spatial attention listen to
  buddies, GNS-style, from the same pair numbers.
- `--rigid-loss W`: the gap between two points of one moving rigid object must
  match the truth in every frame.
