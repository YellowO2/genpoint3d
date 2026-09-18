# Ideas, open questions, deliberate omissions

What the diagram cannot show: things not built, things undecided, and threads
worth returning to. The diagram in `map.html` covers what exists.

## Not built yet

From the paper:

- Visibility head. `Linear(D -> 1)` + BCE, predicting occluded or off-screen
  per frame. `model.py` has only `Linear(dim, 3)` for position. AJ and OA both
  score occlusion, so neither is reportable until this exists.
- Autoregressive sliding window. Generate W frames, slide by stride S, feed the
  previous window back as a causal prefix. How the paper handles video longer
  than one window. We do all 24 frames at once, which only works because Kubric
  clips are exactly 24 frames.
- Diffusion forcing. Independent noise level per frame. `flow.py` accepts
  `per_frame_k` already, it is just off. Needed for stable autoregressive
  rollout.
- Point-count factorization. Reshape N points into random factor pairs each
  step so the model handles any query count.

Ours, not in the paper:

- dt conditioning. The model sees frame index, never elapsed time, so frame
  rate is baked into the learnt motion prior. Feeding dt alongside the noise
  level would make it frame-rate agnostic. None of the papers we read do this.

## Open questions

- Spatial attention per block (our paper's choice, what we do) versus once at
  the end (genpt, cheaper). Ablate later.
- Virtual points, CoTracker3's O(N^2) saver. Currently off.
- Config is unsettled: dim 256 / depth 6 / 4 heads is what trained, not what
  was chosen. TAPIP3D trains on ~28x our samples, so size may not be the lever.
- Normalisation statistic. `median` distance to camera is the default;
  `mean` and `centroid_max` are cached too, so all three are one flag apart and
  nothing has compared them.
- `TRAJ_SCALE` is 0.8344, measured on 200 clips. Cheap to redo on all 3493 now
  that the cache stores metres.

## Generalized spatiotemporal conditioning mask

Instead of Gen-points' whole-frame cutoff (visual conditioning for `t < T_C`,
null after), train with a superset scheme: sometimes whole frames masked,
sometimes only parts of a frame (spatial / random spatiotemporal, MAE-style).

Why it could be interesting:

- One model then covers tracking, forecasting, and "keep tracking through a
  visual dropout or partial occlusion of the conditioning".
- In 3D specifically, partial 3D occlusion conditioning is not well explored.
- Whole-frame masking stays a subset of the training distribution, so the
  standard forecasting benchmark is still usable at eval — contrary to an
  earlier worry, this is not lost.

Costs and open questions:

- Metric risk: steps spent on the partial-mask regime that benchmarks do not
  reward. Could weaken whole-frame numbers, or help as regularization.
  Empirical, unknown until run.
- No standard benchmark for partial conditioning dropout, so demonstrating it
  would need a custom eval.
- Spatial masks instead of a per-frame boolean; more masking hyperparameters.

Logged 2026-09-10. Revisit once the whole-frame model produces sensible
tracking and forecasting.

## Metric units vs per-clip units — does losing the gravity prior hurt?

Everything is normalised by a per-clip scale, then by the global `TRAJ_SCALE`,
so the model works in units of "fraction of this scene", not metres.

Why it is that way: metric scale is not recoverable from images. A photo of a
real kitchen and a photo of a dollhouse kitchen are identical; nothing in the
pixels says which. Per-clip normalisation makes the task scale-invariant, and
the same physical motion stays self-consistent because the object shrinks along
with the scene — a ball moving an eighth of its own diameter reads the same in
a small room and on a street. DUSt3R normalises by average distance to the
origin for the same reason, and 4RC inherits it.

The cost: physics constants stop being constant. Gravity is 9.8 m/s²
everywhere, but in per-clip units a falling ball accelerates at 4.9 in a 2 m
room and 0.49 on a 20 m street. The model cannot learn one universal "things
fall like this" — it must infer the scene's scale from the geometry input and
adapt. For a forecasting model, where falling is a large share of the motion,
that is a real loss.

The ablation: Kubric gives ground-truth metres, so we can train both ways on
identical data — per-clip units versus one global metric scale — and compare
forecasting error. This is the only place we will ever be able to measure it;
on real video the metric arm is impossible.

If metric wins by a lot, the gravity prior matters and it is worth keeping
without true metric scale. Options then: predict scene scale as an auxiliary
output; feed a scale token; or normalise by something physically anchored
(estimated camera height) rather than by scene extent.

Logged 2026-09-11, from noticing that the same ball in two differently-sized
scenes gets different target numbers. Revisit once there is a forecasting
number to compare against.
