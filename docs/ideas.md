# Future directions / ideas to revisit

Parking lot for things we deliberately deferred. Not for v1.

## Generalized spatiotemporal conditioning mask
**Idea:** instead of Gen-points' whole-frame cutoff (visual conditioning present
for `t < T_C`, null for `t >= T_C`), train with a *superset* masking scheme:
sometimes whole frames masked, sometimes only parts of a frame (spatial /
random spatiotemporal, MAE-style).

**Why it could be interesting:**
- One model then covers: tracking, forecasting, AND "keep tracking through a
  visual dropout / partial occlusion of the conditioning".
- In 3D specifically, "partial 3D occlusion conditioning" is not well explored.
- Whole-frame masking stays a subset of the training distribution, so the
  standard forecasting benchmark is still usable at eval time (just apply the
  whole-frame mask) — contrary to an earlier worry, this is NOT lost.

**Costs / open questions:**
- Metric risk: training steps spent on the partial-mask regime that the
  benchmarks don't reward — could slightly weaken whole-frame (forecasting)
  numbers, or could help as regularization. Empirical, unknown until run.
- No standard benchmark for the "partial conditioning dropout" capability, so
  demonstrating it works would need a custom eval.
- Extra complexity: spatial masks instead of a per-frame boolean; more masking
  hyperparameters.

**Status:** logged 2026-09-10. Revisit after the v1 unified model (whole-frame
cutoff only) is training and producing sensible tracking + forecasting.

## Metric units vs per-clip units — does losing the gravity prior hurt?

**The situation:** `transform.py` normalises everything (pointmap, anchor,
trajectory) by a **per-clip** scale, then applies the global `TRAJ_SCALE`. So
the model works in units of "fraction of this scene", not metres.

**Why it is that way:** metric scale is not recoverable from images. A photo of
a real kitchen and a photo of a dollhouse kitchen are identical; nothing in the
pixels says which. Per-clip normalisation makes the task scale-invariant, and
the same physical motion stays self-consistent because the object shrinks along
with the scene — a ball moving an eighth of its own diameter reads the same in
a small room and on a street. MotionForesight does the same
(`_compute_pj_norm`, per clip).

**The cost:** physics constants stop being constant. Gravity is 9.8 m/s²
everywhere, but in per-clip units a falling ball accelerates at 4.9 in a 2 m
room and 0.49 on a 20 m street. The model cannot learn one universal "things
fall like this" — it must first infer the scene's scale from the geometry
input, then adapt. For a *forecasting* model, where falling is a large share of
the motion, that is a real loss.

**The ablation:** Kubric gives ground-truth metres, so we can train both ways on
identical data — per-clip units vs one global metric scale — and compare
forecasting error. This is the only place we will ever be able to measure it;
on real video the metric arm is impossible.

**If metric wins by a lot,** the gravity prior matters and it is worth finding a
way to keep it without true metric scale. Options to think about then: predict
scene scale as an auxiliary output; feed a scale token; or normalise by
something physically anchored (e.g. estimated camera height) rather than by
scene extent.

**Status:** logged 2026-09-11, from the user noticing that the same ball in two
differently-sized scenes gets different target numbers. Revisit once step 2
trains and there is a forecasting number to compare against.
