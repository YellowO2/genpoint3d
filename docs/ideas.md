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
