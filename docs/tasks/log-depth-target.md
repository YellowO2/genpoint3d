# Log-depth target

**Why this is next.** Substituting the true depth channel into run7's
prediction recovers 0.149 of the 0.202 gap between our 3D score (0.453) and our
2D score (0.655). Roughly three quarters of what separates them is depth error.
We regress raw metric z; every reference reparameterises. DELTA predicts
`log(d_t / d_1)` and ablated it against inverse and Euclidean depth.

It needs no cache rebuild, which is why it outranks resolution and width.

## The design problem to solve first

`ClipDataset` divides the whole target by one scalar, `traj_scale`:

    traj = traj / self.traj_scale        # scripts/train.py

and `to_metres` inverts that with one scalar per clip:

    scene = pts * ts + b["norm_offset"]  # scripts/train.py:312

A log ratio and a lateral displacement are different quantities with different
spreads, so **one scalar cannot bring both to unit variance**. Flow matching
mixes the target with `x0 ~ N(0, I)`; if a channel sits far below unit variance
the interpolant is nearly pure noise in that channel and the model learns to
output `-x0` for it. That is exactly the failure `check_transform.py`'s scale
sanity test exists to catch.

So `traj_scale` has to become per-channel, `(lateral, lateral, log_depth)`, and
that propagates into `NormStats` and the `norm_traj_scale` the batch carries.

## Proposed target

Per point, with `z0` its own frame-0 depth in scene-normalised units:

    x, y  ->  (x_t - x_0) / s_xy              as --target displacement already does
    z     ->  log(z_t / z0) / s_z

Inverse, in `to_metres`:

    xy_scene = pred_xy * s_xy + offset_xy
    z_scene  = z0 * exp(pred_z * s_z)
    metres   = scene * norm_scale + norm_mean

`z0` must be added to the batch, as `norm_z0` of shape `(B, 1, N, 1)`, because
the inverse is no longer affine and `norm_offset` cannot carry it.

Clamp `z` away from zero before the log. Kubric depth is positive, but a
normalised z can cross zero if `norm_mean` is subtracted first -- check which
side of the mean subtraction the log has to sit on. **This is the part most
likely to be wrong, and it is silent: a NaN would show up only as a dead loss.**

## Order of work

1. Extend `scripts/calibrate_traj_scale.py` with `--target logdepth`, printing
   the two constants separately. It reads the cache and is minutes on gdev.
2. Put them in `transform.py` as `TRAJ_SCALE_LOGDEPTH = (s_xy, s_z)`. Leave it
   `None` until measured, and have `train.py` refuse to start on `None` -- a
   wrong scale trains to a plausible-looking loss and a useless model.
3. Make `traj_scale` per-channel through `ClipDataset`, `NormStats` and
   `to_metres`.
4. Round-trip test in `check_transform.py`: encode then decode must return the
   original metres to float tolerance, including points at the depth extremes.
   Assert no NaN for z near zero.
5. Only then train. Match run7 otherwise so the target is the single change.

## Status

Not started. The measurement that motivates it is in `runs/README.md`
("Most of the 3D error is depth error"), and `--oracle-z` in `scripts/evaluate.py`
reproduces it.
