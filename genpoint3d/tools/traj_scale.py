"""The constants that bring each training target to unit variance."""

# Dataset-level constant: the spread of scene-normalised trajectories.
# Scene normalisation makes *geometry* O(1), but leaves trajectories at ~0.26 --
# and flow matching mixes them with `x0 ~ N(0, I)`, so the target must be near
# unit variance or the interpolant is mostly noise. One global number, not
# per-clip, so no sample's own motion leaks into its normalisation (and so a
# fast clip stays genuinely faster than a slow one). Recalibrate with
# `scripts/calibrate_motion_scale.py` whenever the training set changes.
TRAJ_SCALE = 0.8344  # 200 Kubric clips, 3.2M visible coords, --norm-mode median

# The displacement target is 8.4x smaller: measured on the same 200 clips, only
# 14% of an absolute trajectory's magnitude is motion -- the rest is static
# scene layout. Reusing the absolute constant would leave the target at std
# ~0.12 against unit-variance noise, which is the `-x0` failure recorded in
# docs/misc.md.
TRAJ_SCALE_DISP = 0.0992
