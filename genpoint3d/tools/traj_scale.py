"""The constants that set the units the model sees."""

# One ruler for every clip, in metres per unit: positions are divided by it,
# never by anything a clip's own frames say, so nothing of a forecast frame
# leaks into the units, and gravity is one number everywhere. Kubric's typical
# scene distance (the median per clip is 8.9 m on our clips, 9.3 m on the old
# Kubric, 2.5-12 m across), so a typical scene keeps the sizes the model knows.
SCENE_SCALE = 9.0
# Inputs further than this from the camera (metres) are held at it: beyond
# Kubric's background (tracked points to 51 m), so it cuts only real video's sky.
RADIUS = 60.0

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
