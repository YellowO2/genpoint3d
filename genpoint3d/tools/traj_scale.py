"""The constants that set the units the model sees."""

# A typical clip's unit, in metres: Kubric's typical scene distance (the
# median per clip is 8.9 m on our clips, 9.3 m on the old Kubric), the sizes
# the model knows. Each clip's own unit scales it by the clip's frame-0
# spread against a typical one (MATCH_SPREAD; `ClipDataset._clip_unit`), from
# frame 0 alone so nothing of a forecast frame leaks into it: a 3 m scene and
# a street look the size the model knows, and tracking is the same at any scale.
SCENE_SCALE = 9.0
# A typical Kubric frame's spread of cell positions in model units (the median
# over clips is 0.37 on our clips, 0.33 on the old Kubric): each clip's own
# frame-0 spread is this in its unit, and the matcher brings it to 1.
MATCH_SPREAD = 0.35
# Gravity, which a clip's own unit does not keep one number: told to the
# model per clip, down, in the clip's unit per frame squared, over a typical
# clip's, so a typical clip's is 1.
GRAVITY = 9.81            # m / s^2
KUBRIC_FPS = 12           # every Kubric clip we train on (MOVi's 24 frames in 2 s)
GRAVITY_REF = GRAVITY / KUBRIC_FPS ** 2 / SCENE_SCALE
# Inputs further than this from the camera (metres) are held at it: beyond
# Kubric's background (tracked points to 51 m), so it cuts only real video's sky.
RADIUS = 60.0
# Points are tracked only within this distance of the camera at their query
# frame (metres): Kubric's sky is the inside of a dome 40 m out with a photo
# on it, a surface real video does not have, and nothing else is that far.
REACH = 30.0

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
