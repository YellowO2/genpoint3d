"""Build the model a run's settings describe."""

from genpoint3d.tools.traj_scale import TRAJ_SCALE, TRAJ_SCALE_DISP
from genpoint3d.train.model.PointDiT import PointDiT


def build_model(settings, feat_dim: int | None, has_feats: bool) -> PointDiT:
    """The model a run's settings describe: the run's own, or a checkpoint's
    (`Settings(**ckpt["args"])`) in evaluate.py and visualize_pred.py.

    A setting missing from a checkpoint means it was trained before that flag
    existed, and the default given here is how every model was built until
    then -- a checkpoint has to be evaluated the way it was trained.
    """
    a = lambda name, default: getattr(settings, name, default)
    displacement = a("target", "absolute") == "displacement"
    return PointDiT(dim=settings.dim, depth=settings.depth, num_heads=settings.heads,
                    # --dino 0 is a model with no DINOv3 input of any kind.
                    cross_attn=has_feats and bool(a("dino", 1)),
                    feat_dim=feat_dim,
                    adapter_depth=a("adapter_depth", 1),
                    upsample=a("upsample", 1),
                    traj_scale=TRAJ_SCALE_DISP if displacement else TRAJ_SCALE,
                    displacement=displacement,
                    # Absent means trained before the lookup was fixed.
                    legacy_lookup=bool(a("legacy_lookup", 1)),
                    time_norm=bool(a("time_norm", 0)),
                    # Absent means a checkpoint from before the prior existed.
                    locality=bool(a("locality", 0)),
                    correlate=bool(a("correlate", 0)),
                    # Absent means trained with the flat prior and the random
                    # projections; their parameters are what the file holds.
                    locality_mode=a("locality_mode", "legacy"),
                    locality_wide=a("locality_wide", 8.0),
                    cross_window=a("cross_window", 0),
                    corr_mode=a("corr_mode", "legacy"),
                    # Absent means a checkpoint from before the flag existed,
                    # and every one of those was causal.
                    causal=bool(a("causal", 1)),
                    costvol=bool(a("costvol", 0)),
                    cv_k=a("cv_k", 16),
                    cv_support=a("cv_support", 8),
                    # Absent means a checkpoint from before the match could be
                    # trained: no head in the file, the raw cosine in the model.
                    match_learn=bool(a("match_learn", 0)),
                    match_dim=a("match_dim", 64),
                    match_topk=a("match_topk", 1),
                    fine=bool(a("fine", 0)),
                    fine_levels=a("fine_levels", 4),
                    fine_radius=a("fine_radius", 3),
                    fine_xyz=bool(a("fine_xyz", 0)),
                    fine_waves=a("fine_waves", 0),
                    fine_width=a("fine_width", 64),
                    fine_neigh=a("fine_neigh", 0),
                    fine_prev=bool(a("fine_prev", 0)),
                    # Absent means a checkpoint from before the head existed.
                    visibility=a("vis_loss", 0.0) > 0,
                    vis_feedback=bool(a("vis_feedback", 0)),
                    motion_waves=bool(a("motion_waves", 0)),
                    buddies=bool(a("buddies", 0)),
                    buddy_bias=bool(a("buddy_bias", 0)),
                    virtual=a("virtual", 0),
                    updater=a("updater", "pointdit"))

