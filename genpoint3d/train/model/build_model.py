"""Build the model a run's settings describe."""

from genpoint3d.train.model.PointDiT import PointDiT
from genpoint3d.train.model.WorldModel import WorldModel


def build_model(settings, feat_dim: int) -> PointDiT | WorldModel:
    """The model of a run's settings, or of a checkpoint's
    (`Settings.saved(ckpt["args"])`); a setting the checkpoint predates is off."""
    a = lambda name, default: getattr(settings, name, default)
    if a("world", 0):
        return WorldModel(dim=settings.dim, depth=settings.depth, num_heads=settings.heads, feat_dim=feat_dim,
                          cross_window=settings.cross_window, fine_neigh=settings.fine_neigh,
                          group_dim=settings.group_feedback, feedback_scale=settings.feedback_scale, patch=settings.patch)
    return PointDiT(dim=settings.dim, depth=settings.depth, num_heads=settings.heads, feat_dim=feat_dim,
                    cross_window=a("cross_window", 49), fine_neigh=a("fine_neigh", 32),
                    visibility=a("vis_loss", 0.0) > 0,
                    group_feedback=a("group_feedback", 0),
                    feedback_scale=a("feedback_scale", 1.0),
                    rigid_proposal=bool(a("rigid_proposal", 0)),
                    fit_robust=a("fit_robust", 0),
                    fit_fresh=a("fit_fresh", 0.0),
                    patch=a("patch", 0),
                    centre=a("centre_loss", 0.0) > 0,
                    follow_fit=bool(a("follow_fit", 0)))
