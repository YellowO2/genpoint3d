"""Build the model a run's settings describe."""

from genpoint3d.train.model.PointDiT import PointDiT


def build_model(settings, feat_dim: int) -> PointDiT:
    """The model of a run's settings, or of a checkpoint's
    (`Settings(**ckpt["args"])`); a setting the checkpoint predates is off."""
    a = lambda name, default: getattr(settings, name, default)
    return PointDiT(dim=settings.dim, depth=settings.depth, num_heads=settings.heads, feat_dim=feat_dim,
                    cross_window=a("cross_window", 49), fine_neigh=a("fine_neigh", 32),
                    visibility=a("vis_loss", 0.0) > 0,
                    group_feedback=a("group_feedback", 0),
                    feedback_scale=a("feedback_scale", 1.0),
                    bidirectional="b" in a("schedule", ""),
                    rope_current=bool(a("rope_current", 0)),
                    rigid_proposal=bool(a("rigid_proposal", 0)))
