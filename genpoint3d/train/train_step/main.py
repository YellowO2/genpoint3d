"""One training step: learn from each micro-batch, then update the weights."""

from genpoint3d.train.Hardware import Hardware
from genpoint3d.train.Losses import Losses
from genpoint3d.train.Optimiser import Optimiser
from genpoint3d.train.train_step.learn import learn


def train_step(model, micro_batches, optimiser: Optimiser, settings,
               hardware: Hardware) -> list[Losses]:
    """Forward, backward and update, for `--accum` micro-batches.

    The gradient passes from `learn` to the optimiser through the model:
    each `learn` adds its gradient to the weights' `.grad`, and
    `optimiser.step` reads those, updates the weights and clears them.
    """
    losses = []
    for batch in micro_batches:
        losses.append(learn(model, batch, settings, hardware))
    optimiser.step(model)
    return losses
