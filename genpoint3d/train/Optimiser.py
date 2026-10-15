"""Step the weights: AdamW on a warmup-then-cosine schedule, and their average."""

import logging

import torch

from genpoint3d.train.Hardware import Hardware

log = logging.getLogger(__name__)


class Optimiser:
    """AdamW, its learning-rate schedule, the gradient clip, and the EMA.

    The EMA is what gets scored and saved: `swap_in_average` puts it in the
    model for that, and `swap_out_average` puts the live weights back.
    """

    def __init__(self, model, settings, hardware: Hardware) -> None:
        self.clip = settings.clip
        self.opt = torch.optim.AdamW(model.parameters(), lr=settings.lr,
                                     weight_decay=settings.wdecay)
        self.ema = _Average(model, settings.ema) if settings.ema > 0 else None
        warmup = max(1, int(settings.steps * settings.warmup_frac))
        warm = torch.optim.lr_scheduler.LinearLR(self.opt, 0.01, 1.0, warmup)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, settings.steps - warmup)
        self.sched = torch.optim.lr_scheduler.SequentialLR(self.opt, [warm, cos], [warmup])
        self.live = None
        log.info(f"lr {settings.lr:.1e} | warmup {warmup} steps ({settings.warmup_frac:.0%})"
                 f" | wdecay {settings.wdecay:g} | amp {'bf16' if hardware.amp else 'off'}")

    def step(self, model) -> None:
        """One optimiser step on the gradient the micro-batches left."""
        if self.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        if self.ema is not None:
            self.ema.update(model)
        self.sched.step()

    def lr(self) -> float:
        return self.sched.get_last_lr()[0]

    def swap_in_average(self, model) -> None:
        """Score the averaged weights, which are also what gets saved --
        evaluating the live weights and shipping the EMA ones would report a
        number for a checkpoint nobody has."""
        if self.ema is not None:
            self.live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(self.ema.state_dict(model))

    def swap_out_average(self, model) -> None:
        """Training continues on the live weights."""
        if self.live is not None:
            model.load_state_dict(self.live)
            self.live = None


class _Average:
    """A slowly-following copy of the weights, used for evaluation.

    SGD leaves the weights jittering around a minimum rather than sitting in
    it, and an average over recent steps lands closer to the middle than any
    single step does. Diffusion models gain more from this than most, and the
    paper lists EMA among its training ingredients (section 4.1).

    `decay` warms up: averaging over 1000 steps is meaningless at step 10, when
    the initialisation would still dominate.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.n = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v.detach().float(), 1 - d)

    def state_dict(self, model) -> dict:
        """The averaged weights, in the model's own dtypes."""
        sd = model.state_dict()
        return {k: (self.shadow[k].to(v.dtype) if k in self.shadow else v.clone())
                for k, v in sd.items()}
