"""Step the weights with AdamW, and keep their running average."""

import logging

import torch

from genpoint3d.train.Hardware import Hardware

log = logging.getLogger(__name__)


class Optimiser:
    """AdamW with warmup then cosine, the gradient clip, and the EMA of the
    weights, which is what gets scored and saved (`swap_in_average`)."""

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

    def step(self, model) -> float:
        """One step on the micro-batches' gradient; returns its norm before the clip."""
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip or float("inf")))
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        if self.ema is not None:
            self.ema.update(model)
        self.sched.step()
        return norm

    def state(self, model) -> dict:
        """Everything to go on from here (`restore`): the live weights, AdamW's
        moments, the schedule's place and the average. Called with the live
        weights in."""
        ema = None if self.ema is None else {"shadow": self.ema.shadow, "n": self.ema.n}
        return {"live": model.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "ema": ema}

    def restore(self, model, state: dict) -> None:
        model.load_state_dict(state["live"])
        self.opt.load_state_dict(state["opt"])
        self.sched.load_state_dict(state["sched"])
        if self.ema is not None:
            self.ema.shadow, self.ema.n = state["ema"]["shadow"], state["ema"]["n"]

    def lr(self) -> float:
        return self.sched.get_last_lr()[0]

    def swap_in_average(self, model) -> None:
        """Put the averaged weights in, to score and save them."""
        if self.ema is not None:
            self.live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(self.ema.state_dict(model))

    def swap_out_average(self, model) -> None:
        """Training continues on the live weights."""
        if self.live is not None:
            model.load_state_dict(self.live)
            self.live = None


class _Average:
    """A slowly following average of the weights, its decay warming up over
    the first steps so the initialisation does not dominate."""

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
